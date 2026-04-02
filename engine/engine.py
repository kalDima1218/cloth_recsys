import os
import pickle
from collections import Counter
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return vec / norms


def _build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix.astype(np.float32))
    return index


class FashionEngine:
    def __init__(
        self,
        embeddings_npz: str,
        transactions_csv: str,
        min_user_transactions: int = 5,
        lookalike_pool: int = 50,
    ) -> None:
        self._lookalike_pool = lookalike_pool
        self._ranker = None

        self._load_item_embeddings(embeddings_npz)
        self._build_user_profiles(transactions_csv, min_user_transactions)

        print(
            f"[FashionEngine] Ready. "
            f"Items: {self.item_index.ntotal}, "
            f"Users: {self.user_index.ntotal}"
        )

    def _load_item_embeddings(self, npz_path: str) -> None:
        data = np.load(npz_path)
        embeddings: np.ndarray = data["embeddings"]
        article_ids: np.ndarray = data["article_ids"]

        self.embedding_dim: int = embeddings.shape[1]
        self._embeddings_f16: np.ndarray = embeddings.astype(np.float16)
        self._item_article_ids: np.ndarray = article_ids.astype(np.int64)
        self._art_id_to_idx: Dict[int, int] = {
            int(aid): i for i, aid in enumerate(article_ids)
        }
        self.item_index: faiss.IndexFlatIP = _build_faiss_index(embeddings.astype(np.float32))

    def _build_user_profiles(self, transactions_csv: str, min_transactions: int) -> None:
        df = pd.read_csv(
            transactions_csv,
            usecols=["customer_id", "article_id"],
            dtype={"article_id": np.int32},
        )

        counts = df.groupby("customer_id", sort=False).size()
        active_customers = counts[counts > min_transactions].index
        df = df[df["customer_id"].isin(active_customers)]

        print(f"[FashionEngine] Active customers: {len(active_customers):,} / {counts.shape[0]:,}")

        grouped = df.groupby("customer_id", sort=False)["article_id"]
        self._customer_purchases: Dict[str, np.ndarray] = {
            cid: arr.to_numpy(dtype=np.int32) for cid, arr in grouped
        }

        known_ids = set(self._art_id_to_idx.keys())
        user_ids_list: List[str] = []
        user_vecs_list: List[np.ndarray] = []

        for cid, arts in self._customer_purchases.items():
            valid = [a for a in arts if a in known_ids]
            if not valid:
                continue
            idxs = [self._art_id_to_idx[a] for a in valid]
            vecs = self._embeddings_f16[idxs].astype(np.float32)
            user_ids_list.append(cid)
            user_vecs_list.append(_l2_normalize(vecs.mean(axis=0)))

        self._user_ids: np.ndarray = np.array(user_ids_list)
        user_matrix = np.stack(user_vecs_list, axis=0).astype(np.float32)
        self.user_index: faiss.IndexFlatIP = _build_faiss_index(user_matrix)

    def set_ranker(self, ranker) -> None:
        self._ranker = ranker

    def _search_items_scored(self, query_vec: np.ndarray, top_k: int) -> tuple:
        vec = _l2_normalize(np.array(query_vec, dtype=np.float32).reshape(1, -1)).astype(np.float32)
        scores, indices = self.item_index.search(vec, top_k)
        article_ids = []
        faiss_scores = []
        for j, i in enumerate(indices[0]):
            if i >= 0:
                article_ids.append(int(self._item_article_ids[i]))
                faiss_scores.append(float(scores[0][j]))
        return article_ids, faiss_scores

    def get_visually_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> List[int]:
        article_ids, _ = self._search_items_scored(query_embedding, top_k)
        return article_ids

    def get_ranked_recommendations(
        self,
        liked_ids: List[int],
        top_k: int = 10,
        retrieval_k: int = 200,
        customer_id: Optional[str] = None,
    ) -> List[int]:
        valid_liked = [aid for aid in liked_ids if aid in self._art_id_to_idx]
        if not valid_liked:
            return []

        idxs = [self._art_id_to_idx[aid] for aid in valid_liked]
        vecs = self._embeddings_f16[idxs].astype(np.float32)
        user_vec = _l2_normalize(vecs.mean(axis=0))

        cand_ids, cand_scores = self._search_items_scored(user_vec, retrieval_k)

        liked_set = set(liked_ids)
        pairs = [(aid, sc) for aid, sc in zip(cand_ids, cand_scores) if aid not in liked_set]

        if not pairs:
            return []

        if self._ranker is None:
            return [aid for aid, _ in pairs[:top_k]]

        return self._ranker.rank(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            customer_id=customer_id,
            liked_article_ids=liked_ids,
            top_k=top_k,
        )

    def get_lookalike_recommendations(self, liked_ids: List[int], top_k: int = 10) -> List[int]:
        valid_liked = [aid for aid in liked_ids if aid in self._art_id_to_idx]
        if not valid_liked:
            return []

        idxs = [self._art_id_to_idx[aid] for aid in valid_liked]
        vecs = self._embeddings_f16[idxs].astype(np.float32)
        user_vec = _l2_normalize(vecs.mean(axis=0)).reshape(1, -1).astype(np.float32)

        pool = min(self._lookalike_pool, self.user_index.ntotal)
        _, neighbor_indices = self.user_index.search(user_vec, pool)
        neighbor_cids = self._user_ids[neighbor_indices[0]]

        liked_set = set(liked_ids)
        counter: Counter = Counter()
        for cid in neighbor_cids:
            purchases = self._customer_purchases.get(str(cid))
            if purchases is None:
                continue
            for art in purchases:
                art_int = int(art)
                if art_int not in liked_set:
                    counter[art_int] += 1

        return [art for art, _ in counter.most_common(top_k)]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.item_index, os.path.join(directory, "item.faiss"))
        faiss.write_index(self.user_index, os.path.join(directory, "user.faiss"))

        state = {
            "embedding_dim": self.embedding_dim,
            "embeddings_f16": self._embeddings_f16,
            "item_article_ids": self._item_article_ids,
            "art_id_to_idx": self._art_id_to_idx,
            "user_ids": self._user_ids,
            "customer_purchases": self._customer_purchases,
            "lookalike_pool": self._lookalike_pool,
        }
        with open(os.path.join(directory, "state.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[FashionEngine] Saved to: {directory}")

    @classmethod
    def load(cls, directory: str) -> "FashionEngine":
        with open(os.path.join(directory, "state.pkl"), "rb") as f:
            state = pickle.load(f)

        engine = cls.__new__(cls)
        engine.embedding_dim       = state["embedding_dim"]
        engine._embeddings_f16     = state["embeddings_f16"]
        engine._item_article_ids   = state["item_article_ids"]
        engine._art_id_to_idx      = state["art_id_to_idx"]
        engine._user_ids           = state["user_ids"]
        engine._customer_purchases = state["customer_purchases"]
        engine._lookalike_pool     = state["lookalike_pool"]
        engine._ranker             = None

        engine.item_index = faiss.read_index(os.path.join(directory, "item.faiss"))
        engine.user_index = faiss.read_index(os.path.join(directory, "user.faiss"))

        print(
            f"[FashionEngine] Loaded from {directory}. "
            f"Items: {engine.item_index.ntotal}, "
            f"Users: {engine.user_index.ntotal}"
        )
        return engine


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python engine.py <embeddings.npz> <transactions_train.csv> [--save <dir>]")
        sys.exit(1)

    engine = FashionEngine(sys.argv[1], sys.argv[2])

    if "--save" in sys.argv:
        engine.save(sys.argv[sys.argv.index("--save") + 1])

    dummy = np.random.randn(engine.embedding_dim).astype(np.float32)
    print("Visually similar:", engine.get_visually_similar(dummy, top_k=5))

    seed_ids = [int(engine._item_article_ids[i]) for i in range(3)]
    print("Look-alike:", engine.get_lookalike_recommendations(seed_ids, top_k=5))
