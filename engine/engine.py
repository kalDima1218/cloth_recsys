import os
import pickle
import faiss
from collections import Counter
from typing import Dict, List

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
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix.astype(np.float32))
    return index


class FashionEngine:
    _LOOKALIKE_POOL = 50

    def __init__(
        self,
        embeddings_npz: str,
        transactions_csv: str,
        min_user_transactions: int = 5,
        lookalike_pool: int = 50,
    ) -> None:
        self._lookalike_pool = lookalike_pool
        self._load_item_embeddings(embeddings_npz)
        self._build_user_profiles(transactions_csv, min_user_transactions)

    def _load_item_embeddings(self, npz_path: str) -> None:
        data = np.load(npz_path)
        embeddings: np.ndarray = data["embeddings"]
        article_ids: np.ndarray = data["article_ids"]

        self.embedding_dim: int = embeddings.shape[1]
        self._embeddings_f16: np.ndarray = embeddings.astype(np.float16)
        self._item_article_ids: np.ndarray = article_ids.astype(np.int64)
        self._art_id_to_idx: Dict[int, int] = {int(aid): i for i, aid in enumerate(article_ids)}
        self.item_index: faiss.IndexFlatIP = _build_faiss_index(embeddings.astype(np.float32))

    def _build_user_profiles(self, transactions_csv: str, min_transactions: int) -> None:
        df = pd.read_csv(transactions_csv, usecols=["customer_id", "article_id"], dtype={"article_id": np.int32})

        counts = df.groupby("customer_id", sort=False).size()
        active_customers = counts[counts > min_transactions].index
        df = df[df["customer_id"].isin(active_customers)]

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
            vecs = self._embeddings_f16[[self._art_id_to_idx[a] for a in valid]].astype(np.float32)
            user_ids_list.append(cid)
            user_vecs_list.append(_l2_normalize(vecs.mean(axis=0)))

        self._user_ids: np.ndarray = np.array(user_ids_list)
        self.user_index: faiss.IndexFlatIP = _build_faiss_index(
            np.stack(user_vecs_list, axis=0).astype(np.float32)
        )

    def get_visually_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> List[int]:
        vec = _l2_normalize(np.array(query_embedding, dtype=np.float32).reshape(1, -1)).astype(np.float32)
        _, indices = self.item_index.search(vec, top_k)
        return [int(self._item_article_ids[i]) for i in indices[0] if i >= 0]

    def get_lookalike_recommendations(self, liked_ids: List[int], top_k: int = 10) -> List[int]:
        valid_liked = [aid for aid in liked_ids if aid in self._art_id_to_idx]
        if not valid_liked:
            return []

        vecs = self._embeddings_f16[[self._art_id_to_idx[aid] for aid in valid_liked]].astype(np.float32)
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
        engine.item_index = faiss.read_index(os.path.join(directory, "item.faiss"))
        engine.user_index = faiss.read_index(os.path.join(directory, "user.faiss"))

        return engine
