import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT / "engine"))
sys.path.insert(0, str(_SCRIPT_DIR))

ENGINE_CACHE     = _PROJECT_ROOT / "engine_cache"
ARTICLES_CSV     = _PROJECT_ROOT / "articles.csv"
TRANSACTIONS_CSV = _PROJECT_ROOT / "transactions_train.csv"

TRAIN_CUTOFF = pd.Timestamp("2020-08-25")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _compute_user_vec(art_ids, art_id_to_idx, embeddings_f16) -> np.ndarray | None:
    valid = [int(a) for a in art_ids if int(a) in art_id_to_idx]
    if not valid:
        return None
    vecs = embeddings_f16[[art_id_to_idx[a] for a in valid]].astype(np.float32)
    return _l2_normalize(vecs.mean(axis=0))


def build_dataset(
    sampled_users: List[str],
    engine,
    train_txn: pd.DataFrame,
    test_txn: pd.DataFrame,
    tables,
    retrieval_k: int,
) -> pd.DataFrame:
    from features import FEATURE_NAMES, extract_row

    train_by_user = train_txn.groupby("customer_id")["article_id"].apply(list).to_dict()
    test_by_user  = test_txn.groupby("customer_id")["article_id"].apply(list).to_dict()

    rows = []
    for uid in tqdm(sampled_users, desc="Building dataset"):
        train_arts = train_by_user.get(str(uid), [])
        user_vec = _compute_user_vec(train_arts, engine._art_id_to_idx, engine._embeddings_f16)
        if user_vec is None:
            continue

        vec = user_vec.reshape(1, -1).astype(np.float32)
        raw_scores, raw_indices = engine.item_index.search(vec, retrieval_k)
        cand_ids    = [int(engine._item_article_ids[i]) for i in raw_indices[0] if i >= 0]
        cand_scores = [float(raw_scores[0][j]) for j, i in enumerate(raw_indices[0]) if i >= 0]
        cand_map    = dict(zip(cand_ids, cand_scores))

        positives = {int(a) for a in test_by_user.get(str(uid), []) if int(a) in engine._art_id_to_idx}
        if not positives:
            continue

        for pos_id in positives:
            if pos_id not in cand_map:
                pos_vec = engine._embeddings_f16[engine._art_id_to_idx[pos_id]].astype(np.float32)
                cand_map[pos_id] = float(np.dot(user_vec, pos_vec))

        for art_id, score in cand_map.items():
            rows.append([uid, int(art_id in positives)] + extract_row(uid, art_id, score, tables))

    return pd.DataFrame(rows, columns=["customer_id", "label"] + FEATURE_NAMES)


def train(df: pd.DataFrame, output_dir: Path, iterations: int) -> None:
    from catboost import CatBoostRanker, Pool
    from features import CAT_FEATURE_INDICES, FEATURE_NAMES

    user_ids = df["customer_id"].unique()
    n_val    = max(1, int(len(user_ids) * 0.2))
    val_users = set(user_ids[-n_val:])

    train_df = df[~df["customer_id"].isin(val_users)].sort_values("customer_id")
    val_df   = df[df["customer_id"].isin(val_users)].sort_values("customer_id")

    print(f"Train: {len(train_df):,} rows / {len(user_ids) - n_val:,} users")
    print(f"Val:   {len(val_df):,} rows / {n_val:,} users")

    def make_pool(part: pd.DataFrame) -> Pool:
        return Pool(
            data=part[FEATURE_NAMES],
            label=part["label"],
            group_id=part["customer_id"],
            cat_features=CAT_FEATURE_INDICES,
        )

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG",
        custom_metric=["NDCG", "PrecisionAt:top=10", "RecallAt:top=10"],
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        early_stopping_rounds=50,
        use_best_model=True,
        random_seed=42,
        verbose=100,
        thread_count=-1,
    )
    model.fit(make_pool(train_df), eval_set=make_pool(val_df))

    os.makedirs(output_dir, exist_ok=True)
    model.save_model(str(output_dir / "model.cbm"))
    print(f"Model saved: {output_dir / 'model.cbm'}")

    fi = pd.Series(
        model.get_feature_importance(type="PredictionValuesChange"),
        index=model.feature_names_,
    ).sort_values(ascending=False)
    print("\nFeature importances:\n", fi.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users",     type=int, default=20_000)
    parser.add_argument("--retrieval_k", type=int, default=200)
    parser.add_argument("--iterations",  type=int, default=1000)
    parser.add_argument("--output_dir",  type=str, default=str(_SCRIPT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    from engine import FashionEngine
    engine = FashionEngine.load(str(ENGINE_CACHE))

    articles_df = pd.read_csv(
        ARTICLES_CSV,
        usecols=["article_id", "product_type_name", "colour_group_name",
                 "department_name", "section_name", "garment_group_name"],
    )

    txn_df = pd.read_csv(
        TRANSACTIONS_CSV,
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype={"article_id": np.int32},
        parse_dates=["t_dat"],
    )

    train_txn = txn_df[txn_df["t_dat"] < TRAIN_CUTOFF].copy()
    test_txn  = txn_df[txn_df["t_dat"] >= TRAIN_CUTOFF].copy()
    del txn_df

    from features import build_lookup_tables, save_tables
    tables = build_lookup_tables(articles_df, train_txn)
    del articles_df

    os.makedirs(output_dir, exist_ok=True)
    save_tables(tables, str(output_dir / "lookup_meta.pkl"))

    train_counts = train_txn.groupby("customer_id").size().rename("train_cnt")
    test_users   = set(test_txn["customer_id"].unique())

    eligible = train_counts[
        (train_counts >= 5) & train_counts.index.isin(test_users)
    ].reset_index()
    eligible["quintile"] = pd.qcut(eligible["train_cnt"], q=5, labels=False, duplicates="drop")

    n_users = min(args.n_users, len(eligible))
    sampled_parts = []
    for q in eligible["quintile"].unique():
        subset = eligible[eligible["quintile"] == q]
        n_q = max(1, int(n_users * len(subset) / len(eligible)))
        sampled_parts.append(subset.sample(min(n_q, len(subset)), random_state=42))
    sampled_users = pd.concat(sampled_parts)["customer_id"].tolist()[:n_users]

    print(f"Sampled users: {len(sampled_users):,}")

    df = build_dataset(sampled_users, engine, train_txn, test_txn, tables, args.retrieval_k)
    del train_txn, test_txn

    print(f"Dataset: {len(df):,} rows, positives: {df['label'].sum():,} ({df['label'].mean():.2%})")

    train(df, output_dir, args.iterations)


if __name__ == "__main__":
    main()
