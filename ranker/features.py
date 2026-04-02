import pickle
from collections import Counter
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    "faiss_score",
    "item_product_type",
    "item_colour_group",
    "item_department",
    "item_section",
    "item_garment_group",
    "item_global_freq",
    "user_cat_purchase_count",
    "user_dept_purchase_count",
    "user_avg_price",
    "item_avg_price",
    "price_diff",
]

CAT_FEATURE_INDICES: List[int] = [1, 2, 3, 4, 5]


class LookupTables(NamedTuple):
    item_global_freq_map: Dict[int, int]
    item_avg_price_map: Dict[int, float]
    article_meta_map: Dict[int, dict]
    user_avg_price_map: Dict[str, float]
    user_cat_counts_map: Dict[str, Counter]
    user_dept_counts_map: Dict[str, Counter]
    global_avg_price: float


def build_lookup_tables(articles_df: pd.DataFrame, transactions_df: pd.DataFrame) -> LookupTables:
    cat_cols = ["product_type_name", "colour_group_name", "department_name", "section_name", "garment_group_name"]
    article_meta_map: Dict[int, dict] = {}
    for row in articles_df[["article_id"] + cat_cols].itertuples(index=False):
        article_meta_map[int(row.article_id)] = {
            "product_type":  str(row.product_type_name)  or "UNKNOWN",
            "colour_group":  str(row.colour_group_name)  or "UNKNOWN",
            "department":    str(row.department_name)    or "UNKNOWN",
            "section":       str(row.section_name)       or "UNKNOWN",
            "garment_group": str(row.garment_group_name) or "UNKNOWN",
        }

    item_agg = transactions_df.groupby("article_id", sort=False)["price"].agg(["count", "mean"])
    item_global_freq_map: Dict[int, int] = {int(aid): int(cnt) for aid, cnt in item_agg["count"].items()}
    item_avg_price_map: Dict[int, float] = {int(aid): float(m) for aid, m in item_agg["mean"].items()}
    global_avg_price = float(transactions_df["price"].mean())

    user_avg_price_map: Dict[str, float] = {
        str(uid): float(v)
        for uid, v in transactions_df.groupby("customer_id", sort=False)["price"].mean().items()
    }

    meta_cols = articles_df[["article_id", "product_type_name", "department_name"]].copy()
    meta_cols["article_id"] = meta_cols["article_id"].astype(int)
    txn_with_meta = transactions_df[["customer_id", "article_id"]].merge(meta_cols, on="article_id", how="left")

    user_cat_counts_map: Dict[str, Counter] = {}
    for row in txn_with_meta.groupby(["customer_id", "product_type_name"], sort=False).size().reset_index(name="cnt").itertuples(index=False):
        uid = str(row.customer_id)
        if uid not in user_cat_counts_map:
            user_cat_counts_map[uid] = Counter()
        user_cat_counts_map[uid][str(row.product_type_name)] = int(row.cnt)

    user_dept_counts_map: Dict[str, Counter] = {}
    for row in txn_with_meta.groupby(["customer_id", "department_name"], sort=False).size().reset_index(name="cnt").itertuples(index=False):
        uid = str(row.customer_id)
        if uid not in user_dept_counts_map:
            user_dept_counts_map[uid] = Counter()
        user_dept_counts_map[uid][str(row.department_name)] = int(row.cnt)

    return LookupTables(
        item_global_freq_map=item_global_freq_map,
        item_avg_price_map=item_avg_price_map,
        article_meta_map=article_meta_map,
        user_avg_price_map=user_avg_price_map,
        user_cat_counts_map=user_cat_counts_map,
        user_dept_counts_map=user_dept_counts_map,
        global_avg_price=global_avg_price,
    )


def extract_row(
    customer_id: Optional[str],
    article_id: int,
    faiss_score: float,
    tables: LookupTables,
    liked_article_ids: Optional[List[int]] = None,
) -> list:
    aid = int(article_id)
    meta = tables.article_meta_map.get(aid, {})

    product_type  = meta.get("product_type",  "UNKNOWN")
    colour_group  = meta.get("colour_group",  "UNKNOWN")
    department    = meta.get("department",    "UNKNOWN")
    section       = meta.get("section",       "UNKNOWN")
    garment_group = meta.get("garment_group", "UNKNOWN")

    item_global_freq = float(np.log1p(tables.item_global_freq_map.get(aid, 0)))
    item_avg_price = tables.item_avg_price_map.get(aid, tables.global_avg_price)

    if customer_id is not None:
        uid = str(customer_id)
        user_avg_price  = tables.user_avg_price_map.get(uid, tables.global_avg_price)
        user_cat_count  = float(tables.user_cat_counts_map.get(uid, {}).get(product_type, 0))
        user_dept_count = float(tables.user_dept_counts_map.get(uid, {}).get(department, 0))
    else:
        user_avg_price  = tables.global_avg_price
        user_cat_count  = 0.0
        user_dept_count = 0.0
        if liked_article_ids:
            for la in liked_article_ids:
                la_meta = tables.article_meta_map.get(int(la), {})
                if la_meta.get("product_type") == product_type:
                    user_cat_count += 1.0
                if la_meta.get("department") == department:
                    user_dept_count += 1.0
            liked_prices = [
                tables.item_avg_price_map[int(la)]
                for la in liked_article_ids
                if int(la) in tables.item_avg_price_map
            ]
            if liked_prices:
                user_avg_price = float(np.mean(liked_prices))

    return [
        float(faiss_score),
        product_type,
        colour_group,
        department,
        section,
        garment_group,
        item_global_freq,
        user_cat_count,
        user_dept_count,
        float(user_avg_price),
        float(item_avg_price),
        float(item_avg_price) - float(user_avg_price),
    ]


def extract_batch(
    customer_id: Optional[str],
    article_ids: List[int],
    faiss_scores: List[float],
    tables: LookupTables,
    liked_article_ids: Optional[List[int]] = None,
) -> np.ndarray:
    rows = [
        extract_row(customer_id, aid, score, tables, liked_article_ids)
        for aid, score in zip(article_ids, faiss_scores)
    ]
    return np.array(rows, dtype=object)


def save_tables(tables: LookupTables, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tables(path: str) -> LookupTables:
    with open(path, "rb") as f:
        return pickle.load(f)
