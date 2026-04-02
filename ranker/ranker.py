from typing import List, Optional

import numpy as np
from catboost import CatBoostRanker

from features import extract_batch, load_tables, LookupTables


class FashionRanker:
    def __init__(self, model_path: str, lookup_path: str) -> None:
        self._model = CatBoostRanker()
        self._model.load_model(model_path)
        self._tables: LookupTables = load_tables(lookup_path)

    def rank(
        self,
        article_ids: List[int],
        faiss_scores: List[float],
        customer_id: Optional[str] = None,
        liked_article_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None,
    ) -> List[int]:
        if not article_ids:
            return []

        X = extract_batch(customer_id, article_ids, faiss_scores, self._tables, liked_article_ids)
        scores = self._model.predict(X)
        order = np.argsort(scores)[::-1]
        ranked = [article_ids[i] for i in order]
        return ranked[:top_k] if top_k is not None else ranked
