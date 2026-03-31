import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from model import WEIGHTS


def build_transform():
    return WEIGHTS.transforms()


def article_id_to_path(article_id: int, image_root: str) -> str:
    s = str(int(article_id)).zfill(10)
    return os.path.join(image_root, "images", s[:3], f"{s}.jpg")


class HMDataset(Dataset):
    def __init__(
        self,
        articles_csv: str,
        image_root: str,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        random.seed(seed)
        self.image_root = image_root
        self.transform = transform or build_transform()

        df = pd.read_csv(articles_csv, usecols=["article_id", "product_code"])
        df["path"] = df["article_id"].apply(lambda aid: article_id_to_path(aid, image_root))
        mask_exists = df["path"].apply(os.path.exists)
        df = df[mask_exists].reset_index(drop=True)

        groups: Dict[int, List[int]] = {}
        for row in df.itertuples(index=False):
            groups.setdefault(row.product_code, []).append(row.article_id)

        self.product_groups: Dict[int, List[int]] = {
            pc: ids for pc, ids in groups.items() if len(ids) >= 2
        }

        self.valid_product_codes: List[int] = list(self.product_groups.keys())

    def __len__(self) -> int:
        return len(self.valid_product_codes)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        anchor_pc = self.valid_product_codes[idx]
        anchor_id, pos_id = random.sample(self.product_groups[anchor_pc], 2)

        neg_pc = anchor_pc
        while neg_pc == anchor_pc:
            neg_pc = random.choice(self.valid_product_codes)
        neg_id = random.choice(self.product_groups[neg_pc])

        return self._load(anchor_id), self._load(pos_id), self._load(neg_id)

    def _load(self, article_id: int) -> Tensor:
        path = article_id_to_path(article_id, self.image_root)
        img = Image.open(path).convert("RGB")
        return self.transform(img)
