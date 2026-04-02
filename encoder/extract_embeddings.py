import argparse
import os

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import FashionEncoder


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CatalogDataset(Dataset):
    def __init__(self, image_root: str):
        self.transform = _build_transform()
        images_dir = os.path.join(image_root, "images")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"images/ not found at: {images_dir}")

        self.records: list[tuple[str, int]] = []
        for sub in sorted(os.listdir(images_dir)):
            sub_path = os.path.join(images_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            for fname in os.listdir(sub_path):
                if fname.lower().endswith(".jpg"):
                    self.records.append((
                        os.path.join(sub_path, fname),
                        int(os.path.splitext(fname)[0]),
                    ))

        print(f"[CatalogDataset] {len(self.records)} images found")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, article_id = self.records[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), article_id


def extract(checkpoint_path: str, image_root: str, output_dir: str, batch_size: int) -> None:
    device  = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]
    embedding_dim: int = cfg["embedding_dim"]

    model = FashionEncoder(embedding_dim=embedding_dim, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dataloader = DataLoader(
        CatalogDataset(image_root=image_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings:  list[np.ndarray] = []
    all_article_ids: list[int]        = []

    with torch.no_grad():
        for batch_imgs, batch_ids in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                embeddings = model(batch_imgs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_article_ids.extend(batch_ids.numpy().tolist())

    embeddings_array  = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    article_ids_array = np.array(all_article_ids, dtype=np.int64)

    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "embeddings.npz")
    np.savez(npz_path, embeddings=embeddings_array, article_ids=article_ids_array)
    print(f"Saved: {npz_path}")

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_array)
    index_path = os.path.join(output_dir, "embeddings.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved: {index_path} ({index.ntotal} vectors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", default="..")
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    extract(
        checkpoint_path=args.checkpoint,
        image_root=args.image_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
