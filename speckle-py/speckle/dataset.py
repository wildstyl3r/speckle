import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .config import DatasetConfig

HF_LINKS = {
    ("Mnist", True): "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet",
    ("Mnist", False): "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet",
    ("Cifar10", True): "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    ("Cifar10", False): "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
}

IMAGE_COLUMNS = {"Mnist": "image", "Cifar10": "img"}
FILENAMES = {"Mnist": "mnist.parquet", "Cifar10": "cifar10.parquet"}
HWC = {"Mnist": (28, 28, 1), "Cifar10": (32, 32, 3), "SplitLine": (64, 64, 1), "SmoothedSplitLine": (64, 64, 1)}


def hwc(dataset: str):
    return HWC[dataset]


def employ(cfg: DatasetConfig, train: bool) -> torch.Tensor:
    dataset = cfg.dataset
    if dataset in ("SplitLine", "SmoothedSplitLine"):
        return generate(dataset, train)
    return load_or_download(dataset, train)


def generate(dataset: str, train: bool) -> torch.Tensor:
    if dataset == "SplitLine":
        lin = torch.linspace(-1.0, 1.0, 64)
        x_grid = lin.unsqueeze(0).expand(64, 64).unsqueeze(0).unsqueeze(-1)
        y_grid = lin.unsqueeze(1).expand(64, 64).unsqueeze(0).unsqueeze(-1)
        offset = torch.rand(100000, 1, 1, 1) - 0.5
        tilt = torch.rand(100000, 1, 1, 1) * math.pi
        a = tilt.cos()
        b = tilt.sin()
        distances = a * x_grid + b * y_grid + offset
        return (distances >= 0).float()
    raise NotImplementedError(f"dataset {dataset!r} generation is not implemented (todo in Rust source)")


def load_or_download(dataset: str, train: bool) -> torch.Tensor:
    cache_dir = Path("data") / ("train" if train else "test")
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / FILENAMES[dataset]

    if file_path.exists():
        print(f"Found cached dataset at: {file_path}")
    else:
        print("Downloading dataset from Hugging Face...")
        url = HF_LINKS[(dataset, train)]
        tmp_path = file_path.with_suffix(file_path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(file_path)

    table = pq.read_table(file_path)
    image_struct = table.column(IMAGE_COLUMNS[dataset]).combine_chunks()
    bytes_col = image_struct.field("bytes")

    h, w, c = HWC[dataset]
    convert_mode = "L" if c == 1 else "RGB"
    all_images = np.empty((table.num_rows, h, w, c), dtype=np.float32)
    for i, raw in enumerate(bytes_col):
        img = Image.open(io.BytesIO(raw.as_py())).convert(convert_mode)
        arr = np.asarray(img, dtype=np.float32)
        if c == 1:
            arr = arr[:, :, None]
        all_images[i] = arr / 255.0
    return torch.from_numpy(all_images)
