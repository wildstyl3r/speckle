import io
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .config import DatasetConfig

HF_LINKS = {
    ("Mnist", True): ["https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet"],
    ("Mnist", False): ["https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"],
    ("Cifar10", True): ["https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"],
    ("Cifar10", False): ["https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"],
    ("Stl10Unlabeled", True): [
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00000-of-00004-b74cd68890b47699.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00001-of-00004-316ae046518ebbac.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00002-of-00004-4180ec8df3024526.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00003-of-00004-15264226b952a419.parquet",
    ],
}
HF_LINKS[("Stl10Unlabeled", False)] = HF_LINKS[("Stl10Unlabeled", True)]

IMAGE_COLUMNS = {"Mnist": "image", "Cifar10": "img", "Stl10Unlabeled": "image"}
FILENAMES = {"Mnist": "mnist", "Cifar10": "cifar10", "Stl10Unlabeled": "stl10_unlabeled"}
HWC = {
    "Mnist": (28, 28, 1),
    "Cifar10": (32, 32, 3),
    "SplitLine": (64, 64, 1),
    "SmoothedSplitLine": (64, 64, 1),
    "Stl10Unlabeled": (96, 96, 3),
}


def hwc(dataset: str):
    return HWC[dataset]


def employ(cfg: DatasetConfig, train: bool) -> torch.Tensor:
    dataset = cfg.dataset
    if dataset in ("SplitLine", "SmoothedSplitLine"):
        images = generate(dataset, train)
        if cfg.max_images > 0:
            images = images[: cfg.max_images]
        return images
    return load_or_download(dataset, train, cfg.max_images)


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
        return (distances >= 0).to(torch.uint8)
    raise NotImplementedError(f"dataset {dataset!r} generation is not implemented (todo in Rust source)")


def _download(url: str, file_path: Path, attempts: int = 5) -> None:
    tmp_path = file_path.with_suffix(file_path.suffix + ".part")
    for attempt in range(attempts):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        request = urllib.request.Request(url)
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(request) as response:
                resume = getattr(response, "status", 200) == 206 and resume_from > 0
                content_length = int(response.headers.get("Content-Length") or 0)
                with open(tmp_path, "ab" if resume else "wb") as f:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
            expected = resume_from + content_length if resume else content_length
            if expected and tmp_path.stat().st_size != expected:
                raise IOError(f"incomplete download: {tmp_path.stat().st_size} of {expected} bytes")
            tmp_path.rename(file_path)
            return
        except Exception as error:
            print(f"download interrupted ({error}); retrying ({attempt + 1}/{attempts})")
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to download {url} after {attempts} attempts")


def _decode(table, column: str, h: int, w: int, c: int, max_rows: int = -1) -> torch.Tensor:
    image_struct = table.column(column).combine_chunks()
    bytes_col = image_struct.field("bytes")
    n = table.num_rows if max_rows <= 0 else min(max_rows, table.num_rows)
    convert_mode = "L" if c == 1 else "RGB"
    all_images = np.empty((n, h, w, c), dtype=np.uint8)
    for i in range(n):
        img = Image.open(io.BytesIO(bytes_col[i].as_py())).convert(convert_mode)
        arr = np.asarray(img, dtype=np.uint8)
        if c == 1:
            arr = arr[:, :, None]
        all_images[i] = arr
    return torch.from_numpy(all_images)


def load_or_download(dataset: str, train: bool, max_images: int = -1) -> torch.Tensor:
    h, w, c = HWC[dataset]
    column = IMAGE_COLUMNS[dataset]
    split_dir = Path("data") / ("train" if train else "test")
    stem = FILENAMES[dataset]
    urls = HF_LINKS[(dataset, train)]
    cap = max_images if max_images > 0 else None

    full_npy = split_dir / f"{stem}_full.npy"
    capped_npy = split_dir / f"{stem}_{cap}.npy" if cap is not None else full_npy
    if capped_npy.exists():
        print(f"Found decoded dataset cache at: {capped_npy}")
        return torch.from_numpy(np.load(capped_npy))
    if cap is not None and full_npy.exists():
        print(f"Found decoded dataset cache at: {full_npy}")
        return torch.from_numpy(np.load(full_npy))[:cap]

    chunks = []
    done = 0
    for i, url in enumerate(urls):
        if cap is not None and done >= cap:
            break
        file_path = split_dir / f"{stem}.parquet" if len(urls) == 1 else split_dir / stem / f"{i:05d}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            print(f"Found cached shard at: {file_path}")
        else:
            print(f"Downloading dataset from Hugging Face: {url}")
            _download(url, file_path)
        table = pq.read_table(file_path, columns=[column])
        want = -1 if cap is None else cap - done
        chunk = _decode(table, column, h, w, c, want)
        chunks.append(chunk)
        done += chunk.shape[0]
    images = torch.cat(chunks)
    if len(urls) > 1:
        np.save(full_npy if cap is None else capped_npy, images.numpy())
    return images
