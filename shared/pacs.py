"""PACS data access shared by Tasks 2 and 3.

The PACS parquet (Hugging Face mirror ``flwrlabs/pacs``) is decoded once, every image is resized to
256x256 (the manual's resize) and cached as a uint8 tensor per domain under ``data/pacs/`` (git-ignored).

Target-label protection
-----------------------
Sketch class labels are written to a *separate* file and are only returned by
:func:`load_target_labels`, which refuses to run unless a freeze manifest (written after every
checkpoint and prediction has been fixed) exists and still matches the files on disk. Every call is
appended to ``data/pacs/target_label_access_log.jsonl`` so label access is auditable.
Training code only ever receives Sketch *images* via :func:`load_target_images`.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "pacs"
PARQUET = DATA_DIR / "pacs.parquet"
PARQUET_URL = "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/data/train-00000-of-00001.parquet"
PARQUET_SHA256 = "4fc041ee92eec6043fe6e2859e8bdd138e5f958bc621afd153879812cbe65ff5"

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
TARGET_DOMAIN = "sketch"
SOURCE_DOMAINS = [d for d in DOMAINS if d != TARGET_DOMAIN]
EXPECTED_COUNTS = {"photo": 1670, "art_painting": 2048, "cartoon": 2344, "sketch": 3929}  # PACS paper
RESIZE = 256

_LABEL_DIR = DATA_DIR / "_target_labels_do_not_load_during_training"
_ACCESS_LOG = DATA_DIR / "target_label_access_log.jsonl"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _images_path(domain: str) -> Path:
    return DATA_DIR / f"{domain}_images.pt"


def prepare_pacs(force: bool = False) -> dict:
    """Download (if missing), verify, decode, resize to 256x256 and cache every domain.

    Returns a summary dict (image counts, source class counts, class names). Sketch class counts are
    deliberately *not* computed or returned.
    """
    import pyarrow.parquet as pq

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PARQUET.exists():
        print("downloading PACS parquet (191 MB) ...")
        urllib.request.urlretrieve(PARQUET_URL, PARQUET)
    digest = _sha256(PARQUET)
    if digest != PARQUET_SHA256:
        raise RuntimeError(f"PACS parquet checksum mismatch: {digest}")

    table = pq.read_table(PARQUET)
    meta = json.loads(table.schema.metadata[b"huggingface"])
    classes = meta["info"]["features"]["label"]["names"]
    domains = table.column("domain").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    images = table.column("image")

    resize = T.Compose([T.Resize((RESIZE, RESIZE)), T.PILToTensor()])   # bilinear (torchvision default)
    summary = {"parquet_sha256": digest, "classes": classes, "resize": RESIZE, "counts": {}, "source_class_counts": {}}
    for dom in DOMAINS:
        rows = np.flatnonzero(np.asarray(domains) == dom)             # parquet row order is kept
        if len(rows) != EXPECTED_COUNTS[dom]:
            raise RuntimeError(f"{dom}: {len(rows)} images, expected {EXPECTED_COUNTS[dom]}")
        summary["counts"][dom] = int(len(rows))
        if _images_path(dom).exists() and not force:
            continue
        imgs = torch.stack([resize(Image.open(io.BytesIO(images[int(r)].as_py()["bytes"])).convert("RGB")) for r in rows])
        torch.save({"images": imgs, "parquet_rows": torch.as_tensor(rows)}, _images_path(dom))
        if dom == TARGET_DOMAIN:
            _LABEL_DIR.mkdir(exist_ok=True)
            torch.save({"labels": torch.as_tensor(labels[rows]), "parquet_rows": torch.as_tensor(rows)},
                       _LABEL_DIR / f"{dom}_labels.pt")
        else:
            torch.save({"labels": torch.as_tensor(labels[rows]), "parquet_rows": torch.as_tensor(rows)},
                       DATA_DIR / f"{dom}_labels.pt")
        print(f"cached {dom}: {tuple(imgs.shape)}")
    for dom in SOURCE_DOMAINS:
        y = torch.load(DATA_DIR / f"{dom}_labels.pt")["labels"].numpy()
        summary["source_class_counts"][dom] = {c: int((y == k).sum()) for k, c in enumerate(classes)}
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def class_names() -> list[str]:
    return json.loads((DATA_DIR / "summary.json").read_text())["classes"]


def load_source_domain(domain: str) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 images [N,3,256,256] and int64 labels for a labelled source domain."""
    if domain not in SOURCE_DOMAINS:
        raise ValueError(f"{domain!r} is not a source domain; target labels are only available via load_target_labels()")
    return torch.load(_images_path(domain))["images"], torch.load(DATA_DIR / f"{domain}_labels.pt")["labels"]


def load_target_images(domain: str = TARGET_DOMAIN) -> torch.Tensor:
    """Unlabelled target images (uint8 [N,3,256,256]). No labels are loaded."""
    if domain != TARGET_DOMAIN:
        raise ValueError(f"{domain!r} is not the target domain")
    return torch.load(_images_path(domain))["images"]


def file_sha256(path: Path) -> str:
    return _sha256(Path(path))


def write_freeze_manifest(path: Path, files: list[Path], note: str) -> dict:
    """Record the SHA-256 of every checkpoint/prediction file that must be fixed before target labels are read."""
    files = [Path(f) for f in files]
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                "files": {str(f.relative_to(REPO)).replace("\\", "/"): _sha256(f) for f in files}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1))
    return manifest


def load_target_labels(freeze_manifest: Path, domain: str = TARGET_DOMAIN) -> torch.Tensor:
    """Return target labels ONLY if the freeze manifest exists and every file it lists is unchanged.

    This is the single entry point to Sketch labels. Each successful call is logged.
    """
    freeze_manifest = Path(freeze_manifest)
    if not freeze_manifest.exists():
        raise PermissionError("No freeze manifest: fix every checkpoint and prediction before loading target labels.")
    manifest = json.loads(freeze_manifest.read_text())
    for rel, digest in manifest["files"].items():
        f = REPO / rel
        if not f.exists() or _sha256(f) != digest:
            raise PermissionError(f"Frozen file changed or missing since the manifest was written: {rel}")
    caller = traceback.extract_stack(limit=3)[0]
    with open(_ACCESS_LOG, "a") as log:
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "domain": domain,
                              "manifest": str(freeze_manifest.relative_to(REPO)).replace("\\", "/"),
                              "caller": f"{caller.filename}:{caller.lineno}"}) + "\n")
    return torch.load(_LABEL_DIR / f"{domain}_labels.pt")["labels"]
