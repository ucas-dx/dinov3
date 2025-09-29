"""Dataset utilities for training on unlabeled image folders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


def _default_extensions() -> tuple[str, ...]:
    return (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".webp",
    )


class UnlabeledImageFolder(Dataset):
    """Simple dataset that scans a directory of unlabeled images.

    The dataset recursively walks ``root`` (unless ``recursive`` is ``False``)
    and keeps every file that matches one of the provided ``extensions``.
    Each item returns the decoded PIL image together with the original file
    path.  Downstream code can provide a ``target_transform`` to map the path
    to an empty tuple so that the DINOv3 training loop ignores labels.
    """

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder does not exist: {self.root}")

        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(ext.lower() for ext in (extensions or _default_extensions()))
        self.recursive = recursive
        self.samples = self._gather_samples()
        if not self.samples:
            raise RuntimeError(f"No image files with extensions {self.extensions} found in {self.root}")

    def _gather_samples(self) -> list[Path]:
        if self.recursive:
            iterator: Iterable[Path] = self.root.rglob("*")
        else:
            iterator = self.root.glob("*")
        samples = [p for p in iterator if p.is_file() and p.suffix.lower() in self.extensions]
        samples.sort()
        return samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        target = str(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}(root={self.root!s}, recursive={self.recursive})"
        return f"{head}, num_samples={len(self.samples)}"
