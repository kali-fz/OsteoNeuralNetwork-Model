"""Ingest explicitly verified normal radiographs into the ONNM dataset.

This command deliberately does not scrape clinical websites or infer a normal
diagnosis from page text. It accepts either a local source directory or a CSV
whose rows contain an image path, ``normal=1``, anatomy, and source/license
provenance. Kaggle MURA can be downloaded separately with the official Kaggle
CLI, then passed with ``--source-dir``; only ``negative`` studies are accepted.

Examples::

    python scripts/fetch_controls.py --source-dir D:\\datasets\\mura
    python scripts/fetch_controls.py --input-manifest controls_source.csv

The output manifest is consumed automatically by ``onnm.dataset.build_records``.
Images are copied, never moved, and an existing output is left untouched unless
``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

import _bootstrap  # noqa: F401 (path side effect)
from onnm.config import REPO_ROOT
from onnm.io_radiograph import RadiographReadError, read_radiograph

IMAGE_SUFFIXES = {".dcm", ".dicom", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
ANATOMIES = {"pelvis", "hip", "femur", "knee", "long_bone", "other"}


def _normalise_anatomy(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"hips": "hip", "pelvic": "pelvis", "femurs": "femur", "knees": "knee"}
    value = aliases.get(value, value)
    if value not in ANATOMIES:
        raise ValueError(f"anatomy must be one of {sorted(ANATOMIES)}, got {value!r}")
    return value


def _is_normal(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "normal", "negative"}


def _trim_border(image: Image.Image) -> Image.Image:
    """Remove only constant outer padding; keep markers and anatomy intact."""
    image = image.convert("L")
    background = Image.new("L", image.size, image.getpixel((0, 0)))
    bbox = ImageChops.difference(image, background).getbbox()
    if not bbox:
        return image
    left, top, right, bottom = bbox
    margin_x = max(2, round(image.width * 0.01))
    margin_y = max(2, round(image.height * 0.01))
    return image.crop((max(0, left - margin_x), max(0, top - margin_y),
                       min(image.width, right + margin_x), min(image.height, bottom + margin_y)))


def _write_image(source: Path, destination: Path) -> tuple[int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.suffix.lower() in {".dcm", ".dicom"}:
            array, _ = read_radiograph(source)
            values = np.asarray(array, dtype=np.float32)
            low, high = np.percentile(values, [1, 99])
            values = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
            image = Image.fromarray(np.round(values * 255).astype(np.uint8), mode="L")
        else:
            with Image.open(source) as opened:
                opened.verify()
            with Image.open(source) as opened:
                image = opened.copy()
        image = _trim_border(image)
        image.save(destination, format="PNG", optimize=True)
        with destination.open("rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        return image.width, image.height, digest
    except (OSError, ValueError, RadiographReadError) as exc:
        raise ValueError(f"cannot decode {source}: {exc}") from exc


def _rows_from_input(source_dir: Path | None, input_manifest: Path | None) -> list[dict[str, str]]:
    if input_manifest:
        with input_manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        required = {"image", "normal", "anatomy", "source", "license"}
        missing = required - set(rows[0]) if rows else required
        if missing:
            raise ValueError(f"{input_manifest} is missing columns: {sorted(missing)}")
        return [
            {**row, "image": str((input_manifest.parent / row["image"]).resolve())}
            for row in rows
        ]

    if not source_dir:
        raise ValueError("provide --source-dir or --input-manifest")
    rows = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # MURA encodes study labels in directory names: *_negative is normal.
        normal = "negative" if any("_negative" in part.lower() for part in path.parts) else ""
        anatomy = next((part.lower() for part in path.parts if part.lower() in ANATOMIES), "other")
        rows.append({"image": str(path), "normal": normal, "anatomy": anatomy,
                     "source": "local_source", "license": "user-supplied; verify before use"})
    return rows


def ingest(rows: list[dict[str, str]], output_root: Path, manifest_path: Path,
           seed: int, val_fraction: float, force: bool) -> tuple[int, dict[str, int]]:
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} exists; pass --force to rebuild controls")
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    rejected = 0
    for row in rows:
        if not _is_normal(row.get("normal", "")):
            rejected += 1
            continue
        source = Path(row["image"]).expanduser()
        if not source.is_file():
            rejected += 1
            continue
        anatomy = _normalise_anatomy(row["anatomy"])
        source_id = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
        destination = output_root / "staging" / f"control_{source_id}.png"
        if destination.exists() and not force:
            rejected += 1
            continue
        try:
            width, height, digest = _write_image(source, destination)
        except ValueError:
            rejected += 1
            continue
        if digest in seen:
            destination.unlink(missing_ok=True)
            rejected += 1
            continue
        seen.add(digest)
        accepted.append({
            "image": str(destination.relative_to(REPO_ROOT)),
            "image_id": f"control_{source_id}", "label": "0",
            "patient_id": f"control_{source_id}", "anatomy": anatomy, "split": "",
            "source": row["source"], "license": row["license"], "sha256": digest,
            "width": str(width), "height": str(height),
        })

    random.Random(seed).shuffle(accepted)
    val_count = round(len(accepted) * val_fraction)
    for index, row in enumerate(accepted):
        row["split"] = "val" if index < val_count else "train"
        target = output_root / row["split"] / "normal" / Path(row["image"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / row["image"], target)
        row["image"] = str(target.relative_to(REPO_ROOT))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image", "image_id", "label", "patient_id", "anatomy", "split", "source",
        "license", "sha256", "width", "height",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    counts = {
        anatomy: sum(row["anatomy"] == anatomy for row in accepted)
        for anatomy in sorted(ANATOMIES)
    }
    print(f"Accepted normal controls: {len(accepted)}")
    print(f"Rejected/non-normal/invalid/duplicate: {rejected}")
    print(f"Train normal: {len(accepted) - val_count}; val normal: {val_count}")
    print("Anatomy: " + ", ".join(f"{key}={value}" for key, value in counts.items() if value))
    print(f"Manifest: {manifest_path}")
    return len(accepted), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-dir", type=Path)
    group.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "configs" / "controls_manifest.csv"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        ingest(_rows_from_input(args.source_dir, args.input_manifest), args.output_root.resolve(),
               args.manifest.resolve(), args.seed, args.val_fraction, args.force)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())