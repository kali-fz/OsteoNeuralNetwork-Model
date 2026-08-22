"""Download and extract the BTXRD bone-tumor radiograph dataset.

    python scripts/download_btxrd.py

Source: figshare 10.6084/m9.figshare.27865398, published alongside
"A Radiograph Dataset for the Classification, Localization, and Segmentation of
Primary Bone Tumors" (Scientific Data, 2024). ~840 MB, no registration required.

LICENCE: CC BY-NC-ND 4.0. Research use is fine. NoDerivatives means derived
images -- Grad-CAM overlays included -- must not be redistributed. `data/` is
gitignored; keep it that way.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401  (path side effect)
from onnm.config import load_config
from onnm.utils import get_logger

logger = get_logger("download")

FIGSHARE_URL = "https://ndownloader.figshare.com/files/50653575"
EXPECTED_BYTES = 840_474_929
ARTICLE_URL = (
    "https://figshare.com/articles/dataset/"
    "A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_"
    "of_Primary_Bone_Tumors/27865398"
)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, expected_bytes: int | None = None) -> Path:
    """Stream a URL to disk with a progress bar, resuming-safe via a .part file."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        logger.info("archive already present: %s (%s)", dest, human(dest.stat().st_size))
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)

    request = urllib.request.Request(url, headers={"User-Agent": "onnm-downloader/0.1"})
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https URL)
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            chunk = 1024 * 256
            with part.open("wb") as fh:
                while True:
                    buf = response.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total:
                        pct = 100 * done / total
                        bar = "#" * int(pct // 2.5)
                        print(f"\r  [{bar:<40}] {pct:5.1f}%  {human(done)}", end="", flush=True)
            print()
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"download failed ({exc}). If the direct link has rotated, fetch the zip "
            f"manually from {ARTICLE_URL} and place it at {dest}"
        ) from exc

    size = part.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        # A size mismatch here is almost always a truncated transfer or an HTML
        # error page saved as a zip. Failing now beats a confusing unzip error.
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch: got {size} bytes, expected {expected_bytes}. "
            "The transfer was truncated or the link now serves something else. "
            f"Download manually from {ARTICLE_URL}"
        )

    part.replace(dest)
    logger.info("downloaded %s", human(size))
    return dest


def extract(archive: Path, dest_root: Path) -> Path:
    """Extract the zip, flattening a single redundant top-level directory."""
    dest_root.mkdir(parents=True, exist_ok=True)
    staging = dest_root.parent / f"_{dest_root.name}_staging"
    if staging.exists():
        shutil.rmtree(staging)

    logger.info("extracting %s -> %s", archive.name, dest_root)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        # Reject absolute paths and parent traversal before writing anything.
        for name in members:
            target = (staging / name).resolve()
            if not str(target).startswith(str(staging.resolve())):
                raise RuntimeError(f"refusing unsafe archive member: {name!r}")
        zf.extractall(staging)

    entries = [p for p in staging.iterdir() if not p.name.startswith("__MACOSX")]
    source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

    for item in source.iterdir():
        target = dest_root / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))

    shutil.rmtree(staging, ignore_errors=True)
    return dest_root


def summarise(root: Path) -> None:
    print(f"\nExtracted to: {root}")
    for child in sorted(root.iterdir()):
        if child.is_dir():
            n = sum(1 for _ in child.iterdir())
            print(f"  {child.name + '/':<24} {n} entries")
        else:
            print(f"  {child.name:<24} {human(child.stat().st_size)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--archive", default=None, help="use an already-downloaded zip")
    parser.add_argument("--force", action="store_true", help="re-extract even if data exists")
    parser.add_argument("--keep-archive", action="store_true", help="do not delete the zip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg.resolve_path("paths.data_root")
    table_path = data_root / cfg.paths.table_name

    if table_path.is_file() and not args.force:
        logger.info("dataset already extracted at %s (use --force to redo)", data_root)
        summarise(data_root)
        print("\nNext: python scripts/verify_data.py --dump-schema")
        return 0

    archive = Path(args.archive) if args.archive else data_root.parent / "BTXRD.zip"
    if not archive.is_file():
        download(FIGSHARE_URL, archive, EXPECTED_BYTES)

    extract(archive, data_root)
    summarise(data_root)

    if not args.keep_archive and not args.archive:
        archive.unlink(missing_ok=True)
        logger.info("removed archive (pass --keep-archive to retain it)")

    print("\nLicence: CC BY-NC-ND 4.0 -- research use only, do not redistribute")
    print("derived images. `data/` is gitignored; keep it that way.")
    print("\nNext: python scripts/verify_data.py --dump-schema")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("interrupted")
        sys.exit(130)
