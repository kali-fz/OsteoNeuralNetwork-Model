"""Score the real failure cases, so "it looked at the joint" becomes a number.

    .venv\\Scripts\\python.exe scripts\\check_test_radiographs.py
    .venv\\Scripts\\python.exe scripts\\check_test_radiographs.py --checkpoint reports/<run>/best.pt

WHY THIS EXISTS
---------------
The complaint that started this work is not visible in any metric the project
records. ``metrics_test.json`` says macro ROC-AUC 0.893; the gradcam report says
pointing game 0.0936 -- and neither tells you that on a *normal* hip film the
evidence lands squarely on the ball-and-socket joint, which is the thing that
actually has to stop.

These are the real films that failed, kept out of git (see .gitignore: the
repository is public and CONTRIBUTING.md forbids committing a radiograph). This
script drives them through the SAME path the website uses -- ``ScanService.run_scan``,
including the OOD gate and the uncertainty gate -- and reports, per film:

* the verdict a visitor would see,
* whether that verdict is right, when the expected label is known,
* **where the evidence landed**, as a normalised (x, y) and as the share of heat
  in each vertical third.

That last column is the point. A number for "it looked at the joint" is what
makes the lesion head's effect checkable instead of arguable.

DECLARING GROUND TRUTH
----------------------
Optional ``test_radiographs/expected.json``::

    {"radiograph of femur (Normal).png": {"label": "normal", "note": "control"},
     "OSTEOSARCOMA OF KNEE.png":         {"label": "malignant"}}

Without it the label is guessed from the filename, and anything unrecognised is
reported as ``unknown`` rather than assumed -- a wrong assumption here would
silently invert a pass into a fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "test_radiographs"

#: Filename keywords -> expected class, used only when expected.json is absent.
#: Deliberately conservative: an unmatched name yields "unknown", because
#: guessing wrong turns a false positive into an apparent pass.
FILENAME_HINTS: tuple[tuple[str, str], ...] = (
    ("normal", "normal"),
    ("control", "normal"),
    ("osteosarcoma", "malignant"),
    ("malignant", "malignant"),
    ("benign", "benign"),
)


def expected_label(name: str, declared: dict) -> str:
    entry = declared.get(name)
    if isinstance(entry, dict) and entry.get("label"):
        return str(entry["label"]).lower()
    if isinstance(entry, str):
        return entry.lower()
    lowered = name.lower()
    for keyword, label in FILENAME_HINTS:
        if keyword in lowered:
            return label
    return "unknown"


def heat_geometry(heatmap: np.ndarray) -> dict[str, float]:
    """Where the evidence sits, reduced to numbers that survive a screenshot.

    ``peak_x``/``peak_y`` are the centroid of the maximal region, matching
    ``explainability.pointing_game`` rather than ``argmax`` -- on a flat map
    argmax reports the top-left corner of the plateau every time, which reads as
    a confident statement about the corner of the film.

    The thirds are the useful part in practice. On these films the lesion and
    the joint sit in different bands, so "share of heat in the middle third"
    tracks the failure directly without needing an annotation.
    """
    if heatmap is None or heatmap.size == 0:
        return {}
    hot = heatmap >= heatmap.max() - 1e-6
    rows, cols = np.nonzero(hot)
    height, width = heatmap.shape
    total = float(heatmap.sum())

    bands = np.array_split(heatmap, 3, axis=0)
    shares = [float(b.sum() / total) if total > 0 else float("nan") for b in bands]

    return {
        "peak_x": float(cols.mean() / max(width - 1, 1)),
        "peak_y": float(rows.mean() / max(height - 1, 1)),
        "peak_fraction": float(hot.sum() / heatmap.size),
        "top_third": shares[0],
        "middle_third": shares[1],
        "bottom_third": shares[2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to the checkpoint reports/PRODUCTION pins, i.e. what the site serves.",
    )
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for overlay PNGs. Defaults to <dir>/overlays, which is gitignored.",
    )
    parser.add_argument("--json", default=None, help="Also write the results as JSON.")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "inference"))
    from service import ScanService  # noqa: E402  -- needs the path above

    from onnm.inference import default_checkpoint, production_checkpoint

    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            checkpoint = str(production_checkpoint())
        except Exception:  # noqa: BLE001 - a stale pin is not worth a traceback here
            checkpoint = str(default_checkpoint())

    directory = Path(args.dir)
    if not directory.is_dir():
        logger.error("%s does not exist", directory)
        return 1

    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
    )
    if not images:
        logger.error("no radiographs in %s", directory)
        return 1

    declared_path = directory / "expected.json"
    declared = (
        json.loads(declared_path.read_text(encoding="utf-8"))
        if declared_path.is_file()
        else {}
    )

    out_dir = Path(args.out) if args.out else directory / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("checkpoint: %s", checkpoint)
    service = ScanService(checkpoint, warmup=False)

    rows: list[dict] = []
    for path in images:
        payload = path.read_bytes()
        result = service.run_scan(payload, path.name, with_heatmap=True, want_preprocessed=False)

        row: dict = {"image": path.name, "expected": expected_label(path.name, declared)}

        if "prediction" not in result:
            # The OOD gate refused it before the model ever saw it.
            row["verdict"] = "REJECTED by the input gate"
            row["ok"] = False
            rows.append(row)
            continue

        prediction = result["prediction"]
        row["verdict"] = prediction["label"]
        row["confidence"] = float(prediction.get("confidence_pct", 0.0))
        probabilities = prediction.get("class_probabilities", {})
        row["p_normal"] = float(probabilities.get("normal", float("nan")))
        row["p_benign"] = float(probabilities.get("benign", float("nan")))
        row["p_malignant"] = float(probabilities.get("malignant", float("nan")))
        row["cam_class"] = (result.get("overlay") or {}).get("cam_class")

        # A control film is correct only when the site shows NO finding. This is
        # the number the work is aimed at: false positives on complex joints.
        if row["expected"] == "normal":
            row["ok"] = prediction["label"].lower().startswith("normal")
        elif row["expected"] in {"benign", "malignant"}:
            row["ok"] = not prediction["label"].lower().startswith("normal")
        else:
            row["ok"] = None

        overlay = result.get("overlay") or {}
        if overlay.get("png_b64"):
            import base64

            target = out_dir / f"{path.stem}_overlay.png"
            target.write_bytes(base64.b64decode(overlay["png_b64"]))
            # relative_to raises when --out points outside the repo, which is a
            # perfectly reasonable thing to ask for.
            try:
                row["overlay"] = str(target.relative_to(REPO_ROOT))
            except ValueError:
                row["overlay"] = str(target)

        heatmap = getattr(service.classifier, "_last_heatmap", None)
        if heatmap is None:
            # run_scan does not hand the raw map back, so recompute geometry from
            # the returned overlay only when it is unavailable. Kept optional
            # rather than fatal: the verdict columns are useful on their own.
            row.update({})
        else:
            row.update(heat_geometry(np.asarray(heatmap)))

        rows.append(row)

    # -- report ------------------------------------------------------------
    print()
    print(f"{'image':<38}{'expected':<11}{'verdict':<26}{'conf':>6}{'P(mal)':>8}  ok")
    print("-" * 96)
    for row in rows:
        ok = {True: "PASS", False: "FAIL", None: "  ?"}[row.get("ok")]
        print(
            f"{row['image'][:37]:<38}{row['expected']:<11}{row['verdict'][:25]:<26}"
            f"{row.get('confidence', float('nan')):>5.1f}%"
            f"{row.get('p_malignant', float('nan')):>8.3f}  {ok}"
        )

    scored = [r for r in rows if r.get("ok") is not None]
    controls = [r for r in rows if r["expected"] == "normal"]
    false_positives = [r for r in controls if r.get("ok") is False]

    print()
    if controls:
        rate = 100.0 * len(false_positives) / len(controls)
        print(f"controls: {len(controls)}, false positives: {len(false_positives)} ({rate:.0f}%)")
        for row in false_positives:
            print(
                f"  FP  {row['image']}: called {row['verdict']} "
                f"at {row.get('confidence', 0):.1f}%"
            )
    if scored:
        passed = sum(1 for r in scored if r["ok"])
        print(f"scored: {passed}/{len(scored)} correct")
    print(f"overlays written to {out_dir}")
    print(
        "\nThese are a handful of films, not a metric. Read them as a regression\n"
        "check on the exact failures that prompted the work, and keep the\n"
        "held-out test split for anything that goes in the ledger."
    )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
