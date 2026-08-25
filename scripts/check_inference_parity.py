"""Prove that a checkpoint predicts identically on two machines.

Moving the app from Streamlit Community Cloud (x86, CPU torch) to an ARM64 VM
changes the CPU architecture, and with it the oneDNN kernels torch dispatches to.
That is the one change in the migration capable of altering a number the model
shows a clinician, and it would do so silently: nothing errors, the app looks
fine, and only the probabilities move.

Usage is two runs and a comparison.

    # Generate the probe on BOTH machines; the printed sha256 must match.
    python scripts/check_inference_parity.py make-probe --out probe.png

    # On the machine you trust. --device defaults to cpu, and that matters:
    # the migration is CPU to CPU, so recording this baseline on the local
    # ROCm GPU would compare device AND architecture at once, and a failure
    # would not say which one moved. get_device() returns cuda:0 here, so
    # the default is doing real work.
    python scripts/check_inference_parity.py record \\
        --checkpoint reports/<run>/best.pt --image tests/fixtures/x.png \\
        --out parity-local.json

    # on the new VM, same checkpoint, same image
    python scripts/check_inference_parity.py record \\
        --checkpoint /opt/onnm/reports/hosted/best.pt --image /tmp/x.png \\
        --out parity-vm.json

    # anywhere
    python scripts/check_inference_parity.py compare \\
        parity-local.json parity-vm.json

Exits non-zero when any guarded value differs by more than the tolerance, so it
can gate a deployment rather than merely informing one.

Note on ASCII: console output here stays ASCII because Windows consoles default
to cp1252 and cannot encode an em dash. A stray one in a print, or in this
docstring which argparse prints for --help, is a UnicodeEncodeError on the
developer's machine and a silent failure under Task Scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Values whose drift changes what a user is told. Compared with a tight
# tolerance. Note the absence of elapsed_ms and device: those are expected to
# differ between machines and say nothing about correctness.
GUARDED_FLOATS = (
    "lesion_probability",
    "prob_normal",
    "prob_benign",
    "prob_malignant",
    "confidence",
    "max_probability",
    "predictive_entropy",
    "threshold",
    "temperature",
    "heatmap_mean",
    "heatmap_max",
)

GUARDED_EXACT = (
    "label",
    "top_class",
    "cam_class",
    "calibrated",
    "inconclusive",
    "heatmap_argmax",
)

# 1e-4 is deliberately loose enough to absorb the last bits of float32
# associativity differences between BLAS kernels, and far tighter than any
# difference that would move a verdict: the calibrated threshold sits at 0.496,
# so a 1e-4 shift cannot flip a call that was not already exactly on the line.
DEFAULT_TOLERANCE = 1e-4


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_probe(args: argparse.Namespace) -> int:
    """Write a deterministic synthetic radiograph.

    Both machines can generate this independently and get byte-identical files,
    so the parity check needs no file transfer and no BTXRD image. BTXRD is
    CC BY-NC-ND, and a synthetic probe keeps a licensed image off a server
    whose only job is to answer this one question.

    A real radiograph is still the better probe for a final check, because it
    exercises the transform chain on data with realistic statistics. Use both.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(args.seed)
    size = args.size
    yy, xx = np.mgrid[0:size, 0:size]
    shaft = np.exp(-((xx - size / 2) ** 2) / (2 * (size * 0.12) ** 2))
    grad = (yy / size) * 0.25
    noise = rng.normal(0, 0.02, (size, size))
    img = np.clip(0.25 + 0.55 * shaft + grad + noise, 0, 1)

    out = Path(args.out)
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(out, compress_level=6)
    print(f"wrote {out}  sha256={_digest(out)}")
    print("The sha256 must match on both machines. If it does not, the probe")
    print("differs and any parity result is meaningless.")
    return 0


def record(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from onnm.inference import RadiographClassifier

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 2

    # Default cpu, and it matters. The migration this gates is x86 CPU -> ARM64
    # CPU, so the baseline must be CPU as well. Recording it on the local ROCm
    # GPU instead would compare two things at once -- device AND architecture --
    # and a failure would not say which one moved. On the developer machine
    # get_device() returns cuda:0, so leaving this to the default would silently
    # do the wrong thing.
    device = torch.device(args.device)

    # Constructed once and reused. predict_file() would reload the checkpoint per
    # image, and it cannot pass a device through to the constructor anyway.
    classifier = RadiographClassifier(checkpoint, device=device, warmup=False)
    print(f"device: {classifier.device}  arch: {platform.machine()}")

    entries: list[dict[str, Any]] = []
    for image_arg in args.image:
        image = Path(image_arg).resolve()
        if not image.is_file():
            print(f"ERROR: image not found: {image}")
            return 2

        # with_heatmap=True on purpose. Grad-CAM runs a backward pass, which
        # exercises a different set of kernels than the forward pass alone, so
        # a forward-only check would miss exactly the half of the computation
        # most likely to differ across architectures.
        result = classifier.predict(str(image), with_heatmap=True)

        probs = result.class_probabilities
        entry: dict[str, Any] = {
            "image": image.name,
            "image_sha256": _digest(image),
            "label": result.label,
            "top_class": result.top_class,
            "cam_class": result.cam_class,
            "calibrated": bool(result.calibrated),
            "inconclusive": bool(result.inconclusive),
            "lesion_probability": float(result.lesion_probability),
            "prob_normal": float(probs.get("normal", 0.0)),
            "prob_benign": float(probs.get("benign", 0.0)),
            "prob_malignant": float(probs.get("malignant", 0.0)),
            "confidence": float(result.confidence),
            "max_probability": float(result.max_probability),
            "predictive_entropy": float(result.predictive_entropy),
            "threshold": float(result.threshold),
            "temperature": float(result.temperature),
        }

        if result.heatmap is not None:
            heat = np.asarray(result.heatmap, dtype=np.float64)
            peak = np.unravel_index(int(heat.argmax()), heat.shape)
            entry["heatmap_mean"] = float(heat.mean())
            entry["heatmap_max"] = float(heat.max())
            # Where the model is looking matters as much as how hot the peak is.
            # A heatmap that keeps its statistics but moves its peak is a
            # different explanation of the same score.
            entry["heatmap_argmax"] = [int(v) for v in peak]
        else:
            entry["heatmap_mean"] = 0.0
            entry["heatmap_max"] = 0.0
            entry["heatmap_argmax"] = [-1, -1]

        entries.append(entry)
        print(f"recorded {image.name} -> {result.label} ({result.lesion_probability:.6f} lesion)")

    payload = {
        "checkpoint_sha256": _digest(checkpoint),
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch_device": str(classifier.device),
        },
        "results": entries,
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))

    lm, rm = left["machine"], right["machine"]
    print(f"A: {args.left} ({lm['machine']}, {lm.get('torch_device', '?')})")
    print(f"B: {args.right} ({rm['machine']}, {rm.get('torch_device', '?')})")
    if lm.get("torch_device") != rm.get("torch_device"):
        print("")
        print("WARNING: records were taken on different torch devices. A difference")
        print("below may be the device rather than the CPU architecture. Re-record")
        print("both with --device cpu for a clean answer.")
    print("")

    failures: list[str] = []

    # Comparing two different checkpoints would produce a meaningless verdict,
    # so this is a hard stop rather than a warning.
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        print("FAIL: different checkpoints compared.")
        print(f"  A: {left['checkpoint_sha256']}")
        print(f"  B: {right['checkpoint_sha256']}")
        return 1

    by_hash = {entry["image_sha256"]: entry for entry in right["results"]}

    for a in left["results"]:
        name = a["image"]
        b = by_hash.get(a["image_sha256"])
        if b is None:
            failures.append(f"{name}: no matching image in B (compared by content hash)")
            continue

        for field in GUARDED_EXACT:
            if a.get(field) != b.get(field):
                failures.append(f"{name}: {field} differs: {a.get(field)!r} vs {b.get(field)!r}")

        for field in GUARDED_FLOATS:
            av, bv = float(a.get(field, 0.0)), float(b.get(field, 0.0))
            delta = abs(av - bv)
            if delta > args.tolerance:
                failures.append(
                    f"{name}: {field} differs by {delta:.3e} ({av:.8f} vs {bv:.8f})"
                )
            else:
                print(f"  ok  {name:<22} {field:<20} delta {delta:.2e}")

    print("")
    if failures:
        print(f"PARITY FAILED ({len(failures)}):")
        for line in failures:
            print(f"  - {line}")
        print("")
        print("Do NOT cut over. A difference here means the new machine tells")
        print("users something different from the machine the model was measured on.")
        return 1

    print(f"PARITY OK: every guarded value agrees within {args.tolerance:.1e}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare inference output across machines (ARM64 migration gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="run inference and write a parity record")
    rec.add_argument("--checkpoint", required=True, help="path to best.pt")
    rec.add_argument("--image", required=True, nargs="+", help="one or more radiographs")
    rec.add_argument("--out", default="parity.json", help="output JSON path")
    rec.add_argument(
        "--device",
        default="cpu",
        help="torch device (default cpu; the migration being gated is CPU-to-CPU)",
    )
    rec.set_defaults(func=record)

    probe = sub.add_parser("make-probe", help="write a deterministic synthetic radiograph")
    probe.add_argument("--out", default="parity_probe.png")
    probe.add_argument("--seed", type=int, default=20260825)
    probe.add_argument("--size", type=int, default=512)
    probe.set_defaults(func=make_probe)

    cmp_ = sub.add_parser("compare", help="compare two parity records")
    cmp_.add_argument("left")
    cmp_.add_argument("right")
    cmp_.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    cmp_.set_defaults(func=compare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
