"""Manage the ONN model version ledger.

    python scripts/version_model.py list
    python scripts/version_model.py seed --run full-20260822-041653
    python scripts/version_model.py register --run full-20260823-... --level patch
    python scripts/version_model.py render
    python scripts/version_model.py rollback v1.0.0

The ledger is ``model_versions.json``; ``ONN.md`` is rendered from it and is
never hand-edited. See ``onnm.versioning`` for what the levels mean and how
promotion is guarded.

WHAT `register` DOES NOT DO
---------------------------
It does not train, and it does not decide on its own that a run is good. It
reads ``reports/<run>/metrics_test.json`` -- which ``evaluate.py`` wrote --
scores the bone-versus-misc gate against whatever misuse has been reviewed, and
then applies the promotion guard. Registration always succeeds; promotion is
what can be refused, and a refusal leaves ``reports/PRODUCTION`` untouched.

That separation is the whole safety net. A retrain that damages the model
produces a `held` row in the ledger and changes nothing about what is served.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from onnm.ood_eval import evaluate_gate  # noqa: E402
from onnm.utils import get_logger  # noqa: E402
from onnm.versioning import (  # noqa: E402
    MARKDOWN_PATH,
    REGISTRY_PATH,
    Version,
    load_registry,
    register,
    save_registry,
    serving,
    write_markdown,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_OOD_MANIFEST = REPO_ROOT / "configs" / "ood_manifest.csv"


def _read_metrics(run: str, split: str = "test") -> dict[str, float]:
    """Pull the guarded lesion metrics out of what ``evaluate.py`` wrote.

    Malignant recall is taken from the confidence-interval block rather than the
    metric block when both are present, because the CI carries the point
    estimate alongside the interval and the ledger should record the number that
    was reported with its uncertainty, not a second one computed elsewhere.
    """
    path = REPORTS / run / f"metrics_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run:\n"
            f"  python scripts/evaluate.py --checkpoint reports/{run}/best.pt"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    intervals = data.get("confidence_intervals", {})

    out: dict[str, float] = {}
    # Renamed on the way in. `evaluate.py` writes `roc_auc_macro`; the ledger
    # says `macro_roc_auc`, which is the order the metric is spoken in. Mapping
    # explicitly rather than copying the whole block also keeps the ledger from
    # silently acquiring `accuracy`, which on this dataset is a number that
    # looks like progress and is not -- "never malignant" scores 90.9%.
    for source, name in (
        ("roc_auc_macro", "macro_roc_auc"),
        ("pr_auc_macro", "macro_pr_auc"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("f1_macro", "macro_f1"),
        ("malignant_ppv", "malignant_ppv"),
    ):
        if source in metrics:
            out[name] = float(metrics[source])
    recall = intervals.get("malignant_recall") or {}
    if "point" in recall:
        out["malignant_recall"] = float(recall["point"])
        out["malignant_recall_lo"] = float(recall.get("lo", recall["point"]))
        out["malignant_recall_hi"] = float(recall.get("hi", recall["point"]))
    elif "malignant_recall" in metrics:
        out["malignant_recall"] = float(metrics["malignant_recall"])
    return out


def _gate_metrics(ood_manifest: Path, bone_sample: int = 0) -> tuple[dict[str, float], dict]:
    """Score bone-versus-misc. Returns ``(metrics, full_report)``.

    ``bone_sample`` draws that many real radiographs from the dataset so the
    rejection rate is never read without the cost of achieving it. It is 0 by
    default because the daily cycle may run on a machine where the dataset is
    not mounted, and a missing dataset must not fail the ledger.
    """
    images: list[Path] = []
    if bone_sample:
        root = REPO_ROOT / "data" / "raw" / "BTXRD" / "images"
        if root.is_dir():
            images = sorted(p for p in root.iterdir() if p.is_file())[:bone_sample]
        else:
            logger.info("BTXRD images not present -- skipping the bone-acceptance half")

    report = evaluate_gate(ood_manifest, images)
    metrics: dict[str, float] = {}
    if report.misc_rejection is not None:
        metrics["misc_rejection"] = report.misc_rejection
    if report.bone_acceptance is not None:
        metrics["bone_acceptance"] = report.bone_acceptance
    return metrics, report.as_dict()


def _checkpoint_sha256(run: str) -> str:
    """Digest of the run's weights, so the ledger can identify them later.

    Computed once, here, rather than by whatever is asking. A deployed app knows
    only the bytes it downloaded -- not the run name, which is a local
    convention -- so this digest is the only link back from a live model to the
    row describing it.
    """
    import hashlib

    for name in ("best.pt", "last.pt"):
        path = REPORTS / run / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()
    logger.warning("no checkpoint under reports/%s -- the version will carry no digest", run)
    return ""


def _community_summary(store: Path) -> dict:
    index = store / "store.json"
    if not index.is_file():
        return {}
    data = json.loads(index.read_text(encoding="utf-8"))
    return {
        "batches": len(data.get("batches", [])),
        "lesion_rows_total": data.get("lesion_rows", 0),
        "ood_rows_total": data.get("ood_rows", 0),
        "class_balance": data.get("class_balance", {}),
    }


def _pin_production(run: str) -> None:
    """Point ``reports/PRODUCTION`` at a run. This is what changes what serves."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "PRODUCTION").write_text(
        "# The run served by app.py. Written by scripts/version_model.py.\n"
        "# Change it by promoting a version, not by editing this file.\n"
        f"{run}\n",
        encoding="utf-8",
    )


def _write(versions: list[Version]) -> None:
    save_registry(versions)
    write_markdown(versions)
    print(f"wrote {REGISTRY_PATH.name} and {MARKDOWN_PATH.name}")


def cmd_list(args: argparse.Namespace) -> int:
    versions = load_registry()
    if not versions:
        print("nothing registered yet. Seed the first version with:")
        print("  python scripts/version_model.py seed --run <run-directory>")
        return 0
    for version in versions:
        marker = "*" if version.status == "serving" else " "
        auc = version.metrics.get("macro_roc_auc")
        recall = version.metrics.get("malignant_recall")
        misc = version.metrics.get("misc_rejection")
        print(
            f"{marker} {version.version:<10} {version.status:<11} {version.run:<28} "
            f"AUC {auc if auc is None else f'{auc:.4f}'}  "
            f"recall {recall if recall is None else f'{recall:.3f}'}  "
            f"misc {misc if misc is None else f'{misc:.3f}'}"
        )
        if version.held_because:
            print(f"    held: {version.held_because}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    versions = load_registry()
    try:
        metrics = _read_metrics(args.run, args.split)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    gate, report = _gate_metrics(Path(args.ood_manifest), args.bone_sample)
    metrics.update(gate)

    community = _community_summary(Path(args.store))
    versions, added = register(
        versions,
        run=args.run,
        metrics=metrics,
        community=community,
        note=args.note or "",
        level=args.level,
        version=args.version,
        checkpoint_sha256=_checkpoint_sha256(args.run),
    )
    _write(versions)

    print(f"\nregistered {added.version} ({added.status})")
    for name, value in sorted(added.metrics.items()):
        print(f"  {name:<24}{value:.4f}")
    if report.get("misses"):
        print(f"  gate let through: {', '.join(report['misses'][:8])}")

    if added.status == "serving":
        _pin_production(added.run)
        print(f"\npromoted: reports/PRODUCTION now pins {added.run}")
    else:
        current = serving(versions)
        still = f" and still pins {current.run}" if current else ""
        print(f"\nNOT promoted -- {added.held_because}")
        print(
            f"reports/PRODUCTION is unchanged{still}.\n"
            "The checkpoint is on disk and the row is in the ledger; nothing was lost."
        )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Register the first version from an already-trained, already-evaluated run."""
    if load_registry():
        print("the ledger is not empty -- use `register` instead", file=sys.stderr)
        return 2
    args.level = "patch"
    args.version = args.version or "v1.0.0"
    return cmd_register(args)


def cmd_render(args: argparse.Namespace) -> int:
    _write(load_registry())
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Point serving back at an earlier version. The ledger records that it moved."""
    versions = load_registry()
    target = next((v for v in versions if v.version == args.version), None)
    if target is None:
        print(f"no such version: {args.version}", file=sys.stderr)
        return 2
    checkpoint_dir = REPORTS / target.run
    if not checkpoint_dir.is_dir():
        print(
            f"{checkpoint_dir} does not exist -- that run's weights are not on this "
            "machine, so it cannot be served from here.",
            file=sys.stderr,
        )
        return 2

    for version in versions:
        if version.status == "serving":
            version.status = "superseded"
    target.status = "serving"
    _write(versions)
    _pin_production(target.run)
    print(f"rolled back to {target.version} (`{target.run}`)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="run directory under reports/")
        p.add_argument("--split", default="test", help="which metrics_<split>.json to read")
        p.add_argument("--note", default=None, help="free text recorded with the version")
        p.add_argument("--version", default=None, help="set the number explicitly")
        p.add_argument("--store", default="data/community", help="community store to summarise")
        p.add_argument("--ood-manifest", default=str(DEFAULT_OOD_MANIFEST))
        p.add_argument("--bone-sample", type=int, default=200,
                       help="how many BTXRD images to test gate acceptance on (0 to skip)")

    p_list = sub.add_parser("list", help="show the ledger")
    p_list.set_defaults(func=cmd_list)

    p_seed = sub.add_parser("seed", help="register the first version (v1.0.0)")
    add_common(p_seed)
    p_seed.set_defaults(func=cmd_seed)

    p_reg = sub.add_parser("register", help="register a new version and maybe promote it")
    add_common(p_reg)
    p_reg.add_argument("--level", default="patch", choices=("major", "minor", "patch"))
    p_reg.set_defaults(func=cmd_register)

    p_render = sub.add_parser("render", help="regenerate ONN.md from the JSON")
    p_render.set_defaults(func=cmd_render)

    p_back = sub.add_parser("rollback", help="serve an earlier version again")
    p_back.add_argument("version")
    p_back.set_defaults(func=cmd_rollback)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
