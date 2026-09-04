"""Run a batch of experiments unattended, and promote absolutely nothing.

    .venv\\Scripts\\python.exe scripts\\overnight_sweep.py --plan configs/sweeps/lesion_head.yaml
    .venv\\Scripts\\python.exe scripts\\overnight_sweep.py --plan ... --resume
    .venv\\Scripts\\python.exe scripts\\overnight_sweep.py --plan ... --dry-run

WHY A SWEEP AND NOT ONE LONG RUN
--------------------------------
A full 40-epoch run on this machine takes about 21 minutes -- ``full_run.yaml``
says so in its own comments (284 ms/step, ~16 min for 40 epochs), and the
``full-20260822-041653`` run bears it out: config.json at 04:16:53, history.json
at 04:37:58, 26 epochs. Training the same recipe for eight hours does not produce
a better model, it produces an overfitted one; that run early-stopped at epoch 19
of 26 because validation had stopped improving.

So a night is worth roughly twenty experiments, and that is what this spends it
on. The yield comes from breadth (which loss weight?) and from the genuinely
expensive axis (resolution: 384px is 2.25x the pixels of 256px, 512px is 4x), not
from more epochs.

WHAT IT WILL NOT DO
-------------------
It never calls ``version_model.py``. Ten runs produce ten report directories and
one comparison table; a human reads that table and registers ONE winner.
Promotion stays a separate, deliberate act -- which is the whole design of
``onnm.versioning`` -- rather than a side effect of leaving the computer on.

It also continues past a failed run rather than aborting. Losing run 3 of 10 to a
bad config should cost one row of the table, not the remaining seven hours.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401  -- puts src/ on sys.path
from onnm.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

#: Wall-clock ceiling per pipeline step, in seconds. The last line of defence,
#: not the normal path -- see :func:`_run` for how a completed step is really
#: detected. Generous on purpose: a 512px train is the slowest legitimate thing
#: here, at roughly two hours.
STEP_TIMEOUTS: dict[str, int] = {
    "train": 4 * 3600,
    "calibrate": 45 * 60,
    "evaluate": 45 * 60,
    "gradcam": 60 * 60,
}

#: How long to keep waiting for a well-behaved exit once the step's output file
#: exists. Long enough that a process merely finishing up is never cut off, short
#: enough that a teardown hang costs seconds instead of the whole timeout.
EXIT_GRACE_SECONDS = 30


def _run(step: str, argv: list[str], log: Path, expect: Path | None = None) -> bool:
    """Run one pipeline step, treating its OUTPUT FILE as the completion signal.

    WHY NOT SIMPLY WAIT FOR THE PROCESS TO EXIT
    -------------------------------------------
    On this machine -- ROCm 7.2.1 on Windows, RX 7900 XT -- a Python process that
    has run a convolution on the GPU frequently never returns from interpreter
    finalisation. It is not a crash and not a stall in the work: the script runs
    to completion, writes its output, prints its closing message, and *then* sits
    at 100% CPU indefinitely. It survives Stop-Process, and even
    ``faulthandler.dump_traceback_later(exit=True)`` never fires, because by that
    point the watchdog thread is gone. That places it below Python.

    Bisected on 2026-09-04:

        build a model and move it to the GPU, no forward  -> exits cleanly
        load a checkpoint to the GPU, no forward          -> exits cleanly
        the same, plus ONE convolution forward pass       -> hangs
        a dataloader iterated with no model at all        -> exits cleanly

    ``scripts/train.py`` happens to exit cleanly, which is why this went unnoticed
    for so long; ``calibrate.py``, ``evaluate.py``, ``gradcam_report.py`` and
    ``stratified_report.py`` do not. With ``subprocess.run`` waiting on exit, a
    single hung step blocked the entire sweep -- the run that prompted this fix
    sat for 33 minutes on a ``calibrate`` whose ``calibration.json`` had been
    written, complete and valid, in the first 40 seconds.

    So completion is detected from the artefact the step exists to produce. The
    output is whole before the hang begins -- the hang is strictly after the work
    -- which makes this a correct signal rather than a hopeful one. A step that
    exits properly still takes the fast path; the polling only matters when it
    does not.
    """
    logger.info("  %s: %s", step, " ".join(argv[1:]))
    started = time.time()
    timeout = STEP_TIMEOUTS.get(step, 3600)

    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 70 + "\n")
        handle.write(f"{step}: {' '.join(argv)}\n")
        handle.write("=" * 70 + "\n")
        handle.flush()

        process = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)

        produced_at: float | None = None
        while True:
            code = process.poll()
            if code is not None:
                ok = code == 0
                status = "ok" if ok else f"FAILED (exit {code})"
                break

            if time.time() - started > timeout:
                process.kill()
                handle.write(f"\n*** {step} exceeded {timeout}s and was killed ***\n")
                ok, status = False, f"TIMED OUT after {timeout}s"
                break

            if expect is not None and expect.is_file():
                if produced_at is None:
                    produced_at = time.time()
                elif time.time() - produced_at > EXIT_GRACE_SECONDS:
                    process.kill()
                    handle.write(
                        f"\n*** {step} wrote {expect.name} but did not exit within "
                        f"{EXIT_GRACE_SECONDS}s. Killed. This is the known ROCm "
                        "teardown hang; the output is complete. See _run. ***\n"
                    )
                    ok, status = True, "ok (output written, killed after teardown hang)"
                    break

            time.sleep(2.0)

    logger.info("  %s: %s in %.1fs", step, status, time.time() - started)
    return ok


def _materialise(name: str, overrides: dict, directory: Path) -> Path:
    """Write a recipe's inline overrides to a real YAML file.

    train.py takes ``--override <path>`` and nothing else, so an inline block in
    the plan has to become a file. Written into the sweep directory rather than a
    temp dir, so the exact config of every run survives beside its results.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    return path


def _newest_run(tag: str, reports: Path) -> Path | None:
    candidates = sorted(reports.glob(f"{tag}-*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _collect(run_dir: Path, split: str) -> dict:
    """Read back whatever the pipeline managed to produce for one run."""
    out: dict = {"run": run_dir.name}

    metrics = run_dir / f"metrics_{split}.json"
    if metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8")).get("metrics", {})
        out["macro_roc_auc"] = payload.get("roc_auc_macro")
        out["malignant_recall"] = payload.get("malignant_recall")
        out["balanced_accuracy"] = payload.get("balanced_accuracy")
        # Specificity on the normal class is the false-positive rate the owner's
        # sub-10% target is written against, so it is carried explicitly rather
        # than left to be re-derived from a confusion matrix later.
        normal = payload.get("per_class", {}).get("normal", {})
        if normal.get("specificity") is not None:
            out["normal_specificity"] = normal["specificity"]
        errors = payload.get("clinical_errors", {})
        out["normal_called_malignant"] = errors.get("normal_called_malignant")

    cam = run_dir / f"gradcam_{split}" / "gradcam_report.json"
    if cam.is_file():
        local = json.loads(cam.read_text(encoding="utf-8")).get("localisation", {})
        out["pointing_game"] = local.get("pointing_game_accuracy")
        out["mean_iou"] = local.get("mean_iou")
        out["mean_coverage"] = local.get("mean_coverage")

    history = run_dir / "history.json"
    if history.is_file():
        epochs = json.loads(history.read_text(encoding="utf-8"))
        out["epochs"] = len(epochs)
        if epochs and "lesion_loss" in epochs[-1]:
            out["lesion_loss"] = epochs[-1]["lesion_loss"]
    return out


#: Column order for the summary table. Localisation first, because that is what
#: this sweep exists to move; the guarded classification metrics next, because a
#: localisation win that regresses either of them will be refused by
#: onnm.versioning and is therefore not a win.
COLUMNS: list[tuple[str, str, str | None]] = [
    ("run", "run", None),
    ("epochs", "ep", "{:.0f}"),
    ("pointing_game", "point", "{:.4f}"),
    ("mean_iou", "IoU", "{:.4f}"),
    ("malignant_recall", "mal-rec", "{:.4f}"),
    ("macro_roc_auc", "ROC", "{:.4f}"),
    ("normal_specificity", "spec", "{:.4f}"),
    ("normal_called_malignant", "N->M", "{:.0f}"),
    ("lesion_loss", "seg-loss", "{:.4f}"),
]


def _table(rows: list[dict]) -> str:
    widths = {key: max(len(header), 9) for key, header, _ in COLUMNS}
    widths["run"] = max([len(str(r.get("run", ""))) for r in rows] + [3])

    lines = ["  ".join(h.ljust(widths[k]) for k, h, _ in COLUMNS)]
    lines.append("  ".join("-" * widths[k] for k, _, _ in COLUMNS))
    for row in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append("-".ljust(widths[key]))
            elif fmt is None:
                cells.append(str(value).ljust(widths[key]))
            else:
                cells.append(fmt.format(value).ljust(widths[key]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="YAML sweep plan")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--stamp",
        default=None,
        help=(
            "Reuse an existing sweep id instead of minting one. With --resume this "
            "is what lets a killed sweep pick up where it stopped."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip work whose output is already on disk: a run with best.pt is not "
            "retrained, and calibrate/evaluate/gradcam are skipped individually if "
            "their artefact exists."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated run names to execute, ignoring the rest of the plan.",
    )
    args = parser.parse_args()

    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    recipes = plan.get("runs", [])
    base_overrides = [str(o) for o in plan.get("base_overrides", [])]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        recipes = [r for r in recipes if str(r["name"]) in wanted]
    if not recipes:
        logger.error("%s defines no runs to execute", args.plan)
        return 1

    stamp = args.stamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    sweep_dir = REPO_ROOT / "reports" / f"sweep-{stamp}"
    reports = REPO_ROOT / "reports"

    logger.info("sweep %s: %d run(s)", sweep_dir.name, len(recipes))
    for recipe in recipes:
        logger.info("  %-24s %s", recipe["name"], recipe.get("overrides", {}))
    if args.dry_run:
        logger.info("dry run: nothing executed")
        return 0

    sweep_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for index, recipe in enumerate(recipes, start=1):
        name = str(recipe["name"])
        tag = f"sweep-{stamp}-{name}"
        logger.info("[%d/%d] %s", index, len(recipes), name)

        overrides = list(base_overrides)
        if recipe.get("overrides"):
            overrides.append(str(_materialise(name, recipe["overrides"], sweep_dir / "configs")))

        override_argv: list[str] = []
        for path in overrides:
            override_argv += ["--override", path]

        log = sweep_dir / f"{name}.log"
        started = time.time()

        existing = _newest_run(tag, reports) if args.resume else None
        if existing is not None and (existing / "best.pt").is_file():
            logger.info("  train: already done, reusing %s", existing.name)
            run_dir = existing
        else:
            # `expect` is deliberately not passed for training: train.py rewrites
            # best.pt once per improving epoch, so its presence says nothing about
            # completion. train.py also exits cleanly, so it needs no workaround.
            if not _run("train", [PYTHON, "scripts/train.py", *override_argv, "--tag", tag], log):
                results.append({"run": tag, "error": "train failed"})
                continue
            run_dir = _newest_run(tag, reports)

        if run_dir is None:
            results.append({"run": tag, "error": "no run directory"})
            continue
        checkpoint = str(run_dir / "best.pt")

        # calibrate/evaluate/gradcam deliberately get NO --override: since
        # model.build_model_for_checkpoint reads the architecture out of the
        # checkpoint itself, they no longer need to be told about it. That is the
        # same property daily_cycle.py relies on.
        steps: list[tuple[str, list[str], Path]] = [
            (
                "calibrate",
                [PYTHON, "scripts/calibrate.py", "--checkpoint", checkpoint],
                run_dir / "calibration.json",
            ),
            (
                "evaluate",
                [PYTHON, "scripts/evaluate.py", "--checkpoint", checkpoint,
                 "--split", args.split],
                run_dir / f"metrics_{args.split}.json",
            ),
            (
                "gradcam",
                [PYTHON, "scripts/gradcam_report.py", "--checkpoint", checkpoint,
                 "--split", args.split],
                run_dir / f"gradcam_{args.split}" / "gradcam_report.json",
            ),
        ]
        for step_name, argv, artefact in steps:
            if args.resume and artefact.is_file():
                logger.info("  %s: already done", step_name)
                continue
            _run(step_name, argv, log, expect=artefact)

        row = _collect(run_dir, args.split)
        row["minutes"] = round((time.time() - started) / 60.0, 1)
        results.append(row)
        logger.info("  -> %s", {k: v for k, v in row.items() if k != "run"})

        (sweep_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (sweep_dir / "summary.txt").write_text(_table(results) + "\n", encoding="utf-8")

    table = _table(results)
    (sweep_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    logger.info("\n%s\n", table)
    logger.info("sweep complete: %s", sweep_dir)
    logger.info(
        "NOTHING was promoted. Read the table, pick one run, then register it:\n"
        "  python scripts/version_model.py register --run <run> --level major"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
