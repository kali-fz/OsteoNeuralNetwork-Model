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
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
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

#: Breathing room after killing a hung step, before the next one asks for the
#: GPU. The driver does not release a context the instant the process dies.
GPU_SETTLE_SECONDS = 10


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the step AND everything it spawned.

    ``.venv\\Scripts\\python.exe`` on this machine is a **shim**: it re-execs into
    the Windows Store interpreter, so the process handed to this function is the
    *parent* of the one doing the work. ``Popen.kill`` terminates the shim and
    leaves the real interpreter alive -- holding a GPU context, spinning at half
    a core, and invisible in the sweep log because its stdout handle is gone.

    Measured on 2026-09-04, and it cost most of a day. A ``calibrate.py``
    orphaned this way at 07:48 was still burning CPU at 17:10. Orphans accumulate
    across runs, and they are why the ``w010`` calibrate, evaluate and gradcam
    steps produced no artefacts whatsoever: each started, could not obtain the
    GPU behind two dead runs' contexts, and sat there until its own timeout.

    ``taskkill /T`` walks the child tree, which ``Popen.kill`` cannot do.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()
    # The real interpreter is usually already gone; wait only to reap the shim.
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)


def _resolve(expect: Path | Callable[[], Path | None] | None) -> Path | None:
    """Allow a step's completion artefact to be named lazily.

    ``train.py`` stamps its own output directory with the time it started, so the
    path cannot be written down before launching it. A callable defers the
    question until the directory exists.
    """
    if expect is None or isinstance(expect, Path):
        return expect
    return expect()


def _run(
    step: str,
    argv: list[str],
    log: Path,
    expect: Path | Callable[[], Path | None] | None = None,
) -> bool:
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

    ``calibrate.py``, ``evaluate.py``, ``gradcam_report.py`` and
    ``stratified_report.py`` all hang this way. With ``subprocess.run`` waiting on
    exit, a single hung step blocked the entire sweep -- the run that prompted
    this fix sat for 33 minutes on a ``calibrate`` whose ``calibration.json`` had
    been written, complete and valid, in the first 40 seconds.

    ``scripts/train.py`` was believed to be exempt, and this function used to say
    so. **It is not.** It exits cleanly only when the lesion head is off, which is
    exactly the configuration the workaround was first tested against. With
    ``model.lesion_head: true`` it hangs like everything else -- consistent with
    the bisection above, since the decoder adds convolutions. The cost of that
    wrong assumption on 2026-09-04: ``w010`` finished training at 09:20 having
    reached its best epoch, then sat until the four-hour ceiling killed it at
    12:48. So train is polled for its artefact too; see ``_train_artefact``.

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
        killed = False
        while True:
            code = process.poll()
            if code is not None:
                ok = code == 0
                status = "ok" if ok else f"FAILED (exit {code})"
                break

            if time.time() - started > timeout:
                _kill_tree(process)
                handle.write(f"\n*** {step} exceeded {timeout}s and was killed ***\n")
                ok, status = False, f"TIMED OUT after {timeout}s"
                killed = True
                break

            target = _resolve(expect)
            if target is not None and target.is_file():
                if produced_at is None:
                    produced_at = time.time()
                elif time.time() - produced_at > EXIT_GRACE_SECONDS:
                    _kill_tree(process)
                    handle.write(
                        f"\n*** {step} wrote {target.name} but did not exit within "
                        f"{EXIT_GRACE_SECONDS}s. Killed. This is the known ROCm "
                        "teardown hang; the output is complete. See _run. ***\n"
                    )
                    ok, status = True, "ok (output written, killed after teardown hang)"
                    killed = True
                    break

            time.sleep(2.0)

    if killed:
        # Let the driver reclaim the context before the next step asks for it.
        time.sleep(GPU_SETTLE_SECONDS)

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
        report = json.loads(cam.read_text(encoding="utf-8"))
        local = report.get("localisation", {})
        out["pointing_game"] = local.get("pointing_game_accuracy")
        out["mean_iou"] = local.get("mean_iou")
        out["mean_coverage"] = local.get("mean_coverage")
        # WHICH INSTRUMENT PRODUCED THAT NUMBER. A lesion-head run is scored on
        # the head's map and the baseline on Grad-CAM, which is the right
        # comparison -- each run judged on the explanation it would actually
        # serve -- but only if the column says so. Without it the table silently
        # compares two different measurements and looks like one.
        out["map_source"] = {"lesion_head": "lesion", "gradcam": "cam"}.get(
            local.get("map_source"), local.get("map_source")
        )
        # Chance is per-run because it depends on the films scored, and a
        # pointing game is meaningless without it: these lesion boxes cover
        # roughly a tenth of the frame, so ~0.10 is what a random peak scores.
        out["chance_pointing_game"] = local.get("chance_pointing_game")
        # Grad-CAM on the SAME lesion-head checkpoint. The head's own score says
        # the decoder learned; this says whether the shared backbone moved with
        # it. A run that wins the first and not the second added a picture.
        cam_local = report.get("localisation_gradcam", {})
        out["cam_pointing_game"] = cam_local.get("pointing_game_accuracy")
        # The share of HEALTHY films the map claims a lesion on. Every other
        # column here is computed over annotated films only, so this is the one
        # that speaks to the original complaint -- evidence landing on a normal
        # joint -- and the one to read against the sub-10% target. Unlike
        # normal_specificity it cannot be bought by moving the decision
        # threshold, because the classifier is not involved in it.
        out["normal_flagged"] = report.get("normal_activation", {}).get(
            "flagged_fraction"
        )

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
    # `map` and `chance` sit either side of `point` on purpose. Read left to
    # right they say: this number came from THIS instrument, and a random peak
    # would have scored THAT on the same films. A pointing game read without
    # both of those neighbours is the mistake this project already made once.
    ("map_source", "map", None),
    ("pointing_game", "point", "{:.4f}"),
    ("chance_pointing_game", "chance", "{:.4f}"),
    ("cam_pointing_game", "cam-pt", "{:.4f}"),
    ("mean_iou", "IoU", "{:.4f}"),
    # Sits next to IoU because they are the two halves of the same question:
    # does the map find the lesion when there is one, and does it stay quiet
    # when there is not. Lower is better here, which is why it is labelled with
    # what it counts rather than with a bare metric name.
    ("normal_flagged", "FP-map", "{:.4f}"),
    ("malignant_recall", "mal-rec", "{:.4f}"),
    ("macro_roc_auc", "ROC", "{:.4f}"),
    ("normal_specificity", "spec", "{:.4f}"),
    ("normal_called_malignant", "N->M", "{:.0f}"),
    ("lesion_loss", "seg-loss", "{:.4f}"),
]


def _table(rows: list[dict]) -> str:
    # Cells are formatted before the widths are measured, rather than assuming a
    # 9-character floor. A value wider than its column does not wrap, it shunts
    # every column to its right out of alignment for that one row, and a table
    # nobody can read down a column is the one thing this file exists to produce.
    formatted: list[list[str]] = []
    for row in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append("-")
            elif fmt is None:
                cells.append(str(value))
            else:
                cells.append(fmt.format(value))
        formatted.append(cells)

    widths = {
        key: max([len(header)] + [len(cells[i]) for cells in formatted])
        for i, (key, header, _) in enumerate(COLUMNS)
    }

    lines = ["  ".join(h.ljust(widths[k]) for k, h, _ in COLUMNS)]
    lines.append("  ".join("-" * widths[k] for k, _, _ in COLUMNS))
    for cells in formatted:
        lines.append(
            "  ".join(
                c.ljust(widths[k]) for c, (k, _, _) in zip(cells, COLUMNS, strict=True)
            )
        )
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
            # Training needs the same artefact detection as every other step --
            # see _run for why the original "train exits cleanly" assumption was
            # wrong, and what it cost. The signal is summary.json, NOT best.pt:
            # best.pt is rewritten once per improving epoch, so its presence says
            # nothing about completion, whereas summary.json is written exactly
            # once, after the last epoch.
            #
            # The directory is stamped with train.py's own start time, so it
            # cannot be named in advance. Anything already matching this tag is
            # snapshotted first and ignored, otherwise a re-run of a tag that
            # completed earlier would read the OLD summary.json in the seconds
            # before the new directory appears and declare victory instantly.
            preexisting = {p.name for p in reports.glob(f"{tag}-*")}

            def _train_artefact(_seen: set[str] = preexisting, _tag: str = tag) -> Path | None:
                fresh = [p for p in reports.glob(f"{_tag}-*") if p.name not in _seen]
                if not fresh:
                    return None
                return max(fresh, key=lambda p: p.stat().st_mtime) / "summary.json"

            if not _run(
                "train",
                [PYTHON, "scripts/train.py", *override_argv, "--tag", tag],
                log,
                expect=_train_artefact,
            ):
                results.append({"run": tag, "error": "train failed"})
                continue
            run_dir = _newest_run(tag, reports)

        if run_dir is None:
            results.append({"run": tag, "error": "no run directory"})
            continue
        checkpoint = str(run_dir / "best.pt")

        # calibrate/evaluate/gradcam get THE SAME override chain as train, and
        # leaving it off was a real bug rather than a tidy simplification.
        #
        # build_model_for_checkpoint honours the checkpoint's `model` block and
        # NOTHING ELSE, by design -- a checkpoint records where its data lived
        # when it was trained, which is not necessarily where yours is now. So
        # everything else still comes from YAML, and two things that matter came
        # from the wrong YAML:
        #
        #   data.image_size -- these scripts build their own transforms. A run
        #     trained at 384px was about to be calibrated, evaluated and scored
        #     at base.yaml's 256px, which would have quietly invalidated every
        #     row of Block B and Block C.
        #
        #   the threshold block -- full_run.yaml calibrates specificity_floor at
        #     0.80 and base.yaml calibrates sensitivity_floor at 0.95. Dropping
        #     the overrides moved w010 and w025 to a lesion threshold of ~0.16
        #     against v1.0.0's 0.4959. Promoting one of those would have shipped
        #     a far more trigger-happy operating point while every guarded
        #     metric still looked like an improvement, because the guards are
        #     computed from argmax and never see the threshold.
        #
        # daily_cycle.py still passes none and still works, which is the property
        # build_model_for_checkpoint exists to give it. The difference is that a
        # sweep KNOWS its recipe, so withholding it buys nothing.
        steps: list[tuple[str, list[str], Path]] = [
            (
                "calibrate",
                [PYTHON, "scripts/calibrate.py", "--checkpoint", checkpoint,
                 *override_argv],
                run_dir / "calibration.json",
            ),
            (
                "evaluate",
                [PYTHON, "scripts/evaluate.py", "--checkpoint", checkpoint,
                 *override_argv, "--split", args.split],
                run_dir / f"metrics_{args.split}.json",
            ),
            (
                "gradcam",
                [PYTHON, "scripts/gradcam_report.py", "--checkpoint", checkpoint,
                 *override_argv, "--split", args.split],
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
        "HOW TO READ IT: `map` says which explanation each row was scored on --\n"
        "  `cam` is Grad-CAM, `lesion` is the head's own map -- and `chance` is\n"
        "  what a peak dropped at random scores on the same films. A `point`\n"
        "  that does not clear `chance` is not a weak result, it is no result.\n"
        "  `cam-pt` is Grad-CAM on a lesion-head checkpoint: if `point` rises\n"
        "  while `cam-pt` does not, the run added a second output without moving\n"
        "  the classifier, which is not what the head was built for."
    )
    logger.info(
        "NOTHING was promoted. Read the table, pick one run, then register it:\n"
        "  python scripts/version_model.py register --run <run> --level major"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
