"""Score the bone-versus-not-bone gate against reviewed data.

WHY THIS IS A SEPARATE MEASUREMENT
----------------------------------
Every headline number in this project is about the lesion task: can the model
tell normal from benign from malignant, given a radiograph. None of them say
anything about the question a public deployment actually faces first, which is
whether the thing it was handed is a radiograph at all.

That gate (``onnm.ood`` stage 1) is four hand-tuned thresholds. It was
calibrated by looking at BTXRD and a handful of photographs, and until the
community loop started recording rejections there was no dataset on which it
could be scored at all -- so "the model is getting better at bone versus misc"
was not a claim anyone could check.

This module makes it checkable. Two rates, deliberately reported separately
rather than folded into one score:

    misc_rejection    of the confirmed non-radiographs a human reviewed, what
                      share does the gate turn away? Higher is better.
    bone_acceptance   of known radiographs, what share does the gate let
                      through? Higher is better.

They trade against each other -- a gate that rejects everything scores 1.0 on
the first and 0.0 on the second -- so a single number would hide exactly the
failure worth catching. The version ledger guards ``misc_rejection`` and prints
``bone_acceptance`` beside it for that reason.

A KNOWN UNDERCOUNT
------------------
Shared images are stored as single-channel PNGs, because the storage path
de-identifies by re-encoding to greyscale. The colorfulness check -- the one
that catches a photograph fastest -- therefore cannot fire on the stored copy
the way it did on the upload. ``misc_rejection`` measured here is a **lower
bound** on the live gate, and a miss is not proof the gate missed it in
production. Stated in the result as ``greyscale_lower_bound`` so a reader of the
ledger is not left to rediscover it.

Torch-free: numpy, PIL and ``onnm.ood`` only, so it runs in the daily cycle
without loading the model at all.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .ood import validate_payload
from .utils import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GateReport:
    """How well the gate separates radiographs from everything else."""

    misc_total: int = 0
    misc_rejected: int = 0
    bone_total: int = 0
    bone_accepted: int = 0
    #: Names of the confirmed non-radiographs the gate let through. These are
    #: the interesting failures: each one reached the classifier and received a
    #: clinical-sounding verdict.
    misses: list[str] = field(default_factory=list)
    greyscale_lower_bound: bool = True

    @property
    def misc_rejection(self) -> float | None:
        """Share of confirmed misuse the gate turns away. None with no data."""
        if not self.misc_total:
            return None
        return self.misc_rejected / self.misc_total

    @property
    def bone_acceptance(self) -> float | None:
        """Share of known radiographs the gate lets through. None with no data."""
        if not self.bone_total:
            return None
        return self.bone_accepted / self.bone_total

    def as_dict(self) -> dict:
        return {
            "misc_total": self.misc_total,
            "misc_rejected": self.misc_rejected,
            "misc_rejection": self.misc_rejection,
            "bone_total": self.bone_total,
            "bone_accepted": self.bone_accepted,
            "bone_acceptance": self.bone_acceptance,
            "misses": self.misses,
            "greyscale_lower_bound": self.greyscale_lower_bound,
        }


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _accepts(path: Path) -> bool | None:
    """Whether the gate accepts this file as a radiograph. None if unreadable."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    return validate_payload(payload, path.name).is_radiograph


def evaluate_gate(
    ood_manifest: Path | str,
    bone_images: list[Path] | None = None,
) -> GateReport:
    """Score the gate on reviewed non-radiographs, and optionally on radiographs.

    ``ood_manifest`` is the cumulative manifest written by
    ``scripts/sync_community.py``: every row is an image a human confirmed is
    not a bone radiograph. ``bone_images`` is any set of known radiographs --
    the daily cycle passes a sample of BTXRD -- and exists so the rejection rate
    is never read without the cost of achieving it.

    An empty or missing manifest yields a report with no rates rather than an
    error, because "nobody has approved any misuse yet" is the ordinary state on
    day one and must not fail the pipeline.
    """
    report = GateReport()

    manifest = Path(ood_manifest)
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = _resolve(row["image"])
                if not path.is_file():
                    continue
                accepted = _accepts(path)
                if accepted is None:
                    continue
                report.misc_total += 1
                if accepted:
                    report.misses.append(row.get("image_id") or path.stem)
                else:
                    report.misc_rejected += 1
    else:
        logger.info("no OOD manifest at %s -- nothing confirmed as misuse yet", manifest)

    for path in bone_images or []:
        accepted = _accepts(Path(path))
        if accepted is None:
            continue
        report.bone_total += 1
        report.bone_accepted += int(accepted)

    return report
