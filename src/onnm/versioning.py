"""Model version registry: what shipped, what it scored, and what is serving.

WHY THIS EXISTS
---------------
The community loop retrains on data the model itself helped collect. That is a
feedback loop, and feedback loops drift: a run can be worse than the one before
it while every individual step looks reasonable, and without a ledger the only
record of "the model used to be better" is a memory of a number.

So every training generation is registered here before anything is promoted,
and promotion is a *separate*, guarded act. A regression therefore costs
nothing: the new version is recorded as held, `reports/PRODUCTION` never moves,
and the previous checkpoint keeps serving. Rolling back is not a recovery
procedure, it is the default outcome of a bad run.

TWO FILES, ONE SOURCE OF TRUTH
------------------------------
``model_versions.json`` is the data. ``ONN.md`` is rendered from it by
:func:`render_markdown` and is never hand-edited -- a ledger that can disagree
with itself is worse than no ledger, and a test asserts the two are in step.

Both live at the repo root rather than under ``reports/``, which is gitignored.
The checkpoints cannot be committed (they are hundreds of megabytes, and the
weights derive from a CC BY-NC-ND dataset), so the ledger is the part of a
version that survives a fresh clone, and it has to be tracked.

VERSION NUMBERS
---------------
``MAJOR.MINOR.PATCH``, and the distinction is about *what changed*, not about
how much better it got:

    major   a different model: another architecture family, another task head.
    minor   a deliberate recipe change -- augmentation, loss, backbone settings.
            The comparison to the previous version is not apples to apples.
    patch   the same recipe, more data. This is what the daily community loop
            produces, and it is the only bump that is fully automatic.

A patch bump therefore means "same experiment, larger training set", which is
exactly the claim a daily retrain is entitled to make.

THE GUARDED METRICS
-------------------
Two, because the project is trying to get better at two different things and
one of them is invisible to the usual numbers:

    macro_roc_auc      ranking quality on the lesion task. Threshold-
                       independent, so it cannot be recovered by re-tuning.
    malignant_recall   the metric the clinical case actually rests on.
    misc_rejection     the share of confirmed non-radiographs the gate turns
                       away: bone versus not-bone, measured on real misuse
                       rather than on the handful of photographs the thresholds
                       were originally tuned against.

Accuracy is deliberately not among them -- "never malignant" scores 90.9% on
this dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = REPO_ROOT / "model_versions.json"
MARKDOWN_PATH = REPO_ROOT / "ONN.md"

#: Metrics compared when deciding whether a new version may be promoted, and
#: the direction that counts as better. All three are "higher is better", but
#: naming the map keeps a future lower-is-better metric from being added
#: silently on the wrong side of the comparison.
GUARDED_METRICS: tuple[str, ...] = (
    "macro_roc_auc",
    "malignant_recall",
    "misc_rejection",
)

#: How much a guarded metric may fall before promotion is refused.
#:
#: Not zero. Bootstrap noise on a 536-image test split moves malignant recall by
#: more than a point between identical runs, so a zero-tolerance gate would
#: block every version on measurement noise and teach whoever is watching to
#: override it -- which is worse than no gate. Not large either: 0.01 is well
#: inside the ±0.14 confidence interval on malignant recall, so this catches a
#: collapse, not a wobble.
REGRESSION_TOLERANCE = 0.01

#: Statuses a version can hold.
#:
#: `serving` is the one pinned in reports/PRODUCTION; `held` scored worse than
#: the incumbent and was deliberately not promoted; `superseded` served once and
#: has since been replaced.
STATUSES = ("serving", "held", "superseded")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


@dataclass
class Version:
    """One training generation."""

    version: str
    created_at: str
    run: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    community: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    parent: str | None = None
    #: Why promotion was refused, when it was. Empty on a promoted version.
    held_because: str = ""
    #: sha256 of this version's ``best.pt``.
    #:
    #: The one field that identifies a version from the *outside*. Run names are
    #: a local convention -- the hosted app calls its directory "hosted"
    #: regardless of which generation is in it -- so given only a deployed
    #: checkpoint, the digest is the only way back to a row in this ledger.
    #: That is what lets the app say which version it is actually serving, as
    #: opposed to which one it was configured to serve.
    checkpoint_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "run": self.run,
            "status": self.status,
            "metrics": self.metrics,
            "community": self.community,
            "note": self.note,
            "parent": self.parent,
            "held_because": self.held_because,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Version:
        return cls(
            version=data["version"],
            created_at=data.get("created_at", ""),
            run=data.get("run", ""),
            status=data.get("status", "held"),
            metrics=dict(data.get("metrics", {})),
            community=dict(data.get("community", {})),
            note=data.get("note", ""),
            parent=data.get("parent"),
            held_because=data.get("held_because", ""),
            checkpoint_sha256=data.get("checkpoint_sha256", ""),
        )


def parse_version(text: str) -> tuple[int, int, int]:
    """``"v1.0.1"`` or ``"1.0.1"`` -> ``(1, 0, 1)``."""
    cleaned = text.strip().lstrip("vV")
    parts = cleaned.split(".")
    if len(parts) != 3:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}") from exc
    return major, minor, patch


def format_version(parts: tuple[int, int, int]) -> str:
    return "v{}.{}.{}".format(*parts)


def bump(text: str, level: str = "patch") -> str:
    """Next version at the given level. ``patch`` is what a data-only retrain gets."""
    major, minor, patch = parse_version(text)
    if level == "major":
        return format_version((major + 1, 0, 0))
    if level == "minor":
        return format_version((major, minor + 1, 0))
    if level == "patch":
        return format_version((major, minor, patch + 1))
    raise ValueError(f"level must be major, minor or patch, not {level!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def load_registry(path: Path | str = REGISTRY_PATH) -> list[Version]:
    """Every registered version, oldest first. Empty when nothing is registered."""
    path = Path(path)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = [Version.from_dict(item) for item in data.get("versions", [])]
    return sorted(versions, key=lambda v: parse_version(v.version))


def save_registry(versions: list[Version], path: Path | str = REGISTRY_PATH) -> Path:
    path = Path(path)
    ordered = sorted(versions, key=lambda v: parse_version(v.version))
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "updated_at": _now(),
                "versions": [version.as_dict() for version in ordered],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def find_by_sha(versions: list[Version], sha256: str) -> Version | None:
    """Which version a checkpoint file is, by digest. None when unrecognised.

    Unrecognised is a real and useful answer, not a failure: it means the thing
    being served was never registered, which is exactly what a hand-uploaded or
    half-finished publish looks like from the outside.
    """
    if not sha256:
        return None
    wanted = sha256.strip().lower()
    for version in reversed(versions):
        if (version.checkpoint_sha256 or "").lower() == wanted:
            return version
    return None


def latest(versions: list[Version]) -> Version | None:
    return versions[-1] if versions else None


def serving(versions: list[Version]) -> Version | None:
    """The version currently pinned for inference, if any."""
    for version in reversed(versions):
        if version.status == "serving":
            return version
    return None


# ---------------------------------------------------------------------------
# The promotion decision
# ---------------------------------------------------------------------------
def compare(candidate: dict[str, float], incumbent: dict[str, float]) -> dict[str, float]:
    """Signed change per guarded metric. Missing on either side is skipped.

    Skipped rather than treated as zero: a metric the incumbent never measured
    says nothing about whether the candidate regressed, and scoring it as "no
    change" would let a genuinely worse run through on a technicality.
    """
    deltas: dict[str, float] = {}
    for name in GUARDED_METRICS:
        if name in candidate and name in incumbent:
            deltas[name] = float(candidate[name]) - float(incumbent[name])
    return deltas


def should_promote(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    tolerance: float = REGRESSION_TOLERANCE,
) -> tuple[bool, str]:
    """Whether a new version may take over serving. Returns ``(ok, reason)``.

    The first version is always promoted -- there is nothing to be worse than.
    After that, any guarded metric falling by more than ``tolerance`` blocks it,
    and the reason names every metric that fell so the ledger records what
    actually went wrong rather than "regression".

    Note what this does *not* do: it never demands improvement. A retrain that
    holds every metric flat is promoted, because the point of the daily loop is
    to accumulate data, and a generation that adds twelve images should not have
    to prove a measurable gain to be allowed to exist.
    """
    if incumbent is None:
        return True, "first registered version"

    deltas = compare(candidate, incumbent)
    if not deltas:
        return False, "no guarded metric could be compared against the incumbent"

    regressions = {
        name: delta for name, delta in deltas.items() if delta < -tolerance
    }
    if regressions:
        detail = ", ".join(f"{name} {delta:+.4f}" for name, delta in sorted(regressions.items()))
        return False, f"regressed beyond {tolerance:g}: {detail}"

    summary = ", ".join(f"{name} {delta:+.4f}" for name, delta in sorted(deltas.items()))
    return True, f"no regression ({summary})"


def register(
    versions: list[Version],
    *,
    run: str,
    metrics: dict[str, float],
    community: dict[str, Any] | None = None,
    note: str = "",
    level: str = "patch",
    version: str | None = None,
    checkpoint_sha256: str = "",
    tolerance: float = REGRESSION_TOLERANCE,
) -> tuple[list[Version], Version]:
    """Add a version and decide whether it serves. Returns ``(versions, added)``.

    Registration always happens. Promotion is conditional, and the two are
    separate on purpose: a held version is still a fact about the project worth
    keeping -- "we tried more data on this date and it got worse" is exactly the
    thing that is lost when only successes are recorded.
    """
    incumbent = serving(versions)
    previous = latest(versions)
    number = version or (bump(previous.version, level) if previous else "v1.0.0")
    if any(existing.version == number for existing in versions):
        raise ValueError(f"{number} is already registered")

    ok, reason = should_promote(metrics, incumbent.metrics if incumbent else None, tolerance)
    added = Version(
        version=number,
        created_at=_now(),
        run=run,
        status="serving" if ok else "held",
        metrics=dict(metrics),
        community=dict(community or {}),
        note=note,
        parent=previous.version if previous else None,
        held_because="" if ok else reason,
        checkpoint_sha256=checkpoint_sha256,
    )
    if ok and incumbent is not None:
        incumbent.status = "superseded"
    return [*versions, added], added


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return "-" if value in (None, "") else str(value)


def _serving_line(current: Version | None) -> str:
    return f"{current.version} (`{current.run}`)" if current else "nothing pinned"


def render_markdown(versions: list[Version]) -> str:
    """Render the ledger. Generated -- never hand-edit ``ONN.md``."""
    current = serving(versions)
    head = [
        "# ONN: model version ledger",
        "",
        "**Generated from `model_versions.json` by `scripts/version_model.py render`.**",
        "Do not edit by hand: the JSON is the source of truth and a test asserts",
        "this file is in step with it.",
        "",
        "Every training generation is registered here *before* anything is promoted,",
        "and promotion is a separate, guarded act. A run that regresses is recorded as",
        "`held`, `reports/PRODUCTION` does not move, and the previous checkpoint keeps",
        "serving, so a bad retrain costs a row in this table and nothing else.",
        "",
        "| level | means |",
        "|---|---|",
        "| major | a different model, another architecture family or task head |",
        "| minor | a deliberate recipe change (augmentation, loss, backbone) |",
        "| patch | the same recipe with more data, which is what the daily community loop produces |",
        "",
        f"**Serving now:** {_serving_line(current)}",
        "",
        "---",
        "",
        "## Versions",
        "",
    ]

    if not versions:
        return "\n".join([*head, "_Nothing registered yet._", ""])

    table = [
        "| version | date | status | run | macro ROC-AUC | malignant recall "
        "| misc rejection | community rows |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for version in reversed(versions):
        rows = version.community.get("lesion_rows_total", "")
        table.append(
            "| **{}** | {} | {} | `{}` | {} | {} | {} | {} |".format(
                version.version,
                version.created_at,
                version.status,
                version.run,
                _fmt(version.metrics.get("macro_roc_auc")),
                _fmt(version.metrics.get("malignant_recall")),
                _fmt(version.metrics.get("misc_rejection")),
                _fmt(rows),
            )
        )

    detail: list[str] = ["", "---", "", "## Detail", ""]
    for version in reversed(versions):
        detail.append(f"### {version.version}: {version.status}")
        detail.append("")
        detail.append(f"- **Registered** {version.created_at}")
        detail.append(f"- **Run** `{version.run}`")
        if version.parent:
            detail.append(f"- **Parent** {version.parent}")
        if version.note:
            detail.append(f"- **Note** {version.note}")
        if version.held_because:
            detail.append(f"- **Not promoted:** {version.held_because}")
        if version.checkpoint_sha256:
            detail.append(f"- **best.pt sha256** `{version.checkpoint_sha256}`")
        if version.community:
            bits = ", ".join(f"{k} = {v}" for k, v in sorted(version.community.items()))
            detail.append(f"- **Community data** {bits}")
        if version.metrics:
            detail.append("")
            detail.append("| metric | value |")
            detail.append("|---|---|")
            for name, value in sorted(version.metrics.items()):
                detail.append(f"| {name} | {_fmt(value)} |")
        detail.append("")

    return "\n".join([*head, *table, *detail])


def write_markdown(versions: list[Version], path: Path | str = MARKDOWN_PATH) -> Path:
    path = Path(path)
    path.write_text(render_markdown(versions), encoding="utf-8")
    return path
