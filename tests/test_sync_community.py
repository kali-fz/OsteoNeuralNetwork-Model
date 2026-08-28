"""Tests for the approval-to-training path.

``sync_community.py`` is the one command between "I approved these images" and
"the next training run uses them", and it has two properties worth defending.

The first is **idempotency**. It will be run repeatedly, sometimes twice by
accident, and sometimes on a machine whose previous run was killed halfway.
Rebuilding the cumulative manifest from the batch directories rather than
appending to it makes that safe: the same store always yields the same manifest.
An appending implementation would duplicate rows into the training set, which
does not raise, does not look wrong in any log, and quietly reweights the
classes it happens to duplicate.

The second is that the **misc/lesion separation survives the rebuild**. It is
enforced at export time, but the rebuild reads files rather than the API, and a
confirmed non-radiograph appearing in ``controls_manifest.csv`` is the hotdog
labelled "normal bone" arriving by a different route.

No network: the export result is a plain dict, so the whole path from claimed
rows to cumulative manifest runs against fixtures.
"""

from __future__ import annotations

import base64
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_batch import (  # noqa: E402
    MANIFEST_COLUMNS,
    OOD_MANIFEST_COLUMNS,
    manifest_path,
    write_batch,
)
from sync_community import _rebuild  # noqa: E402


def _png(value: int = 128) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.full((256, 256), value, dtype=np.uint8), mode="L").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _row(submission_id: str, label: str, bucket: str, shade: int = 100) -> dict:
    return {
        "submission_id": submission_id,
        "created_at": "t",
        "admin_bucket": bucket,
        "admin_label": label,
        "image_b64": _png(shade),
        "model_label": "benign",
        "lesion_probability": 0.5,
        "checkpoint": "full",
        "ood_flagged": 0,
        "ood_score": None,
        "triage_bucket": bucket,
        "user_says_wrong": 0,
    }


def _write(batch_id: str, rows: list[dict], store: Path) -> dict | None:
    return write_batch({"batch_id": batch_id, "rows": rows}, store / batch_id)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    directory = tmp_path / "community"
    directory.mkdir()
    return directory


def _manifest_rows(target: Path) -> list[dict]:
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_batch_splits_by_what_it_retrains(store: Path) -> None:
    summary = _write(
        "batch-1",
        [
            _row("bone-1", "benign", "valid_bone"),
            _row("hotdog-1", "misc", "misc", shade=240),
        ],
        store,
    )
    assert summary is not None
    assert summary["lesion"]["count"] == 1
    assert summary["ood"]["count"] == 1
    assert (store / "batch-1" / "images" / "bone-1.png").is_file()
    assert (store / "batch-1" / "ood_negatives" / "hotdog-1.png").is_file()


def test_rebuilding_twice_produces_the_same_manifest(store: Path, tmp_path: Path) -> None:
    """The property that makes it safe to re-run after an interrupted sync."""
    _write("batch-1", [_row("bone-1", "benign", "valid_bone")], store)
    _write("batch-2", [_row("bone-2", "malignant", "valid_bone", shade=60)], store)

    target = tmp_path / "controls_manifest.csv"
    first = _rebuild(store, "manifest.csv", target, MANIFEST_COLUMNS)
    text = target.read_text(encoding="utf-8")
    second = _rebuild(store, "manifest.csv", target, MANIFEST_COLUMNS)

    assert [r["image_id"] for r in first] == ["bone-1", "bone-2"]
    assert first == second
    assert target.read_text(encoding="utf-8") == text


def test_a_second_sync_does_not_duplicate_earlier_rows(store: Path, tmp_path: Path) -> None:
    """Duplicated rows would reweight classes without raising anything.

    This is why the manifest is rebuilt from the batch directories rather than
    appended to: a duplicate is invisible in every log the training run writes.
    """
    target = tmp_path / "controls_manifest.csv"
    _write("batch-1", [_row("bone-1", "benign", "valid_bone")], store)
    _rebuild(store, "manifest.csv", target, MANIFEST_COLUMNS)
    _write("batch-2", [_row("bone-2", "normal", "valid_bone", shade=30)], store)
    rows = _rebuild(store, "manifest.csv", target, MANIFEST_COLUMNS)

    ids = [row["image_id"] for row in rows]
    assert ids == ["bone-1", "bone-2"]
    assert len(ids) == len(set(ids))


def test_a_confirmed_non_radiograph_never_reaches_the_lesion_manifest(
    store: Path, tmp_path: Path
) -> None:
    """The invariant, restated at the rebuild step.

    The split is decided at export time, but the rebuild reads files rather than
    the API, so it is checked again where the training set is actually written.
    """
    _write(
        "batch-1",
        [
            _row("bone-1", "benign", "valid_bone"),
            _row("hotdog-1", "misc", "misc", shade=240),
            # The other direction: a hotdog the classifier confidently diagnosed,
            # filed as a contradiction. Still a non-radiograph.
            _row("hotdog-2", "misc", "contradiction", shade=250),
        ],
        store,
    )
    lesion = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    ood = _rebuild(store, "ood_manifest.csv", tmp_path / "o.csv", OOD_MANIFEST_COLUMNS)

    assert [r["image_id"] for r in lesion] == ["bone-1"]
    assert sorted(r["image_id"] for r in ood) == ["hotdog-1", "hotdog-2"]
    assert all(row["is_radiograph"] == "0" for row in ood)
    assert "label" not in OOD_MANIFEST_COLUMNS, (
        "the OOD manifest must have no column able to express a lesion class"
    )


def test_a_corrected_false_rejection_retrains_the_classifier(
    store: Path, tmp_path: Path
) -> None:
    """A radiograph the gate wrongly turned away is a bone film, not misuse.

    The label decides the destination, not the bucket: the bucket records that
    the gate was wrong, the label records what the image actually was.
    """
    _write("batch-1", [_row("falserej-1", "normal", "contradiction")], store)
    lesion = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    assert [r["image_id"] for r in lesion] == ["falserej-1"]
    assert lesion[0]["label"] == "0"


def test_a_row_whose_image_vanished_is_dropped_not_carried(
    store: Path, tmp_path: Path
) -> None:
    """Otherwise the shortfall surfaces as a warning inside a training log.

    ``build_records`` skips a manifest row whose file is missing, so carrying it
    would mean the training set is quietly smaller than the manifest claims. The
    Drive store not being mounted is the ordinary way this happens.
    """
    _write("batch-1", [_row("bone-1", "benign", "valid_bone"),
                       _row("bone-2", "normal", "valid_bone", shade=30)], store)
    (store / "batch-1" / "images" / "bone-1.png").unlink()

    rows = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    assert [r["image_id"] for r in rows] == ["bone-2"]


def test_a_store_outside_the_repo_gets_absolute_paths(tmp_path: Path) -> None:
    """Colab keeps the store on Drive, which has no repo-relative form at all.

    ``Path.relative_to`` raises rather than inventing one, so the manifest falls
    back to an absolute path -- which ``build_records`` passes through unchanged.
    """
    outside = manifest_path(tmp_path / "images" / "x.png")
    assert Path(outside).is_absolute()

    inside = manifest_path(REPO_ROOT / "data" / "community" / "b" / "images" / "x.png")
    assert inside == "data/community/b/images/x.png"


def test_community_rows_are_pinned_to_the_train_split(store: Path, tmp_path: Path) -> None:
    """Val and test stay pure BTXRD, or no score in overview.md is comparable."""
    _write("batch-1", [_row("bone-1", "benign", "valid_bone")], store)
    rows = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    assert {row["split"] for row in rows} == {"train"}


def _batch_manifest(store: Path, batch_id: str) -> list[dict]:
    """The rows exactly as ``write_batch`` wrote them, before any rebuild."""
    path = store / batch_id / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_unrelated_uploads_get_different_patient_groups(store: Path, tmp_path: Path) -> None:
    """Grouping must not collapse unrelated images into one group.

    Two different pictures are two groups, which is what lets make_splits.py
    distribute them independently, and what keeps community rows from ever
    colliding with a BTXRD surrogate patient id.
    """
    _write("batch-1", [_row("bone-1", "benign", "valid_bone"),
                       _row("bone-2", "normal", "valid_bone", shade=30)], store)
    rows = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    groups = [row["patient_id"] for row in rows]
    assert len(set(groups)) == len(groups) == 2
    assert all(group.startswith("community-") for group in groups)


def test_the_same_image_uploaded_twice_is_one_patient_group(store: Path) -> None:
    """The property that stops a repeat upload inflating the validation score.

    Two submissions of the identical file are two submission ids, and under the
    previous rule that made them two patient groups -- so make_splits.py, whose
    entire job is keeping a group on one side of a split, was free to train on
    one copy and validate on the other. The score then partly measured recall of
    an image the model had already seen.

    Byte-identical images therefore share a group, and this is asserted against
    the batch manifest rather than the rebuilt one because the rebuild drops the
    second copy outright -- a different guarantee, tested below.
    """
    _write("batch-1", [_row("first-upload", "malignant", "valid_bone"),
                       _row("second-upload", "malignant", "valid_bone")], store)

    rows = _batch_manifest(store, "batch-1")
    assert [row["image_id"] for row in rows] == ["first-upload", "second-upload"]
    assert rows[0]["sha256"] == rows[1]["sha256"], "fixture should produce one image twice"
    assert rows[0]["patient_id"] == rows[1]["patient_id"]


def test_a_repeat_upload_appears_once_in_the_rebuilt_manifest(
    store: Path, tmp_path: Path
) -> None:
    """One distinct image, one training row, however many times it was sent.

    People test a tool by scanning the same file several times, and each of those
    scans is a separate submission that can be approved separately. Without this
    the picture enters the training set three times and counts three times in the
    loss, silently reweighting whichever class it belongs to.
    """
    _write("batch-1", [_row("upload-a", "normal", "valid_bone"),
                       _row("upload-b", "normal", "valid_bone"),
                       _row("upload-c", "normal", "valid_bone", shade=30)], store)

    rows = _rebuild(store, "manifest.csv", tmp_path / "m.csv", MANIFEST_COLUMNS)
    assert len(rows) == 2, "the two identical uploads should collapse to one row"
    assert len({row["sha256"] for row in rows}) == 2
    # The first copy is the one kept, so repeated rebuilds agree with each other.
    assert [row["image_id"] for row in rows] == ["upload-a", "upload-c"]


def test_deduplication_is_stable_across_rebuilds(store: Path, tmp_path: Path) -> None:
    """Idempotency, restated for the dedup path.

    The manifest is a derived file rebuilt on every sync. If dedup picked a
    different survivor each time, the training set would change without anything
    being approved, which is exactly the class of silent drift this module's
    rebuild-don't-append design exists to prevent.
    """
    _write("batch-1", [_row("upload-a", "benign", "valid_bone"),
                       _row("upload-b", "benign", "valid_bone")], store)

    first = _rebuild(store, "manifest.csv", tmp_path / "one.csv", MANIFEST_COLUMNS)
    second = _rebuild(store, "manifest.csv", tmp_path / "two.csv", MANIFEST_COLUMNS)
    assert first == second
    assert tmp_path.joinpath("one.csv").read_bytes() == tmp_path.joinpath("two.csv").read_bytes()
