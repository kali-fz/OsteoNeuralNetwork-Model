"""Every record in a batch must carry the same keys.

WHY THIS IS ITS OWN FILE
------------------------
``monai.data.utils.list_data_collate`` takes the key set of the first record in a
batch and indexes every other record with it. One record missing one key raises
``KeyError`` -- and it raises on the first step of epoch 1, *after* the whole
cache has been built, which on this dataset is several minutes of work before
anything says a word.

``build_records`` assembles its output from two branches that grew apart:

* BTXRD rows, from ``dataset.xlsx``
* external/community rows, from ``configs/controls_manifest.csv``

The second branch added ``_split`` (consumed by ``filter_by_split``) and, later,
the first branch added ``annotation`` for lesion-mask supervision. Neither key
existed on the other side. The failure was invisible until a manifest actually
had rows in it: with an empty manifest every record comes from one branch, so the
key sets trivially agree and nothing is wrong. The eight community rows approved
in August are what made it real.

Both keys are handled at their source now -- ``_split`` stripped once the filter
has used it, ``annotation`` defaulted on manifest rows -- but the invariant is
what matters, not either fix, because the next key added to one branch will be
just as quiet.
"""

from __future__ import annotations

import pandas as pd
import pytest
from monai.data.utils import list_data_collate

from onnm.dataset import build_dataloader, build_records


@pytest.fixture
def manifest(tmp_path, jpeg_image, png_image, cfg, monkeypatch):
    """A controls manifest with rows in it, which is when the bug appears."""
    path = tmp_path / "controls_manifest.csv"
    pd.DataFrame(
        [
            {
                "image": str(jpeg_image), "image_id": "ctrl-1", "label": 0,
                "patient_id": "ctrl-p1", "split": "train",
            },
            {
                "image": str(png_image), "image_id": "ctrl-2", "label": 1,
                "patient_id": "ctrl-p2", "split": "train",
            },
        ]
    ).to_csv(path, index=False)
    return path


def _key_sets(records) -> set[tuple[str, ...]]:
    return {tuple(sorted(r)) for r in records}


def test_records_from_both_branches_share_one_key_set(cfg, manifest, monkeypatch):
    """BTXRD rows and manifest rows must be indistinguishable to the collator."""
    monkeypatch.setitem(cfg._data["paths"], "controls_manifest", str(manifest))

    records = build_records(cfg, split="train")
    manifest_rows = [r for r in records if str(r["image_id"]).startswith("ctrl-")]
    assert manifest_rows, "the manifest rows were not picked up; the test proves nothing"

    key_sets = _key_sets(records)
    assert len(key_sets) == 1, (
        "records have "
        f"{len(key_sets)} different key sets: {sorted(key_sets)}. list_data_collate "
        "will raise KeyError on the first training step, after the cache is built."
    )


def test_split_bookkeeping_does_not_survive_into_the_batch(cfg, manifest, monkeypatch):
    """`_split` is consumed by filter_by_split and must not reach collation."""
    monkeypatch.setitem(cfg._data["paths"], "controls_manifest", str(manifest))
    records = build_records(cfg, split="train")
    assert all("_split" not in r for r in records)


def test_a_mixed_batch_actually_collates(cfg, manifest, monkeypatch):
    """The end-to-end version: put both kinds of record in one batch and collate it.

    The key-set assertions above would pass on records that still break for some
    other reason, so this drives the real collator rather than a proxy for it.
    """
    monkeypatch.setitem(cfg._data["paths"], "controls_manifest", str(manifest))
    records = build_records(cfg, split="train")

    btxrd = [r for r in records if not str(r["image_id"]).startswith("ctrl-")][:2]
    controls = [r for r in records if str(r["image_id"]).startswith("ctrl-")][:2]
    if not btxrd:
        pytest.skip("BTXRD is not on disk; the mixed-batch case needs both branches")

    loader = build_dataloader(cfg, "train", records=btxrd + controls, shuffle=False)
    batch = next(iter(loader))
    assert "image" in batch and "label" in batch


def test_collate_is_what_would_have_caught_it(records):
    """Pin the mechanism itself, so the reason for all of the above stays legible."""
    ragged = [dict(records[0]), dict(records[1])]
    ragged[0]["extra"] = "only on the first record"
    with pytest.raises(KeyError):
        list_data_collate(ragged)
