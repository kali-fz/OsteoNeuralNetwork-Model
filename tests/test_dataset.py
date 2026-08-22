"""Tests for the transform chain, datasets and dataloaders.

These are the assertions that stand between "the code runs" and "the tensors
mean what the model assumes they mean". Shape and dtype are the easy half; the
ones worth reading are the determinism split (val must be reproducible, train
must not be) and the aspect-ratio check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from onnm.dataset import (
    RobustDataset,
    build_dataloader,
    build_transforms,
    class_weights,
    resolve_column,
)


@pytest.fixture
def sample(jpeg_image: Path) -> dict:
    return {"image": str(jpeg_image), "label": 2, "image_id": "a", "patient_id": "p1"}


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_output_contract(cfg, sample: dict, mode: str) -> None:
    out = build_transforms(cfg, mode)(sample)
    size = int(cfg.data.image_size)
    channels = int(cfg.data.in_channels)

    assert out["image"].shape == (channels, size, size)
    assert out["image"].dtype == torch.float32
    assert torch.isfinite(out["image"]).all(), "non-finite pixels reach the model"
    assert out["label"].dtype == torch.long
    assert 0 <= int(out["label"]) < 3


def test_meta_is_dropped_by_default(cfg, sample: dict) -> None:
    """Metadata must not ride along into collation unless explicitly requested."""
    assert "image_meta_dict" not in build_transforms(cfg, "val")(sample)
    assert "image_meta_dict" in build_transforms(cfg, "val", keep_meta=True)(sample)


def test_normalisation_is_applied(cfg, sample: dict) -> None:
    """After ImageNet normalisation the data must have left the raw [0, 1] range."""
    image = build_transforms(cfg, "val")(sample)["image"]
    assert image.min() < 0.0, "ImageNet mean subtraction was not applied"
    assert -5.0 < float(image.mean()) < 5.0


def test_invalid_mode_rejected(cfg) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        build_transforms(cfg, "validation")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_val_transforms_are_deterministic(cfg, sample: dict) -> None:
    """Evaluation must be reproducible, or no reported metric is trustworthy."""
    transform = build_transforms(cfg, "val")
    first, second = transform(sample)["image"], transform(sample)["image"]
    torch.testing.assert_close(first, second)


def test_train_transforms_are_stochastic(cfg, sample: dict) -> None:
    """Augmentation must actually vary, or it is doing nothing at all.

    Sampled repeatedly because each individual transform is probabilistic: a
    single pair of identical outputs is expected occasionally, twenty is not.
    """
    transform = build_transforms(cfg, "train")
    outputs = [transform(sample)["image"] for _ in range(20)]
    assert any(
        not torch.allclose(outputs[0], other) for other in outputs[1:]
    ), "train augmentation never changed the image"


def test_train_transforms_seed_reproducibly(cfg, sample: dict) -> None:
    """Seeded runs must replay identically, so a result can be reproduced."""
    transform = build_transforms(cfg, "train")

    transform.set_random_state(seed=42)
    first = transform(sample)["image"]
    transform.set_random_state(seed=42)
    second = transform(sample)["image"]

    torch.testing.assert_close(first, second)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_aspect_ratio_is_preserved(cfg, wide_jpeg: Path) -> None:
    """A 4:1 film with a centred square must not come out holding a 4:1 blob.

    Lesion margin and periosteal reaction are morphological signs. Squashing a
    long-bone radiograph into a square deforms exactly the features the model
    is meant to read, so the chain resizes the longest side and pads the rest.
    """
    record = {"image": str(wide_jpeg), "label": 0, "image_id": "w", "patient_id": "p"}
    image = build_transforms(cfg, "val")(record)["image"][0].numpy()

    # Recover the bright square's bounding box.
    mask = image > (image.min() + 0.5 * (image.max() - image.min()))
    rows, cols = np.where(mask)
    assert rows.size > 0, "the square vanished during resizing"

    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    assert 0.75 < height / width < 1.33, (
        f"square became {height}x{width} -- aspect ratio was not preserved"
    )


def test_padding_is_present_for_non_square_input(cfg, wide_jpeg: Path) -> None:
    """A 4:1 input padded to a square must leave constant bands top and bottom."""
    record = {"image": str(wide_jpeg), "label": 0, "image_id": "w", "patient_id": "p"}
    image = build_transforms(cfg, "val")(record)["image"][0].numpy()

    row_variation = image.std(axis=1)
    assert row_variation[:8].mean() < row_variation[110:146].mean(), (
        "expected flat padding at the top and content in the middle"
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_dataloader_batches(cfg, records: list[dict]) -> None:
    loader = build_dataloader(cfg, "val", records=records, shuffle=False)
    batch = next(iter(loader))

    size = int(cfg.data.image_size)
    channels = int(cfg.data.in_channels)
    batch_size = min(int(cfg.loader.batch_size), len(records))

    assert batch["image"].shape == (batch_size, channels, size, size)
    assert batch["label"].shape == (batch_size,)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].dtype == torch.long


def test_dataloader_covers_every_record(cfg, records: list[dict]) -> None:
    loader = build_dataloader(cfg, "test", records=records, shuffle=False)
    seen = sum(int(batch["label"].shape[0]) for batch in loader)
    assert seen == len(records), "test loader dropped samples"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------
def test_robust_dataset_substitutes_bad_file(cfg, records: list[dict], corrupt_file: Path) -> None:
    """One undecodable file must not kill a run that is otherwise fine."""
    poisoned = [dict(records[0]) | {"image": str(corrupt_file)}, *records]
    dataset = RobustDataset(data=poisoned, transform=build_transforms(cfg, "val"))

    item = dataset[0]
    assert item["image"].shape == (3, int(cfg.data.image_size), int(cfg.data.image_size))
    assert str(corrupt_file) in dataset.failed_paths


def test_robust_dataset_gives_up_when_everything_fails(cfg, corrupt_file: Path) -> None:
    """Wholesale corruption must raise rather than loop forever."""
    broken = [
        {"image": str(corrupt_file), "label": 0, "image_id": str(i), "patient_id": str(i)}
        for i in range(3)
    ]
    dataset = RobustDataset(data=broken, transform=build_transforms(cfg, "val"), max_retries=4)

    with pytest.raises(RuntimeError, match="consecutive samples failed"):
        _ = dataset[0]


# ---------------------------------------------------------------------------
# Class weighting and schema resolution
# ---------------------------------------------------------------------------
def test_class_weights_favour_the_rare_class() -> None:
    """Weights must rank inversely to frequency and average to 1."""
    records = (
        [{"label": 0}] * 1879 + [{"label": 1}] * 1525 + [{"label": 2}] * 342
    )
    weights = class_weights(records)

    assert weights.shape == (3,)
    assert weights[2] > weights[1] > weights[0], "rare malignant class must weigh most"
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)


def test_class_weights_survive_an_absent_class() -> None:
    """A split missing a class must not produce inf or NaN weights."""
    weights = class_weights([{"label": 0}] * 10 + [{"label": 1}] * 5)
    assert torch.isfinite(weights).all()


def test_resolve_column_is_case_insensitive() -> None:
    import pandas as pd

    df = pd.DataFrame({"Image_ID": [1], "Tumor_Type": ["osteosarcoma"]})

    assert resolve_column(df, ["image_id"]) == "Image_ID"
    assert resolve_column(df, ["missing", "tumor_type"]) == "Tumor_Type"
    assert resolve_column(df, ["nope"]) is None


def test_resolve_column_required_raises_with_guidance() -> None:
    import pandas as pd

    from onnm.config import ConfigError

    with pytest.raises(ConfigError, match="verify_data.py"):
        resolve_column(pd.DataFrame({"a": [1]}), ["b"], required=True)
