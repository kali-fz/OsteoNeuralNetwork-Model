"""Dataset construction: records, MONAI transform chains, and dataloaders.

The pipeline is deliberately split into a *deterministic prefix* (load, intensity
normalisation, resize) and a *stochastic suffix* (augmentation). MONAI's
``CacheDataset`` caches exactly up to the first random transform, so this split
is what makes caching worth anything: the expensive JPEG decode and resize run
once, the cheap augmentations run every epoch.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    RandGaussianNoised,
    RandHistogramShiftd,
    RandRotated,
    RandZoomd,
    RepeatChanneld,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
)

from . import CLASS_NAMES
from .config import REPO_ROOT, Config, ConfigError
from .io_radiograph import LoadRadiographd, RadiographReadError
from .utils import get_logger, load_json

logger = get_logger(__name__)

MODES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Schema resolution
#
# Column names live in `configs/base.yaml`, not in this file, and were read off
# the actual BTXRD release with `scripts/verify_data.py --dump-schema`. Lookup
# stays case-insensitive and accepts a list of candidates so that a re-release
# with renamed columns is a config edit rather than a code change.
# ---------------------------------------------------------------------------
def resolve_column(
    df: pd.DataFrame, candidates: str | Sequence[str], required: bool = False
) -> str | None:
    """Return the first candidate column present in ``df``, matched case-insensitively."""
    if isinstance(candidates, str):
        candidates = [candidates]
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        hit = lookup.get(str(candidate).strip().lower())
        if hit is not None:
            return hit
    if required:
        raise ConfigError(
            f"none of {list(candidates)} found in CSV columns {list(df.columns)}; "
            "run `python scripts/verify_data.py --dump-schema` and update configs/base.yaml"
        )
    return None


def read_table(cfg: Config) -> pd.DataFrame:
    """Load the BTXRD metadata table.

    The release ships ``dataset.xlsx``, not the ``dataset.csv`` the paper's prose
    implies, so both are supported and dispatch is on the file extension.
    """
    data_root = cfg.resolve_path("paths.data_root")
    table_path = data_root / cfg.paths.table_name

    if not table_path.is_file():
        raise FileNotFoundError(
            f"{table_path} not found. Run `python scripts/download_btxrd.py` first."
        )

    if table_path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(table_path, sheet_name=cfg.paths.get("table_sheet", 0))
        except ImportError as exc:
            raise ImportError("reading .xlsx needs openpyxl: pip install openpyxl") from exc
    else:
        df = pd.read_csv(table_path)

    # Excel round-trips leave stray whitespace in headers often enough to be
    # worth normalising once here rather than debugging later.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def map_labels(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, dict[str, int]]:
    """Derive class indices 0/1/2 from BTXRD's one-hot indicator columns.

    There is no categorical diagnosis column. The three relevant indicators nest:
    ``tumor=0`` is normal, and among ``tumor=1`` rows exactly one of ``benign``
    or ``malignant`` is set. Malignant is tested first so a row flagged both ways
    resolves to the more serious call -- under-calling a cancer is the costlier
    error.

    Rows that satisfy no rule become ``-1`` and are reported by the caller rather
    than dropped in silence, because an unmapped row is most likely a malignant
    one and losing those would quietly gut the rarest class.
    """
    labels_cfg = cfg.labels
    class_to_idx = {name: idx for idx, name in enumerate(labels_cfg.classes)}

    tumor_col = resolve_column(df, labels_cfg.tumor_column, required=True)
    rules: list[tuple[str, int]] = []
    for entry in labels_cfg.class_columns:
        column = resolve_column(df, entry["column"], required=True)
        rules.append((column, class_to_idx[entry["class"]]))

    def to_index(row: pd.Series) -> int:
        if _as_flag(row[tumor_col]) == 0:
            return class_to_idx["normal"]
        for column, index in rules:
            if _as_flag(row[column]) == 1:
                return index
        return -1  # tumour present but neither benign nor malignant flagged

    logger.info(
        "labels derived from indicator columns: tumor=%r, %s",
        tumor_col,
        ", ".join(f"{c}->{cfg.labels.classes[i]}" for c, i in rules),
    )
    return df.apply(to_index, axis=1), class_to_idx


def _as_flag(value: Any) -> int:
    """Coerce a one-hot cell to 0/1, tolerating Excel's float and string forms."""
    try:
        return 1 if int(float(value)) == 1 else 0
    except (TypeError, ValueError):
        return 1 if str(value).strip().lower() in {"1", "y", "yes", "true"} else 0


def derive_groups(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Reconstruct a surrogate patient identifier for leakage-free splitting.

    BTXRD ships no patient id, but the images are plainly not independent: 2826
    of 3746 fall into runs of consecutive ``image_id`` values that agree on
    centre, age, sex, anatomy and diagnosis while differing in view -- multiple
    projections of one patient. Splitting per image would scatter those siblings
    across train and test, so the reported score would measure memorisation of a
    lesion the model had already seen from another angle.

    A new group starts when any of those attributes changes, or when the numeric
    part of the id is not consecutive with the previous row. The heuristic can
    merge two genuinely distinct patients who are adjacent and identical on every
    recorded field; that direction of error only withholds training data, which
    is the safe way to be wrong.
    """
    if str(cfg.split.group_strategy) == "image":
        logger.warning(
            "group_strategy=image: every image is its own group, so multiple views of "
            "one patient may straddle the split. Disclose this alongside any result."
        )
        return df[resolve_column(df, cfg.labels.id_column, required=True)].astype(str)

    id_col = resolve_column(df, cfg.labels.id_column, required=True)
    key_cols: list[str] = []
    for candidate in (
        list(cfg.columns.demographic)
        + list(cfg.columns.anatomy)
        + [cfg.labels.tumor_column]
        + [e["column"] for e in cfg.labels.class_columns]
        + list(cfg.labels.subtype_columns.benign)
        + list(cfg.labels.subtype_columns.malignant)
    ):
        resolved = resolve_column(df, candidate)
        if resolved is not None and resolved not in key_cols:
            key_cols.append(resolved)

    numbers = (
        df[id_col].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    )
    keys = df[key_cols].astype(str).agg("|".join, axis=1)

    order = numbers.argsort(kind="stable")
    group_ids = pd.Series(index=df.index, dtype=object)
    current = 0
    prev_key: str | None = None
    prev_num: int | None = None

    for position in order:
        idx = df.index[position]
        num = numbers.iloc[position]
        key = keys.iloc[position]
        consecutive = (
            prev_num is not None
            and num is not pd.NA
            and int(num) == prev_num + 1
        )
        if prev_key is None or key != prev_key or (
            bool(cfg.split.require_consecutive_ids) and not consecutive
        ):
            current += 1
        group_ids.at[idx] = f"G{current:05d}"
        prev_key = key
        prev_num = int(num) if num is not pd.NA else None

    n_groups = group_ids.nunique()
    logger.info(
        "surrogate patient grouping: %d groups over %d images (%d images share a group)",
        n_groups,
        len(df),
        len(df) - sum(1 for _, c in group_ids.value_counts().items() if c == 1),
    )
    return group_ids


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def build_records(cfg: Config, split: str | None = None) -> list[dict[str, Any]]:
    """Build the list of ``{"image", "label", "image_id", "patient_id"}`` dicts.

    Every path is checked for existence up front. Missing or unmapped rows are
    dropped with a counted summary, so a broken extraction surfaces here in
    seconds rather than as a crash at epoch three.
    """
    data_root = cfg.resolve_path("paths.data_root")
    images_dir = data_root / cfg.paths.images_dirname

    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found; check paths.images_dirname")

    df = read_table(cfg)
    logger.info("read %s: %d rows, %d columns", cfg.paths.table_name, len(df), len(df.columns))

    label_series, _ = map_labels(df, cfg)
    df = df.assign(_label=label_series, _group=derive_groups(df, cfg))

    id_col = resolve_column(df, cfg.labels.id_column, required=True)

    # Index the images directory by stem. The table records every id with a
    # ".jpeg" extension, but the release mixes ".jpeg" and ".jpg" on disk, so
    # matching on the literal filename alone drops real files.
    by_stem: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if path.is_file():
            by_stem.setdefault(path.stem.lower(), path)

    records: list[dict[str, Any]] = []
    n_unmapped = 0
    n_missing = 0
    for _, row in df.iterrows():
        if row["_label"] < 0:
            n_unmapped += 1
            continue

        raw_id = str(row[id_col]).strip()
        candidate = images_dir / raw_id
        path = candidate if candidate.is_file() else by_stem.get(Path(raw_id).stem.lower())
        if path is None:
            n_missing += 1
            continue

        records.append(
            {
                "image": str(path),
                "label": int(row["_label"]),
                "image_id": Path(raw_id).stem,
                "patient_id": str(row["_group"]),
            }
        )

    # External controls carry their own verified label, provenance and split.
    # They are optional so the original BTXRD-only workflow remains unchanged.
    controls_path = cfg.resolve_path("paths.controls_manifest")
    if controls_path.is_file():
        controls = pd.read_csv(controls_path)
        required = {"image", "image_id", "label", "patient_id", "split"}
        missing = required - set(controls.columns)
        if missing:
            raise ConfigError(f"{controls_path} is missing columns: {sorted(missing)}")
        for _, row in controls.iterrows():
            raw_image = Path(str(row["image"])).expanduser()
            image = raw_image if raw_image.is_absolute() else (REPO_ROOT / raw_image).resolve()
            if image.is_file() and int(row["label"]) == 0:
                records.append({"image": str(image), "label": 0, "image_id": str(row["image_id"]),
                                "patient_id": str(row["patient_id"]), "_split": str(row["split"])})

    if n_unmapped:
        logger.warning(
            "%d rows are flagged tumor=1 but neither benign nor malignant, and were "
            "dropped. Check labels.class_columns against the real indicator columns -- "
            "an unmapped row is most likely malignant, the class least able to lose cases.",
            n_unmapped,
        )
    if n_missing:
        logger.warning("%d rows referenced image files that do not exist", n_missing)
    if not records:
        raise RuntimeError(
            "build_records produced 0 usable records -- the label map or image paths are wrong"
        )

    if split is not None:
        records = filter_by_split(records, cfg, split)

    counts = np.bincount([r["label"] for r in records], minlength=len(CLASS_NAMES))
    logger.info(
        "records[%s]: %d total (%s)",
        split or "all",
        len(records),
        ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts, strict=False)),
    )
    return records


def filter_by_split(
    records: list[dict[str, Any]], cfg: Config, split: str
) -> list[dict[str, Any]]:
    """Keep only records whose ``image_id`` belongs to the named split."""
    if split not in MODES:
        raise ValueError(f"split must be one of {MODES}, got {split!r}")

    splits_path = cfg.resolve_path("paths.splits_file")
    if not splits_path.is_file():
        raise FileNotFoundError(
            f"{splits_path} not found. Run `python scripts/make_splits.py` first."
        )
    splits = load_json(splits_path)
    wanted = set(splits[split])
    subset = [r for r in records if r.get("_split") == split or r["image_id"] in wanted]
    if not subset:
        raise RuntimeError(f"split {split!r} matched 0 records -- splits.json is stale")
    return subset


def class_weights(
    records: list[dict[str, Any]], num_classes: int = 3, beta: float = 1.0
) -> torch.Tensor:
    """Inverse-frequency class weights, tempered by ``beta`` and normalised to mean 1.

    Used as the ``alpha`` term of the focal loss. Normalising to mean 1 keeps the
    loss on roughly the same scale as unweighted cross-entropy, so the learning
    rate does not have to be retuned when weighting is toggled.

    ``beta`` controls how hard the correction pushes:

    * ``1.0`` -- full inverse frequency. On BTXRD that weights malignant about
      3.7x normal, which maximises sensitivity and is the right default when a
      missed cancer is the dominant cost.
    * ``0.5`` -- square root. The standard compromise when full weighting is
      over-calling.
    * ``0.0`` -- uniform. No class weighting; use this when a balanced sampler
      is already handling the imbalance.

    This is the most direct knob on the sensitivity/specificity trade-off, so it
    is exposed rather than hard-coded: if normal controls are being called as
    lesions, this is the first number to lower.
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    counts = np.bincount([r["label"] for r in records], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (num_classes * counts)) ** float(beta)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def sample_weights(records: list[dict[str, Any]], num_classes: int = 3) -> torch.Tensor:
    """Per-sample draw probabilities that equalise the classes in a batch.

    Each sample is weighted by ``1 / count(its class)``, so every class
    contributes the same total mass and ``WeightedRandomSampler`` draws them in
    equal proportion. With BTXRD's 70% split that turns a 16/13/3 batch into
    roughly 11/11/11.

    Sampling with replacement means the 240-odd malignant training images are
    seen several times per epoch. That is the point, and it is also the risk:
    combined with a weighted loss it double-corrects the imbalance. See
    ``onnm.losses`` and ``loader.balanced_sampler`` in the config.
    """
    labels = np.asarray([r["label"] for r in records], dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return torch.as_tensor(1.0 / counts[labels], dtype=torch.double)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _foreground_selector(threshold: float):
    """Build a ``select_fn`` for ``CropForegroundd`` that cannot select nothing.

    The obvious ``lambda x: x > threshold`` has a failure mode that only shows up
    on real data: a film that is entirely below the threshold -- a blank export,
    a badly under-exposed study, a mask image -- selects an empty region, the
    crop collapses a spatial axis to zero, and the *next* transform is the one
    that raises. The traceback then points at ``Resized`` and says nothing about
    the crop, on an image nobody has looked at.

    So an empty selection falls back to keeping the whole frame. A blank film is
    then merely uninformative rather than fatal, which is the correct handling:
    ``RobustDataset`` should not have to substitute a sample over this, and an
    evaluation run must not lose an image to it.
    """

    def select(image):
        mask = image > threshold
        if bool(mask.any()):
            return mask
        logger.warning(
            "no pixel exceeds the %.3f foreground threshold; keeping the full frame "
            "uncropped. The image may be blank or severely under-exposed.",
            threshold,
        )
        return image >= image.min()  # all-True, and works for numpy and torch alike

    return select


def build_transforms(cfg: Config, mode: str, keep_meta: bool = False) -> Compose:
    """Assemble the MONAI transform chain for one mode.

    Args:
        cfg: Loaded project config.
        mode: ``"train"`` adds augmentation; ``"val"``/``"test"`` are deterministic.
        keep_meta: Retain ``image_meta_dict``. Useful for the sanity notebook and
            for Grad-CAM overlays that need the pre-resize geometry; dropped in
            the training loop so it does not ride along through collation.

    Note:
        With ``data.crop_foreground`` enabled the geometry between the original
        image and the model input is no longer the scale-and-pad that
        ``explainability.map_box_to_model_space`` reproduces, so ground-truth
        lesion-box scoring is disabled rather than reported wrongly.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    data = cfg.data
    size = int(data.image_size)
    lower, upper = (float(v) for v in data.intensity_percentiles)

    # -- deterministic prefix: cached by CacheDataset ----------------------
    stages: list[Any] = [
        LoadRadiographd(keys=["image"]),
        # "no_channel" is explicit on purpose: MONAI's inference guesses wrong
        # on square images, where H, W and a channel axis are indistinguishable.
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        # Percentile scaling rather than min-max: radiographs carry collimation
        # borders and burned-in L/R markers at the extremes of the histogram,
        # and a single saturated marker pixel would otherwise compress the
        # entire diagnostic range into a narrow band.
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=lower, upper=upper, b_min=0.0, b_max=1.0,
            clip=True, relative=False,
        ),
    ]

    # -- optional foreground crop (deterministic, so it stays in the prefix) --
    # Runs after intensity scaling so the threshold is in [0, 1] and means the
    # same thing on a 12-bit DICOM and an 8-bit JPEG.
    if bool(data.get("crop_foreground", False)):
        stages.append(
            CropForegroundd(
                keys=["image"],
                source_key="image",
                select_fn=_foreground_selector(float(data.get("crop_threshold", 0.05))),
                margin=int(data.get("crop_margin", 8)),
                allow_smaller=True,
            )
        )

    stages += [
        # Resize the LONGEST side, then pad -- preserving aspect ratio. Lesion
        # margin and periosteal reaction are morphological signs; squashing a
        # long-bone film into a square deforms exactly the cues that matter.
        Resized(keys=["image"], spatial_size=size, size_mode="longest", mode="bilinear"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(size, size), mode="constant"),
    ]

    # -- stochastic suffix: re-run every epoch -----------------------------
    if mode == "train":
        aug = cfg.augment
        stages += [
            # Horizontal only. Left and right limbs are both anatomically valid,
            # so a mirror is a real radiograph. A VERTICAL flip is not: an
            # upside-down film never occurs, and training on one teaches the
            # model to be invariant to something that carries real information.
            RandFlipd(keys=["image"], spatial_axis=1, prob=float(aug.hflip_prob)),
            RandRotated(
                keys=["image"],
                range_x=float(np.deg2rad(float(aug.rotate_degrees))),
                prob=float(aug.rotate_prob),
                keep_size=True,
                mode="bilinear",
                padding_mode="zeros",
            ),
            RandZoomd(
                keys=["image"],
                min_zoom=float(aug.zoom_range[0]),
                max_zoom=float(aug.zoom_range[1]),
                prob=float(aug.zoom_prob),
                mode="bilinear",
                keep_size=True,
            ),
            # Stands in for exposure and processing differences between the
            # machines and hospitals BTXRD was collected from.
            RandAdjustContrastd(
                keys=["image"],
                prob=float(aug.contrast_prob),
                gamma=tuple(float(g) for g in aug.gamma_range),
            ),
            RandGaussianNoised(
                keys=["image"], prob=float(aug.noise_prob), mean=0.0,
                std=float(aug.noise_std),
            ),
        ]

        # -- aggressive block, all off by default (prob 0) ------------------
        # Enabled by configs/overnight.yaml. Each targets a specific way the
        # model can cheat rather than learn anatomy.

        # One affine instead of separate rotate/zoom. RandAffine subsumes
        # rotation, scale, shear and translation, so enabling it *alongside*
        # RandRotated and RandZoomd would compose two independent warps and
        # interpolate twice -- blurring fine trabecular texture, which is
        # exactly the signal a bone-lesion model needs. Set rotate_prob and
        # zoom_prob to 0 when this is on; configs/overnight.yaml does.
        if float(aug.get("affine_prob", 0.0)) > 0:
            stages.append(
                RandAffined(
                    keys=["image"],
                    prob=float(aug.affine_prob),
                    rotate_range=(float(np.deg2rad(float(aug.get("affine_degrees", 20.0)))),),
                    shear_range=(float(aug.get("affine_shear", 0.05)),),
                    translate_range=(float(aug.get("affine_translate_px", 16.0)),) * 2,
                    scale_range=(tuple(float(v) for v in aug.get("affine_scale", [-0.15, 0.15])),),
                    mode="bilinear",
                    padding_mode="zeros",
                    cache_grid=False,
                )
            )

        # Occlusion. Stands in for overlapping soft tissue, hardware, lead
        # aprons and burned-in markers -- and, more to the point, stops the
        # network resting on any single region of a normal film. A model that
        # has to classify with a patch missing cannot key on one landmark.
        if float(aug.get("dropout_prob", 0.0)) > 0:
            hole = int(aug.get("dropout_hole_px", 32))
            stages.append(
                RandCoarseDropoutd(
                    keys=["image"],
                    holes=int(aug.get("dropout_holes", 1)),
                    max_holes=int(aug.get("dropout_max_holes", 5)),
                    spatial_size=(hole // 2, hole // 2),
                    max_spatial_size=(hole, hole),
                    dropout_holes=True,
                    # 0 reads as an unexposed patch, which is a thing that
                    # genuinely appears on film; random fill would invent a
                    # texture the model would then learn to recognise.
                    fill_value=0.0,
                    prob=float(aug.dropout_prob),
                )
            )

        # Non-linear intensity remap: harsher than gamma, because it can move
        # parts of the histogram in opposite directions. BTXRD is multi-centre
        # and the false positives here arrive on outside films, so invariance
        # to a processing pipeline the model has never seen is the whole point.
        if float(aug.get("histogram_prob", 0.0)) > 0:
            stages.append(
                RandHistogramShiftd(
                    keys=["image"],
                    num_control_points=tuple(
                        int(v) for v in aug.get("histogram_control_points", [5, 10])
                    ),
                    prob=float(aug.histogram_prob),
                )
            )

    # -- deterministic tail ------------------------------------------------
    stages += [
        RepeatChanneld(keys=["image"], repeats=int(data.in_channels)),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=np.asarray(data.norm_mean, dtype=np.float32),
            divisor=np.asarray(data.norm_std, dtype=np.float32),
            channel_wise=True,
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        EnsureTyped(keys=["label"], dtype=torch.long, track_meta=False),
    ]
    if not keep_meta:
        stages.append(DeleteItemsd(keys=["image_meta_dict"]))

    return Compose(stages)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def _unwrap_read_error(exc: BaseException) -> RadiographReadError | None:
    """Find a :class:`RadiographReadError` anywhere in an exception's cause chain.

    MONAI's ``Compose`` catches whatever a transform raises and re-raises it as
    a bare ``RuntimeError("applying transform ...")``, keeping the original only
    as ``__cause__``. Catching ``RadiographReadError`` directly therefore never
    matches, so the chain has to be walked explicitly.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, RadiographReadError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


class RobustDataset(Dataset):
    """Dataset that survives an undecodable file instead of killing the run.

    On a read failure it logs the offending path once and substitutes a
    neighbouring index. Resilience lives here rather than inside
    ``LoadRadiographd`` so that the transform stays a pure function and the
    substitution policy is visible in one obvious place.

    Substitution is a *training-time* convenience. Evaluation should use
    ``scripts/verify_data.py --deep`` to prove every test file decodes, because
    silently swapping a sample would corrupt a reported metric.
    """

    def __init__(self, data: list[dict[str, Any]], transform: Any, max_retries: int = 8) -> None:
        super().__init__(data=data, transform=transform)
        self.max_retries = max_retries
        self._failed: set[str] = set()

    def __getitem__(self, index: int) -> Any:
        n = len(self.data)
        for attempt in range(self.max_retries):
            probe = (index + attempt) % n
            try:
                return super().__getitem__(probe)
            except Exception as exc:  # noqa: BLE001 - narrowed by _unwrap_read_error
                read_error = _unwrap_read_error(exc)
                if read_error is None:
                    # Not a decode failure -- a genuine bug in the transform
                    # chain. Substituting samples would hide it.
                    raise
                path = str(self.data[probe].get("image", "<unknown>"))
                if path not in self._failed:
                    self._failed.add(path)
                    logger.error("unreadable sample skipped: %s (%s)", path, read_error)
        raise RuntimeError(
            f"{self.max_retries} consecutive samples failed to load starting at index {index}; "
            "the dataset is likely corrupt -- run `python scripts/verify_data.py --deep`"
        )

    @property
    def failed_paths(self) -> set[str]:
        return set(self._failed)


def build_dataset(
    cfg: Config,
    mode: str,
    records: list[dict[str, Any]] | None = None,
    keep_meta: bool = False,
):
    """Build the dataset for one mode, caching when configured to."""
    records = records if records is not None else build_records(cfg, split=mode)
    transform = build_transforms(cfg, mode, keep_meta=keep_meta)
    cache_rate = float(cfg.loader.cache_rate)

    if cache_rate > 0:
        # 3,746 images at 256x256 float32 is ~2.9 GB fully cached, which fits
        # comfortably in 32 GB. copy_cache=False avoids duplicating each cached
        # tensor per access; safe because every downstream random transform
        # allocates a new array rather than writing in place.
        return CacheDataset(
            data=records,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=0,
            copy_cache=False,
            progress=True,
        )
    return RobustDataset(data=records, transform=transform)


def build_sampler(cfg: Config, records: list[dict[str, Any]]):
    """Build the training sampler, refusing to double-correct the imbalance.

    Returns ``None`` unless ``loader.balanced_sampler`` is set, in which case a
    ``WeightedRandomSampler`` draws all three classes with equal probability.

    The guard is the important part. A weighted loss and a balanced sampler each
    fix the imbalance on their own; together they apply the correction twice and
    the model learns to over-predict malignant, which shows up as normal
    controls being flagged as lesions. That failure is invisible in the training
    curve, so it is caught here at construction time rather than left to be
    diagnosed from a confusion matrix days later.
    """
    if not bool(cfg.loader.get("balanced_sampler", False)):
        return None

    from torch.utils.data import WeightedRandomSampler

    loss_is_weighted = (
        cfg.loss.get("alpha", None) is not None
        or (bool(cfg.loss.get("auto_alpha", True)) and str(cfg.loss.name).lower() != "ce")
    )
    if loss_is_weighted:
        raise ConfigError(
            "\n".join(
                [
                    "loader.balanced_sampler and a class-weighted loss are both enabled.",
                    "Each corrects the class imbalance on its own; together they correct",
                    "it twice and the model over-predicts malignant, which surfaces as",
                    "normal films being called lesions. Pick one:",
                    "  * sampler   -> set loss.auto_alpha: false (and leave loss.alpha unset)",
                    "  * weighting -> set loader.balanced_sampler: false",
                ]
            )
        )

    weights = sample_weights(records, num_classes=int(cfg.model.num_classes))
    configured = cfg.loader.get("samples_per_epoch", None)
    num_samples = int(configured) if configured else len(records)

    counts = np.bincount([r["label"] for r in records], minlength=int(cfg.model.num_classes))
    logger.info(
        "balanced sampler: %d samples/epoch drawn with replacement from %s "
        "(expect ~%d per class per epoch instead of %s)",
        num_samples,
        dict(zip(CLASS_NAMES, counts.tolist(), strict=False)),
        num_samples // max(int(cfg.model.num_classes), 1),
        counts.tolist(),
    )
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def build_dataloader(
    cfg: Config,
    mode: str,
    records: list[dict[str, Any]] | None = None,
    shuffle: bool | None = None,
    keep_meta: bool = False,
) -> DataLoader:
    """Build the dataloader for one mode.

    On Windows, worker processes are *spawned*, not forked: each one re-imports
    the module and receives its own copy of the cache, so ``num_workers > 0``
    alongside ``cache_rate=1.0`` multiplies memory instead of throughput. The
    shipped default is ``num_workers=0``, where the cache makes workers
    redundant anyway. Any script that constructs a dataloader still needs an
    ``if __name__ == "__main__":`` guard.
    """
    records = records if records is not None else build_records(cfg, split=mode)
    dataset = build_dataset(cfg, mode, records=records, keep_meta=keep_meta)
    loader_cfg = cfg.loader
    num_workers = int(loader_cfg.num_workers)
    is_train = mode == "train"

    # A sampler and shuffle=True are mutually exclusive in PyTorch: the sampler
    # already defines the draw order.
    sampler = build_sampler(cfg, records) if is_train else None
    use_shuffle = (is_train if shuffle is None else shuffle) and sampler is None

    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.batch_size),
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.pin_memory) and torch.cuda.is_available(),
        persistent_workers=bool(loader_cfg.persistent_workers) and num_workers > 0,
        drop_last=bool(loader_cfg.drop_last_train) and is_train,
        collate_fn=list_data_collate,
    )
