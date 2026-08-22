"""Private, de-identified local storage for user-uploaded radiographs."""

from __future__ import annotations

import io
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

DEFAULT_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "data" / "user_uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")
DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
STANDARD_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# DICOM Basic Application Confidentiality Profile identifiers commonly carrying PII.
IDENTIFYING_DICOM_KEYWORDS = {
    "AccessionNumber",
    "AdditionalPatientHistory",
    "AdmissionID",
    "CurrentPatientLocation",
    "InstitutionAddress",
    "InstitutionName",
    "InstitutionalDepartmentName",
    "IssuerOfPatientID",
    "MedicalRecordLocator",
    "OperatorsName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientAddress",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientID",
    "PatientMotherBirthName",
    "PatientName",
    "PatientTelephoneNumbers",
    "PatientSex",
    "PerformingPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianName",
    "ReferringPhysicianTelephoneNumbers",
    "RequestingPhysician",
    "StationName",
    "StudyID",
}


class StorageError(RuntimeError):
    """An upload could not be validated or stored safely."""


@dataclass(frozen=True)
class StoredUpload:
    upload_id: str
    original_filename: str
    path: Path
    is_dicom: bool


def _secure_permissions(path: Path, mode: int) -> None:
    with suppress(OSError):
        path.chmod(mode)


def _safe_original_filename(filename: str) -> str:
    basename = Path(filename.replace("\\", "/")).name.strip()
    sanitized = SAFE_FILENAME.sub("_", basename)[:255]
    return sanitized or "radiograph"


def _validated_user_directory(user_id: str, root: str | Path | None = None) -> Path:
    try:
        canonical_id = str(uuid.UUID(user_id))
    except (ValueError, TypeError) as exc:
        raise StorageError("Invalid account identifier.") from exc

    upload_root = Path(root or DEFAULT_UPLOAD_ROOT).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    user_directory = upload_root / canonical_id
    user_directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    _secure_permissions(upload_root, 0o700)
    _secure_permissions(user_directory, 0o700)
    return user_directory


def _anonymize_dicom(payload: bytes, destination: Path) -> None:
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=True)
        if "PixelData" not in dataset:
            raise StorageError("The DICOM file contains no pixel data.")

        dataset.remove_private_tags()

        def remove_identifying_value(container, element) -> None:
            if (
                element.VR == "PN"
                or element.tag.group == 0x0010
                or element.VR in {"DA", "DT", "TM"}
            ):
                del container[element.tag]

        dataset.walk(remove_identifying_value)
        for keyword in IDENTIFYING_DICOM_KEYWORDS:
            if keyword in dataset:
                del dataset[keyword]

        dataset.PatientIdentityRemoved = "YES"
        dataset.DeidentificationMethod = "ONNM local header de-identification"
        dataset.StudyInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = generate_uid()
        dataset.SOPInstanceUID = generate_uid()

        if not hasattr(dataset, "file_meta"):
            dataset.file_meta = FileMetaDataset()
        if "TransferSyntaxUID" not in dataset.file_meta:
            dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        if "SOPClassUID" in dataset:
            dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID

        try:
            dataset.save_as(str(destination), enforce_file_format=True)
        except TypeError:
            dataset.save_as(str(destination), write_like_original=False)
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError("The DICOM file could not be anonymized.") from exc


def _sanitize_standard_image(payload: bytes, destination: Path) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.width * image.height > 100_000_000:
                raise StorageError("The image dimensions are too large.")
            # Re-encoding pixel data as PNG removes EXIF, comments, and other metadata.
            image.convert("L").save(destination, format="PNG", optimize=True)
    except StorageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise StorageError("The uploaded image could not be decoded.") from exc


def save_upload(
    payload: bytes,
    *,
    user_id: str,
    original_filename: str,
    root: str | Path | None = None,
) -> StoredUpload:
    if not payload:
        raise StorageError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise StorageError(f"Uploads must be no larger than {limit_mb} MB.")

    safe_name = _safe_original_filename(original_filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in DICOM_SUFFIXES | STANDARD_SUFFIXES:
        raise StorageError("Unsupported radiograph file type.")

    is_dicom = suffix in DICOM_SUFFIXES
    upload_id = str(uuid.uuid4())
    destination = _validated_user_directory(user_id, root) / (
        f"{upload_id}.dcm" if is_dicom else f"{upload_id}.png"
    )

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        if is_dicom:
            _anonymize_dicom(payload, temporary)
        else:
            _sanitize_standard_image(payload, temporary)
        os.replace(temporary, destination)
        _secure_permissions(destination, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    return StoredUpload(upload_id, safe_name, destination, is_dicom)


def delete_upload(path: str | Path) -> None:
    Path(path).unlink(missing_ok=True)


def is_user_file(user_id: str, path: str | Path, root: str | Path | None = None) -> bool:
    try:
        user_directory = _validated_user_directory(user_id, root).resolve()
        Path(path).resolve().relative_to(user_directory)
        return Path(path).is_file()
    except (OSError, ValueError, StorageError):
        return False
