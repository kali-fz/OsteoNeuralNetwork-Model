from __future__ import annotations

import uuid

import pydicom
from PIL import Image

from storage import is_user_file, save_upload


def test_standard_image_is_uuid_named_and_metadata_free(tmp_path, jpeg_image) -> None:
    user_id = str(uuid.uuid4())
    stored = save_upload(
        jpeg_image.read_bytes(),
        user_id=user_id,
        original_filename="../../patient-name.jpeg",
        root=tmp_path,
    )

    assert stored.original_filename == "patient-name.jpeg"
    assert stored.path.parent == tmp_path / user_id
    assert stored.path.suffix == ".png"
    uuid.UUID(stored.path.stem)
    with Image.open(stored.path) as image:
        assert image.getexif() == {}
    assert is_user_file(user_id, stored.path, tmp_path)
    assert not is_user_file(str(uuid.uuid4()), stored.path, tmp_path)


def test_dicom_headers_are_anonymized(tmp_path, mono2_dicom) -> None:
    dataset = pydicom.dcmread(mono2_dicom)
    dataset.PatientName = "Jane^Patient"
    dataset.PatientID = "MRN-123"
    dataset.InstitutionName = "Example Hospital"
    dataset.add_new((0x0011, 0x1010), "LO", "private secret")
    dataset.save_as(mono2_dicom, enforce_file_format=True)

    stored = save_upload(
        mono2_dicom.read_bytes(),
        user_id=str(uuid.uuid4()),
        original_filename="patient-name.dcm",
        root=tmp_path,
    )
    anonymized = pydicom.dcmread(stored.path)

    assert "PatientName" not in anonymized
    assert "PatientID" not in anonymized
    assert "InstitutionName" not in anonymized
    assert anonymized.PatientIdentityRemoved == "YES"
    assert not any(element.tag.is_private for element in anonymized.iterall())
    assert anonymized.pixel_array.shape == dataset.pixel_array.shape



def test_is_user_file_is_read_only(tmp_path) -> None:
    # A pure authorization check must not create directories as a side effect:
    # previewing scan history for a user who owns nothing should leave the
    # storage root untouched.
    user_id = str(uuid.uuid4())
    probe = tmp_path / "somewhere" / "file.png"

    assert not is_user_file(user_id, probe, tmp_path / "uploads")

    assert not (tmp_path / "uploads").exists()
    assert not (tmp_path / "uploads" / user_id).exists()
