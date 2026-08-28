"""HTTP shell around :mod:`service`, for the Cloudflare Container.

WHAT THIS IS AND IS NOT
-----------------------
A deliberately boring FastAPI app. Every interesting decision lives in
``service.ScanService``; this file only turns HTTP into a call and a call into
JSON. Keeping it thin is what lets ``scripts/check_inference_parity.py`` compare
the container against a local run without the transport being a variable.

REACHABILITY
------------
This process is **not** on the public internet. Cloudflare Containers are
addressed only through their Durable Object binding, so the sole caller is the
Pages Function. The bearer check below is therefore defence in depth rather than
the primary control: it exists so that a future misconfiguration which did expose
the port would fail closed instead of serving free inference to anyone who found
it.

WHY THE MODEL LOADS AT IMPORT
-----------------------------
The container is billed by wall-clock runtime and sleeps after 90 seconds of
idleness, so the expensive work is deliberately front-loaded into startup. By the
time the first request arrives the checkpoint is resident and warmed. The Pages
Function calls ``/health`` the moment a visitor picks a file, precisely so this
cost is paid while they are still looking at the file picker.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

# The image copies the repository's ``src/`` tree in whole, so both the ``onnm``
# package and the top-level ``community`` module resolve. Setting this here as
# well as in the Dockerfile means ``python inference/main.py`` also works from a
# checkout, which is how the parity gate is run locally.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from onnm.io_radiograph import RadiographReadError, UnsupportedFormatError  # noqa: E402
from service import ScanService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("onnm.inference")

#: Mirrors ``src/storage.py:MAX_UPLOAD_BYTES``. Enforced again here because this
#: process must be safe even if the caller's own limit is ever relaxed, and
#: because a 4 GiB container should never be asked to buffer an arbitrary body.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

CHECKPOINT = os.environ.get("ONNM_CHECKPOINT", "/opt/onnm/best.pt")
INFERENCE_KEY = os.environ.get("INFERENCE_KEY", "")

app = FastAPI(title="ONNM inference", docs_url=None, redoc_url=None, openapi_url=None)

# Loaded once, at import, before the server accepts connections.
logger.info("loading checkpoint %s", CHECKPOINT)
SERVICE = ScanService(CHECKPOINT, warmup=True)
logger.info("checkpoint ready: %s", SERVICE.describe().get("run", "?"))


def require_key(request: Request) -> None:
    """Constant-time bearer check.

    Compared with ``hmac.compare_digest`` for the same reason
    ``cloudflare/src/worker.js`` hand-rolls ``timingSafeEqual``: a naive ``==``
    leaks the key one byte at a time to anyone who can measure the response.

    An unset ``INFERENCE_KEY`` refuses every request rather than allowing them.
    A deployment that forgot the secret should be visibly broken, not quietly
    open.
    """
    if not INFERENCE_KEY:
        raise HTTPException(status_code=503, detail="INFERENCE_KEY is not configured")
    header = request.headers.get("authorization", "")
    presented = header[7:] if header.startswith("Bearer ") else ""
    if not hmac.compare_digest(presented, INFERENCE_KEY):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the facts the frontend needs to configure itself.

    Unauthenticated on purpose. It returns nothing about any user and no model
    weights, and the Pages Function uses it as the warm-up ping; requiring a key
    would mean the warm-up path could fail for a reason unrelated to the model
    being ready.
    """
    return {"ok": True, "model": SERVICE.describe()}


@app.post("/infer", dependencies=[Depends(require_key)])
async def infer(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    cam_class: str = Form(default="auto"),
    with_heatmap: bool = Form(default=True),
    want_preprocessed: bool = Form(default=True),
) -> JSONResponse:
    """Run one scan.

    Multipart rather than a raw body because the caller is a Worker building a
    ``FormData``, and because the filename travels as part of the format. The
    filename is not cosmetic here: ``predict`` dispatches its decoder on the
    suffix, and a DICOM that arrived as ``blob`` would be read as an image.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload is {len(payload)} bytes; limit is {MAX_UPLOAD_BYTES}",
        )

    filename = file.filename or "upload"
    try:
        result = SERVICE.run_scan(
            payload,
            filename,
            threshold=threshold,
            cam_class=cam_class,
            with_heatmap=with_heatmap,
            want_preprocessed=want_preprocessed,
        )
    except UnsupportedFormatError as exc:
        # 415: the caller sent something this service will never accept.
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except RadiographReadError as exc:
        # 422: a supported container that could not be decoded. Distinct from
        # 415 because the two mean different things to a user -- "wrong kind of
        # file" versus "this file is damaged".
        raise HTTPException(status_code=422, detail=f"could not decode: {exc}") from exc
    except ValueError as exc:
        # Raised by resolve_cam_index for an unknown cam_class. A caller bug,
        # not a user one, but still a 400 rather than a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104 - container-internal; not publicly routable
        port=int(os.environ.get("PORT", "8080")),
        log_level="info",
    )
