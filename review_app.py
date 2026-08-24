"""ONNM review console — approve community submissions, then push them to training.

    .venv\\Scripts\\python.exe -m streamlit run review_app.py --server.port 8502

WHY THIS IS A SEPARATE APP
--------------------------
The review queue used to live in a sidebar expander inside ``app.py``. That was
the wrong home for it twice over.

It is the wrong *place*: reviewing means looking at radiographs and deciding
what they are, which needs the width of a page, not a 300-pixel sidebar column
squeezed beside a file uploader.

And it is the wrong *deployment*. ``app.py`` is the thing published to Streamlit
Community Cloud for the public, so every guard on the review queue inside it is
a guard that has to hold in a process strangers are talking to. This file is
never deployed. It runs from your checkout, against your admin key, on
localhost. The strongest protection available for the review path is that the
code implementing it is not running on the public host at all.

``app.py`` keeps its sidebar entry so the loop is discoverable, but this is the
console to actually work in.

WHAT IT NEEDS
-------------
    $env:ONNM_COMMUNITY_URL = "https://onnm-community.kali-fz.workers.dev"
    $env:ONNM_ADMIN_KEY     = "..."

Without both it refuses to render anything, and it will only open for the one
account the Worker and the D1 schema pin -- see ``community.ADMIN_USER_ID``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from community import (  # noqa: E402
    ADMIN_EMAIL,
    ADMIN_USER_ID,
    get_client,
)
from community_ui import community_status, render_admin_review  # noqa: E402

st.set_page_config(page_title="ONNM review console", page_icon="🦴", layout="wide")

st.title("ONNM review console")
st.caption("This console runs locally. The admin key stays on this machine.")


# ---------------------------------------------------------------------------
# Configuration gate
# ---------------------------------------------------------------------------
client = get_client()
if not client.enabled:
    st.error("`ONNM_COMMUNITY_URL` and `ONNM_COMMUNITY_KEY` are not set.")
    st.code(
        '$env:ONNM_COMMUNITY_URL = "https://onnm-community.kali-fz.workers.dev"\n'
        '$env:ONNM_COMMUNITY_KEY = "<the app key>"\n'
        '$env:ONNM_ADMIN_KEY     = "<the admin key>"',
        language="powershell",
    )
    st.stop()

if not client.admin_enabled:
    st.error("`ONNM_ADMIN_KEY` is not set, so the review endpoints are unreachable.")
    st.caption(
        "It is deliberately absent from the hosted app. A leak of the app's key must "
        "not be able to approve its own training data."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Identity
#
# There is no sign-in here. The console runs on your machine, behind your admin
# key, and the Worker independently refuses any /admin request that does not
# name the pinned account -- so a login form would be theatre: it would check a
# claim this process has no way to verify and no authority to grant.
#
# The identity is asserted, not proven, and the proof lives at the edge.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session")
    st.write(f"Reviewing as **{ADMIN_EMAIL}**")
    st.caption(f"`{ADMIN_USER_ID}`")
    st.caption(
        "Pinned in the Worker and in a CHECK constraint on `users`. Changing who "
        "can review takes a code change and a migration, which is the intended "
        "amount of friction for the only path into the training set."
    )
    st.divider()

    st.subheader("Push to training")
    st.caption(
        "Claims the approved rows, writes their images, and rebuilds "
        "`configs/controls_manifest.csv`. The base configuration already reads that file."
    )
    store = st.text_input(
        "Store directory",
        value="data/community",
        help="Where the claimed batches are written. Claiming is irreversible from "
             "here, so in Colab this must point at Drive rather than at a runtime "
             "that is about to be wiped.",
    )
    dry_run = st.checkbox("Dry run (claim nothing)", value=True)
    if st.button("Sync approved rows to training", type="primary", width="stretch"):
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "sync_community.py"),
            "--store", store,
        ]
        if dry_run:
            command.append("--dry-run")
        with st.spinner("Connecting to Cloudflare..."):
            finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command, cwd=REPO_ROOT, capture_output=True, text=True,
                env={**os.environ}, check=False,
            )
        st.code(finished.stdout or "(no output)", language="text")
        if finished.returncode != 0:
            st.error("The sync failed.")
            if finished.stderr:
                st.code(finished.stderr, language="text")
        else:
            success_message = (
                "Sync complete." if not dry_run else "Dry run complete. Nothing was claimed."
            )
            st.success(success_message)
            community_status.clear()

    st.divider()
    if st.button("Refresh counts", width="stretch"):
        community_status.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------
render_admin_review(ADMIN_USER_ID, ADMIN_EMAIL)
