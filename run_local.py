"""Launch the ONNM app locally, bound to loopback only.

Why this exists
---------------
Streamlit binds 0.0.0.0 by default, which on a LAN -- or behind a forwarding
router -- makes an uploaded radiograph reachable by other machines. This app
handles medical images, so a local run should be loopback-only.

That used to be pinned in `.streamlit/config.toml`, but that file is committed
and Streamlit Community Cloud reads it too: forcing an address or a port there
risks the hosted deploy never coming up, with no useful error. So the local
binding lives here instead, where it cannot affect the hosted app.

    python run_local.py            # http://localhost:8501
    python run_local.py --port 8600
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="localhost",
                        help="override at your own risk; the default keeps uploads off the LAN")
    args = parser.parse_args()

    app = Path(__file__).resolve().parent / "app.py"
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", args.address,
        "--server.port", str(args.port),
    ]
    if args.address not in ("localhost", "127.0.0.1"):
        print(
            f"WARNING: binding {args.address} exposes uploaded radiographs beyond this "
            "machine. Only do this on a network you control.",
            file=sys.stderr,
        )
    print(f"Starting ONNM on http://{args.address}:{args.port}\n")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
