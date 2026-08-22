"""Make ``src/onnm`` importable when the package has not been pip-installed yet.

Importing this module for its side effect keeps every script runnable straight
from a clone -- useful precisely during Milestone 0, when the install itself is
what is being debugged.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
