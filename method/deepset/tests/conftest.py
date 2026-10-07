"""Pytest bootstrap: wire stage2/stage3 import roots before collecting tests."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.paths import ensure_import_path  # noqa: E402

ensure_import_path()
