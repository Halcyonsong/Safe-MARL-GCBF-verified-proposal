"""Official GCBF+ baseline validation entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pathlib as _pathlib
import sys as _sys
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from core.paths import project_root, ensure_import_path
_ROOT = project_root()
ensure_import_path()



def main() -> None:
    ensure_import_path()
    runpy.run_path(str(Path(__file__).with_name("test_official.py")), run_name="__main__")


if __name__ == "__main__":
    main()
