"""Official GCBF baseline training entrypoint (algo=gcbf).

Reuses baselines/gcbfplus/code/train_official.py — same CLI, default algo overridden
by scripts/run_baseline_gcbf.sh to ``gcbf``.
"""

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
    # Same official trainer; --algo gcbf passed by launcher.
    runpy.run_path(
        str(_ROOT / "baseline" / "gcbfplus" / "train_official.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
