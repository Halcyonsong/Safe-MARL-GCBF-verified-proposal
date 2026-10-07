"""Common path bootstrap for stage / baseline entrypoints.

Repo layout (2026-10-07):
  method/deepset/{stage1,stage2,stage3,config,scripts,tests,output}
  method/meanfield/{stage_mf,config,scripts,output}
  baselines/<algo>/{code,config,log,output,scripts}
  acceptance/
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _ensure_baseline_namespace(root: pathlib.Path) -> None:
    """Map ``import baseline.<algo>`` → ``baselines/<algo>/code`` (filesystem hard-cut)."""
    if "baseline" in sys.modules and getattr(sys.modules["baseline"], "__restructured__", False):
        return

    pkg = types.ModuleType("baseline")
    pkg.__path__ = []  # namespace package
    pkg.__restructured__ = True  # type: ignore[attr-defined]
    sys.modules["baseline"] = pkg

    for algo in ("gcbfplus", "gcbf", "handcbf", "mappo_lag", "informarl"):
        code_dir = root / "baselines" / algo / "code"
        if not code_dir.is_dir():
            continue
        name = f"baseline.{algo}"
        # Load as a package whose __path__ is the code directory.
        spec = importlib.util.spec_from_file_location(
            name,
            code_dir / "__init__.py",
            submodule_search_locations=[str(code_dir)],
        )
        if spec is None or spec.loader is None:
            # empty / missing __init__: still register path for submodule imports
            sub = types.ModuleType(name)
            sub.__path__ = [str(code_dir)]  # type: ignore[attr-defined]
            sys.modules[name] = sub
            setattr(pkg, algo, sub)
            continue
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [str(code_dir)]  # type: ignore[attr-defined]
        sys.modules[name] = mod
        setattr(pkg, algo, mod)
        spec.loader.exec_module(mod)


def ensure_import_path() -> pathlib.Path:
    """Make ``gcbfplus``, ``stage*``, ``stage_mf``, and ``baseline.*`` importable."""
    root = project_root()
    core = root / "core"
    deepset = root / "method" / "deepset"
    meanfield = root / "method" / "meanfield"
    for path in (root, core, deepset, meanfield):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    _ensure_baseline_namespace(root)
    return root
