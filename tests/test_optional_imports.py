"""The torch/schema core remains importable without optional GP packages."""

from __future__ import annotations

import subprocess
import sys


def test_core_non_gp_exports_work_when_gpy_is_blocked() -> None:
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "GPy" or name.startswith("emukit"):
        raise ModuleNotFoundError("blocked for test", name=name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from core import MinMaxScaler, build_cnp
assert MinMaxScaler is not None and build_cnp is not None
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gp_export_has_actionable_error_when_gpy_is_blocked() -> None:
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "GPy" or name.startswith("emukit"):
        raise ModuleNotFoundError("blocked for test", name=name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
try:
    from core import MultiFidelityGP
except ImportError as exc:
    assert ".[gp]" in str(exc)
else:
    raise AssertionError("GP export unexpectedly imported")
'''
    subprocess.run([sys.executable, "-c", code], check=True)
