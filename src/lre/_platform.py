"""Platform fixes needed before Devanagari can be rendered.

Pillow's wheels ship libraqm (complex-script shaping) but load FriBiDi at run time
from ``libfribidi.dylib`` or ``/usr/local/lib`` only. On Apple-Silicon Macs Homebrew
installs it under ``/opt/homebrew/lib``, which dyld does not search, so Devanagari
silently renders with broken matras and conjuncts. dyld reads its search path only
at process start, so the fix is to re-execute the process once with the path set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BREW_LIBS = [Path("/opt/homebrew/lib"), Path("/opt/homebrew/opt/fribidi/lib")]
_FLAG = "LRE_FRIBIDI_REEXEC"


def fribidi_reexec_env() -> dict | None:
    """The environment to re-exec with, or None when no re-exec is needed or possible."""
    if sys.platform != "darwin" or os.environ.get(_FLAG):
        return None
    from PIL import features
    if features.check("raqm"):
        return None
    lib_dir = next((d for d in _BREW_LIBS if (d / "libfribidi.dylib").exists()), None)
    if lib_dir is None:
        return None  # nothing to add; callers report the missing shaping engine themselves
    env = dict(os.environ)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
        p for p in [str(lib_dir), env.get("DYLD_FALLBACK_LIBRARY_PATH", ""), "/usr/local/lib", "/usr/lib"] if p)
    env[_FLAG] = "1"
    return env


def ensure_fribidi(argv: list[str] | None = None) -> None:
    env = fribidi_reexec_env()
    if env is not None:
        os.execve(sys.executable, argv if argv is not None else [sys.executable, *sys.argv], env)
