import os
import sys
from pathlib import Path

from lre._platform import fribidi_reexec_env

# Run tests from the repo root so config.yaml and data/fonts resolve.
os.chdir(Path(__file__).resolve().parents[1])


def pytest_configure(config):
    # Re-run pytest with Homebrew's FriBiDi visible if Pillow cannot find it (macOS only).
    env = fribidi_reexec_env()
    if env is None:
        return
    # Release output capture first, or the new process writes into pytest's capture file.
    capman = config.pluginmanager.getplugin("capturemanager")
    if capman is not None:
        capman.stop_global_capturing()
    os.execve(sys.executable, [sys.executable, "-m", "pytest", *sys.argv[1:]], env)
