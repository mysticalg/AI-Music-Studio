#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_SCRIPT = REPO_ROOT / "plugins" / "AdvancedVSTi" / "scripts" / "fetch_open_instrument_samples.py"


def main() -> int:
    if not SUBMODULE_SCRIPT.exists():
        print(
            "Native VST submodule script not found. Run `git submodule update --init --recursive` first.",
            file=sys.stderr,
        )
        return 1

    completed = subprocess.run([sys.executable, str(SUBMODULE_SCRIPT), *sys.argv[1:]], cwd=REPO_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
