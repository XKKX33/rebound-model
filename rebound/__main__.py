"""`python -m rebound` 的入口，只做转发。"""

from __future__ import annotations

import sys

from rebound.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
