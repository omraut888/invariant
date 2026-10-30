"""Write the Stage 5 memo of a completed run (docs/adr/0017).

Usage: uv run python scripts/write_memo.py RUN_DIR
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from invariant.core.logging import LogMode, configure_logging
from invariant.stages.memo.stage import write_memo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    configure_logging(LogMode.DEV, level="WARNING")
    out = write_memo(args.run_dir)
    print(f"{out.sheet.recommendation}: {out.md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
