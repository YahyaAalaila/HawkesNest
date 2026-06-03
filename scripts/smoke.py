#!/usr/bin/env python3
"""
smoke.py — minimal install verification for HawkesNest.

Runs one EntanglementSuite and one HeterogeneitySuite generation (50 events
each) and checks that outputs are non-empty and files are written.

Usage:
    python scripts/smoke.py           # writes to /tmp/hn_smoke/
    python scripts/smoke.py --out /path/to/dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/hn_smoke"))
    args = parser.parse_args()

    checks_passed = 0

    for SuiteCls, level in [(EntanglementSuite, "L2"), (HeterogeneitySuite, "H1")]:
        suite = SuiteCls()
        out_dir = args.out / suite.suite_name / level
        result = suite.generate(level=level, n_events=50, seed=0, out_dir=out_dir)

        assert result.n_events == 50, f"{suite.suite_name}/{level}: expected 50 events, got {result.n_events}"
        assert (out_dir / "events.csv").exists(), "events.csv missing"
        assert (out_dir / "events.jsonl").exists(), "events.jsonl missing"
        assert (out_dir / "metadata.json").exists(), "metadata.json missing"

        cols = set(result.events.columns)
        assert {"t", "x", "y", "m", "is_triggered"}.issubset(cols), f"missing columns: {cols}"

        print(f"  OK  {suite.suite_name}/{level}  n={result.n_events}  out={out_dir}")
        checks_passed += 1

    print(f"\nSmoke: {checks_passed}/2 checks passed.")


if __name__ == "__main__":
    main()
