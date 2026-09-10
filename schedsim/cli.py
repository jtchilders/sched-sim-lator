"""Command line: python -m schedsim <command> ..."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="schedsim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract the PBS sqlite DB into parquet tables")
    e.add_argument("--db", required=True)
    e.add_argument("--out", default="data/trace")

    r = sub.add_parser("replay", help="replay the real trace through the simulated scheduler")
    r.add_argument("--config", required=True)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, dotted path (e.g. scheduler.cycle_h=0.25)")

    args = ap.parse_args(argv)
    if args.cmd == "extract":
        from .trace.extract import extract
        extract(args.db, args.out)
    elif args.cmd == "replay":
        import yaml
        from .config import ReplayConfig
        from .replay import run_replay
        with open(args.config) as f:
            d = yaml.safe_load(f) or {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            cur = d
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = yaml.safe_load(v)
        run_replay(ReplayConfig.from_dict(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
