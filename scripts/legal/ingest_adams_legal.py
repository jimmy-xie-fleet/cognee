"""Re-index the Adams v0.0.2 counseldesk documents with the legal extraction profile.

Runs in-process (the legal profile is SDK-only), pinned to the same ~/.cognee data
directories the fork server and UI use, so the new dataset shows up beside the
baseline one. Stop the server before running: embedded databases take file locks.

Usage:
    python scripts/legal/ingest_adams_legal.py --dry-run
    python scripts/legal/ingest_adams_legal.py
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

SOURCE = Path("/Users/jimmyxie/Downloads/counseldesk-worlds-1of2/adams-family-redevelopment/v0.0.2")
DATASET = "adams_family_legal"


def source_files() -> list[str]:
    return sorted(
        str(path) for path in SOURCE.rglob("*") if path.is_file() and not path.name.startswith(".")
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--limit", type=int, default=0, help="ingest only the first N files")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="ingest only files whose name contains this substring (repeatable)",
    )
    args = parser.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY (or OPENAI_API_KEY) is not set", file=sys.stderr)
        return 1

    import cognee
    from cognee.domains.legal import legal_profile

    files = source_files()
    if args.match:
        files = [f for f in files if any(m.lower() in f.lower() for m in args.match)]
    if args.limit:
        files = files[: args.limit]
    profile = legal_profile()
    print(f"files={len(files)} dataset={args.dataset} chunk_size={profile['chunk_size']}")

    result = await cognee.remember(
        files,
        dataset_name=args.dataset,
        self_improvement=False,
        dry_run=args.dry_run,
        **profile,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
