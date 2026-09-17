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


def source_files(source: Path) -> list[str]:
    return sorted(
        str(path) for path in source.rglob("*") if path.is_file() and not path.name.startswith(".")
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
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="Directory of counseldesk documents to ingest (default: the Adams v0.0.2 path).",
    )
    parser.add_argument(
        "--profile",
        choices=("legal", "plain"),
        default="legal",
        help=(
            "legal (default): the legal extraction profile. plain: default cognee extraction, "
            "for the baseline dataset the recall eval compares against."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in tokens. Omitted (or 0): cognee's default, for either profile.",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Legal profile only: one LLM call per chunk (legal prompt) instead of plain + legal.",
    )
    parser.add_argument(
        "--keep-low-salience",
        action="store_true",
        help="Legal profile only: keep low-salience (boilerplate) assertions, down-weighted.",
    )
    args = parser.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY (or OPENAI_API_KEY) is not set", file=sys.stderr)
        return 1

    import cognee
    from cognee.domains.legal import legal_profile

    files = source_files(args.source)
    if args.match:
        files = [f for f in files if any(m.lower() in f.lower() for m in args.match)]
    if args.limit:
        files = files[: args.limit]
    if args.profile == "legal":
        profile = legal_profile(
            chunk_size=args.chunk_size or None,
            two_pass=not args.single_pass,
            drop_low_salience=not args.keep_low_salience,
        )
    else:
        profile = {}
        if args.chunk_size:
            profile["chunk_size"] = args.chunk_size
    print(
        f"files={len(files)} dataset={args.dataset} profile={args.profile} "
        f"chunk_size={profile.get('chunk_size', 'default')} "
        f"two_pass={args.profile == 'legal' and not args.single_pass} "
        f"drop_low_salience={args.profile == 'legal' and not args.keep_low_salience}"
    )

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
