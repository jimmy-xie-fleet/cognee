"""Compare what the default extraction and the legal profile produced.

Reads each dataset's own graph database (access control gives every dataset its
own store) and reports node/edge shape plus legal-profile specifics.

Usage:
    python scripts/legal/inspect_datasets.py adams_family_redevelopment adams_family_legal
"""

import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

# "No." is the abbreviation for "number" ("Resolution No. 2026-118"), not a negation, and
# "asserted" is usually adjectival here ("asserted authority"), so both are excluded to keep
# the count honest.
NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|none|cannot|failed to)\b|\bno\b(?!\.)(?! \d)",
    re.IGNORECASE,
)
SPEECH_ACT = re.compile(
    r"\b(denies|denied|alleges|admits|admitted|testifies|testified|contends)\b",
    re.IGNORECASE,
)


def pct(part: int, whole: int) -> str:
    return f"{(100 * part / whole):.0f}%" if whole else "n/a"


async def report(dataset_name: str) -> None:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from sqlalchemy import select

    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        dataset = (
            await session.execute(select(Dataset).where(Dataset.name == dataset_name))
        ).scalar_one_or_none()

    print(f"\n{'=' * 78}\n{dataset_name}")
    if dataset is None:
        print("  (no such dataset)")
        return
    print(f"  id={dataset.id}")

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()

    node_types = Counter(str((props or {}).get("type", "?")) for _id, props in nodes)
    edge_names = Counter(str(edge[2]) for edge in edges)
    print(f"  nodes={len(nodes)}  edges={len(edges)}")
    print(f"  node types: {dict(node_types.most_common(12))}")
    print(f"  top edges : {dict(edge_names.most_common(10))}")

    assertions = [props for _id, props in nodes if (props or {}).get("type") == "Assertion"]
    if not assertions:
        print("  assertions: none (default extraction)")
        return

    total = len(assertions)
    verified = sum(1 for a in assertions if a.get("source_quote_verified") is True)
    with_quote = sum(1 for a in assertions if a.get("source_quote"))
    with_speaker = sum(1 for a in assertions if a.get("asserted_by"))
    grounded = sum(1 for _id, p in nodes if (p or {}).get("ontology_valid") is True)
    bad_names = [
        a.get("name", "")
        for a in assertions
        if NEGATION.search(a.get("name") or "") or SPEECH_ACT.search(a.get("name") or "")
    ]
    print(f"  assertions={total}")
    print(f"    statement_type: {dict(Counter(a.get('statement_type') for a in assertions))}")
    print(f"    polarity      : {dict(Counter(a.get('polarity') for a in assertions))}")
    print(f"    precision     : {dict(Counter(a.get('precision') for a in assertions))}")
    print(
        f"    source_quote  : {with_quote}/{total} present, {verified}/{total} verified "
        f"({pct(verified, total)})"
    )
    print(
        f"    asserted_by   : {with_speaker}/{total} ({pct(with_speaker, total)}) "
        f"| edges={edge_names.get('asserted_by', 0)}"
    )
    print(f"    ontology_valid nodes: {grounded}/{len(nodes)} ({pct(grounded, len(nodes))})")
    print(f"    names breaking the affirmative-proposition rule: {len(bad_names)}")
    for name in bad_names[:8]:
        print(f"      - {name}")


async def main() -> int:
    for dataset_name in sys.argv[1:]:
        await report(dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
