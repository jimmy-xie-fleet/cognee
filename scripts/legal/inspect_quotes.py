"""Why did some assertions fail source-quote verification, and how well did grounding do?"""

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


async def main() -> int:
    dataset_name = sys.argv[1]
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.engine.models.Assertion import verify_source_quote
    from sqlalchemy import select

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        dataset = (
            await session.execute(select(Dataset).where(Dataset.name == dataset_name))
        ).scalar_one_or_none()

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()

    by_id = {str(node_id): (props or {}) for node_id, props in nodes}
    chunks = {i: p for i, p in by_id.items() if p.get("type") == "DocumentChunk"}
    assertions = {i: p for i, p in by_id.items() if p.get("type") == "Assertion"}

    # ontology grounding by node type
    grounded = Counter()
    totals = Counter()
    for props in by_id.values():
        totals[props.get("type")] += 1
        if props.get("ontology_valid") is True:
            grounded[props.get("type")] += 1
    print("ontology_valid by node type:")
    for node_type, total in totals.most_common():
        print(f"  {node_type:15s} {grounded.get(node_type, 0)}/{total}")

    unverified = [
        (i, p) for i, p in assertions.items() if p.get("source_quote_verified") is not True
    ]
    print(f"\nunverified quotes: {len(unverified)}/{len(assertions)}")
    for node_id, props in unverified:
        quote = props.get("source_quote") or ""
        chunk_id = props.get("source_chunk_id")
        chunk = chunks.get(str(chunk_id))
        chunk_text = (chunk or {}).get("text") or ""
        # is it in ANY chunk of this dataset?
        elsewhere = [
            cid for cid, c in chunks.items() if verify_source_quote(quote, c.get("text") or "")
        ]
        print(f"\n  assertion: {props.get('name')[:90]}")
        print(f"    quote   : {quote[:150]!r}")
        print(
            f"    chunk   : {str(chunk_id)[:8]} present={chunk is not None} len={len(chunk_text)}"
        )
        print(f"    found in other chunks: {len(elsewhere)}")
        if chunk_text and quote:
            head = quote.strip()[:40]
            print(
                f"    chunk contains first 40 chars of quote: {head.lower() in chunk_text.lower()}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
