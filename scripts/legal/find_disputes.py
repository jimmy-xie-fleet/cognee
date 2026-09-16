"""Find disputes in a legal-profile dataset without calling an LLM.

Three independent signals, strongest first:
  1. responds_to  - a denial explicitly answering another assertion
  2. stance split - the same proposition asserted with opposite polarity
  3. speaker split- the same proposition asserted by different speakers
Also reports whether anything in the graph was superseded or closed, which is
what would suppress a dispute if cognee had resolved it during ingestion.

Signal 1 is reported twice: once from the ``responds_to`` field (an id another
assertion already carried at extraction time) and once from the ``responds_to``
*edges* the reference resolver writes (see
``cognee/tasks/graph/resolve_assertion_references.py``). The two rarely match: a
resolved paragraph reference often anchors on a ``DocumentChunk`` rather than a single
assertion, so the field-level count undercounts disputes the graph can actually
traverse.
"""

import asyncio
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/legal/find_disputes.py <dataset-name>")
        return 2

    dataset_name = sys.argv[1]
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.tasks.graph.reference_graph_view import DOCUMENT_NODE_TYPES
    from sqlalchemy import select

    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        dataset = (
            await session.execute(select(Dataset).where(Dataset.name == dataset_name))
        ).scalar_one_or_none()

    if dataset is None:
        print(f"No such dataset: {dataset_name}")
        return 1

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()

    assertions = {
        str(node_id): props for node_id, props in nodes if (props or {}).get("type") == "Assertion"
    }
    print(f"dataset={dataset_name} assertions={len(assertions)}")

    # Did anything get resolved away during ingestion?
    closed = [a for a in assertions.values() if a.get("valid_to") is not None]
    superseded_edges = [e for e in edges if (e[3] or {}).get("superseded")]
    contradicts_edges = [e for e in edges if e[2] == "contradicts"]
    print(
        f"suppression check: closed_assertions={len(closed)} "
        f"superseded_edges={len(superseded_edges)} contradicts_edges={len(contradicts_edges)}"
    )

    # 1. explicit responses
    responds = [a for a in assertions.values() if a.get("responds_to")]
    responds_to_assertion = [a for a in responds if str(a.get("responds_to")) in assertions]
    print(
        f"\n1. responds_to: {len(responds)} assertions, "
        f"{len(responds_to_assertion)} resolved to another assertion node"
    )
    for a in responds_to_assertion[:6]:
        target = assertions[str(a.get("responds_to"))]
        print(f"   [{a.get('statement_type')}/{a.get('polarity')}] {a.get('name')[:66]}")
        print(
            f"      -> [{target.get('statement_type')}/{target.get('polarity')}] "
            f"{target.get('name')[:60]}"
        )

    # 1b. explicit responses, at the edge level -- the field may now hold a chunk id when
    # a paragraph anchors several allegations, so this is the count that matters once the
    # reference resolver has run (see scripts/legal/resolve_references_report.py).
    node_type_by_id = {str(node_id): (props or {}).get("type") for node_id, props in nodes}
    responds_to_edges = [e for e in edges if e[2] == "responds_to"]
    assertion_to_assertion = [
        e
        for e in responds_to_edges
        if node_type_by_id.get(str(e[0])) == "Assertion"
        and node_type_by_id.get(str(e[1])) == "Assertion"
    ]
    opposite_polarity_edges = [
        e
        for e in assertion_to_assertion
        if {assertions[str(e[0])].get("polarity"), assertions[str(e[1])].get("polarity")}
        == {"positive", "negative"}
    ]
    passage_level_edges = [
        e for e in responds_to_edges if node_type_by_id.get(str(e[1])) == "DocumentChunk"
    ]
    document_level_edges = [
        e for e in responds_to_edges if node_type_by_id.get(str(e[1])) in DOCUMENT_NODE_TYPES
    ]
    print(
        f"\n1b. responds_to edges: {len(responds_to_edges)} total, "
        f"{len(assertion_to_assertion)} assertion->assertion, "
        f"{len(opposite_polarity_edges)} with opposite polarity"
    )
    print(
        f"    anchored on a passage (DocumentChunk): {len(passage_level_edges)}, "
        f"anchored on a document: {len(document_level_edges)}"
    )
    strategy_counts = Counter(
        (props or {}).get("resolution_strategy") for _, _, _, props in responds_to_edges
    )
    print(f"    resolution_strategy breakdown: {dict(strategy_counts)}")
    for e in opposite_polarity_edges[:6]:
        source, target = assertions[str(e[0])], assertions[str(e[1])]
        print(
            f"   [{source.get('statement_type')}/{source.get('polarity')}] "
            f"{(source.get('name') or '')[:50]} → "
            f"[{target.get('statement_type')}/{target.get('polarity')}] "
            f"{(target.get('name') or '')[:50]}"
        )

    # 2 & 3. same proposition, different stance or speaker
    by_name = defaultdict(list)
    for a in assertions.values():
        by_name[(a.get("name") or "").strip().casefold()].append(a)
    shared = {n: v for n, v in by_name.items() if len(v) > 1}
    stance_split = {
        n: v
        for n, v in shared.items()
        if len({a.get("polarity") for a in v if a.get("polarity") != "unknown"}) > 1
    }
    speaker_split = {
        n: v
        for n, v in shared.items()
        if len({a.get("asserted_by") for a in v if a.get("asserted_by")}) > 1
    }
    print(
        f"\n2/3. distinct propositions={len(by_name)}, shared by 2+ assertions={len(shared)}"
        f"\n     opposite stance on the same proposition: {len(stance_split)}"
        f"\n     different speakers on the same proposition: {len(speaker_split)}"
    )
    for name, group in list(stance_split.items())[:6]:
        print(f"\n   DISPUTE: {name[:88]}")
        for a in group:
            print(
                f"      [{a.get('polarity'):8s}] {a.get('statement_type'):10s} "
                f"by {str(a.get('asserted_by'))[:28]}"
            )

    print("\n   (top repeated propositions, for reference)")
    for name, group in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:5]:
        stances = Counter(a.get("polarity") for a in group)
        print(f"      x{len(group)} {dict(stances)} {name[:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
