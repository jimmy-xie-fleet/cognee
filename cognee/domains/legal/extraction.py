"""Per-chunk extraction for the legal profile: plain graph + assertions, boilerplate filtered.

The legal prompt asks for entities and assertions in one LLM call, and a repeated recall
eval measured the cost: the legal graph lost the plain graph's numeric and date facts
(valuation and timeline questions 15 to 20 points behind) while what it retrieved was
dominated by boilerplate statements -- "defendants repeat their prior responses", "X are
attorneys for Y", certifications that no other action is pending. Two fixes, both here:

* **Two passes per chunk.** Pass 1 is cognee's default extraction (default prompt,
  ``KnowledgeGraph``), byte-identical to plain ingestion; pass 2 is the legal prompt.
  The two graphs are merged into one ``LegalKnowledgeGraph`` before construction, so the
  legal graph is a superset of the plain one. Chunking runs once and both passes read the
  same chunk; the merged graph attaches to that one ``DocumentChunk``.
* **Salience.** The legal prompt marks every assertion ``high``/``medium``/``low``. The
  mark becomes the node's ``importance_weight`` (read by graph-completion scoring), and
  ``low`` assertions are dropped before construction unless a denial or admission (they
  carry the dispute structure) or another retained assertion refers to them.

Wired in through ``cognify(calculate_chunk_graphs=...)``, the hook
``extract_graph_from_data`` calls in place of its own per-chunk LLM call; the profile
returns it from ``legal_profile()``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional

from cognee.domains.legal.models import LegalKnowledgeGraph, LegalNode, Salience
from cognee.infrastructure.llm.extraction.knowledge_graph.extract_content_graph import (
    extract_content_graph,
)
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.modules.engine.models.Assertion import StatementType
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    _ASSERTION_REFERENCE_FIELD_NAMES,
    is_assertion_node,
    prune_extracted_graph,
)
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.shared.logging_utils import get_logger

logger = get_logger("legal.extraction")

PLAIN_ID_PREFIX = "p:"
LEGAL_ID_PREFIX = "l:"

# What a salience mark is worth at retrieval. Graph-completion scoring multiplies a
# node's distance by ``(2 - importance_weight)``, so ``high`` halves the penalty of a
# ``low`` statement that survived the drop (the drop is off, or the statement is a
# denial/admission, or something retained refers to it).
SALIENCE_IMPORTANCE: dict[Salience, float] = {
    Salience.HIGH: 0.9,
    Salience.MEDIUM: 0.5,
    Salience.LOW: 0.2,
}

# Statement types that are never dropped for being boilerplate: a positional "17. Denied."
# is low information as text and is exactly the link between the pleadings.
NEVER_DROPPED_STATEMENT_TYPES = frozenset({StatementType.DENIAL, StatementType.ADMISSION})


@dataclass(frozen=True)
class LegalExtractionOptions:
    two_pass: bool = True
    drop_low_salience: bool = True


@dataclass(frozen=True)
class SalienceDropCounts:
    total_assertions: int
    dropped_assertions: int
    dropped_edges: int
    unscored_assertions: int


# --------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------


def _statement_type_of(node: Node) -> Optional[StatementType]:
    value = getattr(node, "statement_type", None)
    if value is None:
        return None
    if isinstance(value, StatementType):
        return value
    try:
        return StatementType(str(value))
    except ValueError:
        return None


def _prefix_graph_ids(graph: KnowledgeGraph, prefix: str) -> None:
    """Prefix every node id, the edge endpoints, and the assertion references that name a node.

    Reference fields (``asserted_by``, ``attributed_to``, ``responds_to``) hold graph-local
    ids; after the prefix they would resolve by *name* against
    ``generate_node_name(reference)``, which for an id like ``meridian`` no longer matches
    the node named "Meridian Logistics, Inc." -- the speaker would be stored as free text
    and no edge derived. So a reference equal to a node id is prefixed with it. Locator
    text ("Complaint ¶17") never equals a node id and is left alone.
    """
    node_ids = {node.id for node in graph.nodes}
    for node in graph.nodes:
        node.id = f"{prefix}{node.id}"
        if not is_assertion_node(node):
            continue
        for field_name in _ASSERTION_REFERENCE_FIELD_NAMES:
            reference = getattr(node, field_name, None)
            if isinstance(reference, str) and reference in node_ids:
                setattr(node, field_name, f"{prefix}{reference}")
    for edge in graph.edges:
        if edge.source_node_id in node_ids:
            edge.source_node_id = f"{prefix}{edge.source_node_id}"
        if edge.target_node_id in node_ids:
            edge.target_node_id = f"{prefix}{edge.target_node_id}"


def _as_legal_node(node: Node) -> LegalNode:
    """A plain extracted node as a ``LegalNode`` entity: no statement type, so never an assertion."""
    if isinstance(node, LegalNode):
        return node
    return LegalNode(
        id=node.id,
        name=node.name,
        type=node.type,
        description=node.description,
    )


def merge_plain_and_legal_graphs(
    plain: KnowledgeGraph, legal: LegalKnowledgeGraph
) -> LegalKnowledgeGraph:
    """One graph out of the two passes over a chunk.

    Ids are prefixed per pass so nothing collides. A plain entity whose normalized name
    matches a legal entity is folded onto the legal node (its type comes from the
    profile's vocabulary and its description from the legal context); the plain edges
    are repointed and any edge that became a self-loop is dropped. Every other plain node
    is carried over as an entity. Assertions come from the legal pass only.

    Both inputs are mutated (ids rewritten); callers pass fresh extraction output.
    """
    _prefix_graph_ids(plain, PLAIN_ID_PREFIX)
    _prefix_graph_ids(legal, LEGAL_ID_PREFIX)

    legal_id_by_name: dict[str, str] = {}
    for node in legal.nodes:
        if is_assertion_node(node):
            continue
        legal_id_by_name.setdefault(generate_node_name(node.name), node.id)

    alias: dict[str, str] = {}
    carried: list[LegalNode] = []
    for node in plain.nodes:
        survivor = legal_id_by_name.get(generate_node_name(node.name))
        if survivor is not None:
            alias[node.id] = survivor
            continue
        carried.append(_as_legal_node(node))

    merged_edges: list[KGEdge] = list(legal.edges)
    for edge in plain.edges:
        source = alias.get(edge.source_node_id, edge.source_node_id)
        target = alias.get(edge.target_node_id, edge.target_node_id)
        if source == target:
            continue
        edge.source_node_id = source
        edge.target_node_id = target
        merged_edges.append(edge)

    return LegalKnowledgeGraph(nodes=[*legal.nodes, *carried], edges=merged_edges)


# --------------------------------------------------------------------------------------
# Salience
# --------------------------------------------------------------------------------------


def _salience_of(node: Node) -> Optional[Salience]:
    value = getattr(node, "salience", None)
    if value is None:
        return None
    if isinstance(value, Salience):
        return value
    try:
        return Salience(str(value))
    except ValueError:
        return None


def apply_salience(graph: LegalKnowledgeGraph) -> int:
    """Map each assertion's salience onto ``importance_weight``. Returns the unscored count.

    Entities are left alone: their weight stays the chunk's. An assertion the model did
    not score keeps ``None`` (so construction falls back to the chunk) and is counted, so
    an eval can see whether the model fills the field.
    """
    unscored = 0
    for node in graph.nodes:
        if not is_assertion_node(node):
            continue
        salience = _salience_of(node)
        if salience is None:
            unscored += 1
            continue
        node.importance_weight = SALIENCE_IMPORTANCE[salience]
    return unscored


def drop_low_salience_assertions(graph: LegalKnowledgeGraph) -> SalienceDropCounts:
    """Remove ``low`` assertions the graph can spare, pruning their edges and references.

    Kept regardless of salience: denials and admissions (the dispute structure), and any
    assertion another retained assertion refers to through ``responds_to`` or
    ``attributed_to`` (dropping the target would leave the answer pointing at nothing).
    """
    assertion_nodes = [node for node in graph.nodes if is_assertion_node(node)]
    unscored = sum(1 for node in assertion_nodes if _salience_of(node) is None)

    dropped = {
        node.id
        for node in assertion_nodes
        if _salience_of(node) is Salience.LOW
        and _statement_type_of(node) not in NEVER_DROPPED_STATEMENT_TYPES
    }
    protected: set[str] = set()
    for node in assertion_nodes:
        if node.id in dropped:
            continue
        for field_name in ("responds_to", "attributed_to"):
            reference = getattr(node, field_name, None)
            if isinstance(reference, str) and reference in dropped:
                protected.add(reference)
    dropped -= protected

    dropped_edges = prune_extracted_graph(graph, dropped) if dropped else 0
    counts = SalienceDropCounts(
        total_assertions=len(assertion_nodes),
        dropped_assertions=len(dropped),
        dropped_edges=dropped_edges,
        unscored_assertions=unscored,
    )
    if counts.dropped_assertions:
        logger.info(
            "Legal profile dropped %s of %s assertion(s) as low salience (%s edge(s))",
            counts.dropped_assertions,
            counts.total_assertions,
            counts.dropped_edges,
        )
    return counts


# --------------------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------------------


async def extract_legal_chunk_graph(
    text: str, custom_prompt: Optional[str], *, two_pass: bool
) -> LegalKnowledgeGraph:
    """Extract one chunk: the legal pass, plus the default pass merged in when ``two_pass``."""
    if not two_pass:
        return await extract_content_graph(text, LegalKnowledgeGraph, custom_prompt=custom_prompt)

    plain, legal = await asyncio.gather(
        extract_content_graph(text, KnowledgeGraph, custom_prompt=None),
        extract_content_graph(text, LegalKnowledgeGraph, custom_prompt=custom_prompt),
    )
    return merge_plain_and_legal_graphs(plain, legal)


def legal_chunk_graphs(
    *, two_pass: bool = True, drop_low_salience: bool = True
) -> Callable[..., Any]:
    """The ``calculate_chunk_graphs`` callable ``legal_profile()`` hands to ``cognify()``.

    Signature as ``extract_graph_from_data`` calls it:
    ``(data_chunks, graph_model, custom_prompt, **kwargs) -> list[LegalKnowledgeGraph]``.
    ``kwargs`` are cognify's own (they include this very callable) and are not forwarded
    to the LLM. All chunks are extracted concurrently, as the core path does.
    """
    options = LegalExtractionOptions(two_pass=two_pass, drop_low_salience=drop_low_salience)

    async def calculate(data_chunks, graph_model, custom_prompt, **kwargs):
        if graph_model is not LegalKnowledgeGraph:
            raise ValueError(
                "legal_chunk_graphs extracts LegalKnowledgeGraph only; "
                f"got {getattr(graph_model, '__name__', graph_model)!r}"
            )
        with pipeline_stage("extraction"):
            graphs = await asyncio.gather(
                *[
                    extract_legal_chunk_graph(chunk.text, custom_prompt, two_pass=options.two_pass)
                    for chunk in data_chunks
                ]
            )
        unscored_total = 0
        for graph in graphs:
            unscored_total += apply_salience(graph)
            if options.drop_low_salience:
                drop_low_salience_assertions(graph)
        if unscored_total:
            logger.info(
                "Legal profile: %s assertion(s) came back without a salience mark",
                unscored_total,
            )
        return list(graphs)

    calculate.options = options  # type: ignore[attr-defined]
    return calculate
