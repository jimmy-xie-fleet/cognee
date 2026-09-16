"""Recall lane budgets, made configurable by env (Task 7).

Effectiveness over efficiency, by user decision: the legal graph fed the LLM far less
context than the plain graph did, because hybrid lane sizes and the graph-completion
pair-expansion switch were hard-coded per request. `retriever_specific_config` is not
reachable over HTTP, so a UI can never raise them -- an operator's only lever is an
environment variable. This mirrors `cognee/modules/cognify/config.py`: a `BaseSettings`
subclass, one `@lru_cache` getter, and a `to_dict()` for callers that want a plain
mapping (e.g. logging or a status endpoint) instead of the settings object.

Per-request overrides remain authoritative -- see `_hybrid_lane_top_k` in
`cognee/modules/search/methods/get_search_type_retriever_instance.py` -- these are only
the defaults that apply when a caller does not set one explicitly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseSettings):
    # Hybrid recall lane budgets. Raised well above the old request-`top_k`-derived
    # caps (10 for chunks/entities/facts, 20 for statements) so a legal graph's
    # completion gets a context comparable in size to the plain graph's.
    hybrid_chunks_top_k: int = 30
    hybrid_entities_top_k: int = 30
    hybrid_facts_top_k: int = 30
    hybrid_statements_top_k: int = 20
    hybrid_max_edges_per_entity: int = 20
    # Task 9's DISPUTES registry entry reads this.
    disputes_top_k: int = 50
    # Replaces the module-level `PAIR_EXPANSION_ENABLED` constant in
    # cognee/modules/retrieval/utils/assertion_pairs.py.
    graph_completion_pair_expansion: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "hybrid_chunks_top_k": self.hybrid_chunks_top_k,
            "hybrid_entities_top_k": self.hybrid_entities_top_k,
            "hybrid_facts_top_k": self.hybrid_facts_top_k,
            "hybrid_statements_top_k": self.hybrid_statements_top_k,
            "hybrid_max_edges_per_entity": self.hybrid_max_edges_per_entity,
            "disputes_top_k": self.disputes_top_k,
            "graph_completion_pair_expansion": self.graph_completion_pair_expansion,
        }


@lru_cache
def get_retrieval_config():
    return RetrievalConfig()
