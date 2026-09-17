"""Recall lane budgets, made configurable by env.

Hybrid lane sizes are per-request (``retriever_specific_config``), but that dict is not
reachable over HTTP, so a UI can never raise them; an operator's only lever is an
environment variable. This mirrors ``cognee/modules/cognify/config.py``: a ``BaseSettings``
subclass, one ``@lru_cache`` getter, and a ``to_dict()`` for callers that want a plain
mapping (e.g. logging or a status endpoint) instead of the settings object.

Resolution order, see ``_hybrid_lane_top_k`` in
``cognee/modules/search/methods/get_search_type_retriever_instance.py``:

1. an explicit ``retriever_specific_config`` value always wins;
2. otherwise a field here that is **set** (not ``None``) replaces the request's ``top_k``;
3. otherwise the request's own ``top_k``, capped at ``DEFAULT_HYBRID_LANE_TOP_K`` (10).

The three request lanes default to ``None`` on purpose: with nothing set, a default search
behaves exactly as it did before this module existed. Raising them is an operator's
decision for a graph that needs the context (a legal graph's hybrid context measured 15.7k
characters against 57.7k for a plain graph of the same corpus) -- and a cost decision,
since every hybrid completion in the deployment pays for the bigger context.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseSettings):
    # Unset (None) = the request's top_k capped at 10, byte-identical to the behaviour
    # before this module. Every budget that IS set must be at least 1: the request-level
    # `top_k` is rejected at 0 with a 422, and a typo such as `HYBRID_ENTITIES_TOP_K=-5`
    # in .env would otherwise silently starve every hybrid search of its entity lane
    # instead of failing at startup. Note an *empty* `HYBRID_CHUNKS_TOP_K=` is an int
    # parsing error, not "unset": leave the variable out to leave the lane alone.
    hybrid_chunks_top_k: Optional[int] = Field(default=None, ge=1)
    hybrid_entities_top_k: Optional[int] = Field(default=None, ge=1)
    hybrid_facts_top_k: Optional[int] = Field(default=None, ge=1)
    # The statements lane is new with the assertion work and has no pre-existing
    # request-top_k behaviour to preserve; it costs nothing on a graph without an
    # Assertion_name collection, so it carries a real default.
    hybrid_statements_top_k: int = Field(default=20, ge=1)
    hybrid_max_edges_per_entity: int = Field(default=10, ge=1)
    # The DISPUTES registry entry reads this.
    disputes_top_k: int = Field(default=50, ge=1)
    # Gates assertion pair expansion in cognee/modules/retrieval/utils/assertion_pairs.py.
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
