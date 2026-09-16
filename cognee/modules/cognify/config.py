from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.data_models import DefaultContentPrediction, SummarizedContent
from typing import Optional
import os


class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    triplet_embedding: bool = False
    chunks_per_batch: Optional[int] = None
    # Opt-in contradiction detection (issue #3699). Default OFF so the standard
    # cognify pipeline is unchanged. Tunables gate the verdict and the LLM payload.
    contradiction_detection: bool = False
    contradiction_confidence_threshold: float = 0.5
    contradiction_max_facts: int = 500
    # Opt-in audit-grade provenance ledger (env: PROVENANCE_TRACKING). Default
    # OFF so the standard cognify pipeline is unchanged.
    provenance_tracking: bool = False
    # Assertion-reference resolution: what one resolver pass may spend and how sure the
    # tracer has to be before a link is written. reference_llm_max_calls is the whole
    # pass's budget (shared by every reference it traces); reference_tracer_max_iter is
    # the per-reference cap, each iteration being one tool step or one finish.
    reference_llm_max_calls: int = 300
    reference_tracer_max_iter: int = 4
    reference_llm_confidence_threshold: float = 0.6
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "triplet_embedding": self.triplet_embedding,
            "chunks_per_batch": self.chunks_per_batch,
            "contradiction_detection": self.contradiction_detection,
            "contradiction_confidence_threshold": self.contradiction_confidence_threshold,
            "contradiction_max_facts": self.contradiction_max_facts,
            "provenance_tracking": self.provenance_tracking,
            "reference_llm_max_calls": self.reference_llm_max_calls,
            "reference_tracer_max_iter": self.reference_tracer_max_iter,
            "reference_llm_confidence_threshold": self.reference_llm_confidence_threshold,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()
