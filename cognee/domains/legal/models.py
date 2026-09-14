from enum import Enum
from typing import Optional

from pydantic import Field

from cognee.modules.engine.models.Assertion import StatementType
from cognee.shared.data_models import KnowledgeGraph, Node  # whichever provider branch is active


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Precision(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class LegalNode(Node):
    name: str = Field(
        default="",
        description=(
            "The underlying proposition phrased affirmatively as one declarative sentence; "
            "no negation words, no speech-act verbs."
        ),
    )
    statement_type: Optional[StatementType] = Field(
        None,
        description=(
            "Set ONLY for assertion nodes: the speech act (allegation, denial, ...), not "
            "the stance."
        ),
    )
    polarity: Optional[Polarity] = Field(
        None,
        description=(
            "The speaker's stance on the name proposition: positive affirms it, negative "
            "denies or negates it. Independent of statement_type."
        ),
    )
    asserted_by: Optional[str] = Field(
        None, description="id of the node for the person or organization making this statement."
    )
    attributed_to: Optional[str] = Field(
        None,
        description="id of the original author when the speaker reports someone else's opinion or finding.",
    )
    applicable_time: Optional[str] = Field(
        None, description="ISO date (YYYY-MM-DD, YYYY-MM or YYYY) the claimed fact is about."
    )
    applies_from: Optional[str] = Field(
        None, description="ISO start of the period the claim covers (terms, rents, tenancies)."
    )
    applies_to: Optional[str] = Field(None, description="ISO end of that period, if stated.")
    report_date: Optional[str] = Field(
        None, description="ISO date the statement was made, signed, filed or dated."
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions, exceptions or caveats, verbatim where possible.",
    )
    precision: Optional[Precision] = Field(
        None, description="exact, approximate or unknown. Unknown is never zero."
    )
    scope: Optional[str] = Field(
        None,
        description="What the claim is limited to: property/unit, period, denominator (gross vs rentable), case.",
    )
    source_quote: Optional[str] = Field(
        None,
        description="Verbatim contiguous passage copied from the input that supports this claim.",
    )
    responds_to: Optional[str] = Field(
        None,
        description=(
            "Locator of the statement this responds to, e.g. 'Complaint ¶17', or that "
            "node's id when present."
        ),
    )


class LegalKnowledgeGraph(KnowledgeGraph):
    nodes: list[LegalNode] = Field(default_factory=list)
