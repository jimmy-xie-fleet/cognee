from typing import Optional

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from cognee.modules.engine.models.Assertion import CaseInsensitiveEnum, StatementType
from cognee.shared.data_models import KnowledgeGraph, Node  # whichever provider branch is active


class Polarity(CaseInsensitiveEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    # A passage that records no stance at all. Never a default for one that does:
    # construction stores an omitted polarity as "unknown" already.
    UNKNOWN = "unknown"


class Precision(CaseInsensitiveEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class LocatorKind(CaseInsensitiveEnum):
    PARAGRAPH = "paragraph"
    SECTION = "section"
    EXHIBIT = "exhibit"
    COUNT = "count"
    ARTICLE = "article"
    PAGE = "page"
    # The passage names the source as a whole, not a place inside it.
    NONE = "none"


class ReferenceBasis(CaseInsensitiveEnum):
    CITED = "cited"
    POSITIONAL = "positional"
    DESCRIBED = "described"


class Salience(CaseInsensitiveEnum):
    """How much a statement is worth retrieving.

    The eval showed a legal graph's retrieved statements dominated by boilerplate --
    "defendants repeat their prior responses", "X are attorneys for Y", certifications
    that no other action is pending, who appeared at a hearing. Those crowd substantive
    statements out of a context. The model marks them ``low``; the profile drops them
    (or, with the drop off, stores them down-weighted).
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LegalReference(BaseModel):
    """A reference to a source outside the passage being extracted.

    Never a composed string: `document_hint`/`locator_kind`/`locator_value` are kept
    apart so a resolver can match them against known documents without parsing prose.
    """

    document_hint: str = Field(
        "",
        description=(
            "The referenced source as the passage names it, in the passage's own "
            "words: 'the Fester Report', 'the June 10 letter', 'the Complaint', 'my "
            "September 22, 2026 deposition'. Never a filename, never invented."
        ),
    )
    locator_kind: LocatorKind = Field(
        LocatorKind.NONE,
        description=(
            "The kind of place inside that source the passage names; none when the "
            "passage names the source as a whole."
        ),
    )
    locator_value: Optional[str] = Field(
        None,
        description=(
            "The number or letter of that place exactly as written: '13', '4.2', "
            "'C', 'II'. Null when locator_kind is none. Never invent one."
        ),
    )
    date: Optional[str] = Field(
        None,
        description="ISO date the passage attaches to the reference (YYYY-MM-DD, YYYY-MM or YYYY).",
    )
    basis: ReferenceBasis = Field(
        ReferenceBasis.DESCRIBED,
        description=(
            "cited when the passage writes the reference out; positional when a "
            "responsive pleading answers by position and cites nothing; described "
            "otherwise."
        ),
    )


class LegalNode(Node):
    name: str = Field(
        default="",
        description=(
            "For assertion nodes: the underlying proposition phrased affirmatively as one "
            "declarative sentence; no negation words, no speech-act verbs. For entity "
            "nodes: the most complete name in the passage."
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
            "denies or negates it, unknown only when the passage records no stance. "
            "Independent of statement_type."
        ),
    )
    asserted_by: Optional[str] = Field(
        None, description="id of the node for the person or organization making this statement."
    )
    attributed_to: Optional[str] = Field(
        None,
        description=(
            "id of the original author's node when that author appears in THIS "
            "passage; for a statement in another document use `attributed_to_ref`."
        ),
    )
    attributed_to_ref: Optional[LegalReference] = Field(
        None,
        description=(
            "Structured reference to the original author's source when it is in "
            "another document; null when that author appears in THIS passage (use "
            "attributed_to for that) or there is no attribution."
        ),
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
            "id of the node this statement responds to when that node appears in "
            "THIS passage; for a statement in another document use `responds_to_ref`."
        ),
    )
    responds_to_ref: Optional[LegalReference] = Field(
        None,
        description=(
            "Structured reference to the statement this responds to when it is in "
            "another document; null when the answered statement appears in THIS "
            "passage (use responds_to for that) or when there is no response."
        ),
    )
    salience: Optional[Salience] = Field(
        None,
        description=(
            "Assertion nodes only. high: a contested fact, quantity, date, valuation, "
            "term or finding a reader would cite. medium: an ordinary substantive "
            "statement. low: boilerplate -- captions, court/venue/jurisdiction recitals, "
            "appearances and 'attorneys for', 'repeats prior responses' / 'reserves all "
            "rights or positions', certifications that no other action is pending, "
            "'submitted in the context of settlement', attendance at a hearing, meeting "
            "or deposition. Never low for a denial or admission. Null for entities."
        ),
    )
    # Set by the pipeline from ``salience``; never asked of the model, so it is kept out
    # of the JSON schema the LLM sees.
    importance_weight: SkipJsonSchema[Optional[float]] = None


class LegalKnowledgeGraph(KnowledgeGraph):
    nodes: list[LegalNode] = Field(default_factory=list)
