"""What a correct legal extraction of each fixture passage should produce.

One ``LegalKnowledgeGraph`` per passage in ``fixtures/``, hand-written against
``cognee/domains/legal/prompts/legal_extraction_system.txt``: node types come from the
prompt's vocabulary (so ontology grounding recognizes them), ids are human-readable,
every ``source_quote`` is copied verbatim out of its passage, and the qualifier fields
carry the distinctions the profile exists to keep — who said it, when it was said as
against when it happened, whether it was denied, and under what conditions.

Every assertion ``name`` is the underlying proposition phrased affirmatively:
``statement_type`` carries the speech act and ``polarity`` the speaker's stance on that
proposition, so an allegation and the denial answering it share one name and differ only
in speech act, speaker and stance.

``test_extraction_fixtures.py`` feeds these graphs to the real construction path in
place of the LLM's output, so they are both the tests' input and the profile's worked
examples of the behaviour the prompt asks for.
"""

from cognee.domains.legal import LegalKnowledgeGraph, LegalNode, Polarity, Precision
from cognee.modules.engine.models.Assertion import StatementType
from cognee.shared.data_models import Edge as KGEdge


def _complaint_p17_p18_warning() -> LegalKnowledgeGraph:
    """Two pleaded allegations: when they were filed is not when they happened."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="okafor",
                name="Amara Okafor",
                type="Person",
                description="Plaintiff, a dispatch coordinator at Meridian Logistics, Inc.",
            ),
            LegalNode(
                id="meridian",
                name="Meridian Logistics, Inc.",
                type="Company",
                description="Defendant employer.",
            ),
            LegalNode(
                id="complaint",
                name="Complaint in Okafor v. Meridian Logistics, Inc.",
                type="Complaint",
                description="Complaint and demand for jury trial filed March 10, 2026.",
            ),
            LegalNode(
                id="written-warning",
                name="Written warning issued to Amara Okafor",
                type="Notice",
                description="Written warning for insubordination dated March 3, 2025.",
            ),
            LegalNode(
                id="allegation-p17-report",
                name=(
                    "Okafor reported to Meridian's Director of Human Resources that her "
                    "supervisor instructed staff to falsify driver hours-of-service logs"
                ),
                type="Allegation",
                description=(
                    "Paragraph 17 of the Complaint: Okafor pleads that she made the report "
                    "on February 20, 2025."
                ),
                statement_type=StatementType.ALLEGATION,
                polarity=Polarity.POSITIVE,
                asserted_by="okafor",
                applicable_time="2025-02-20",
                report_date="2026-03-10",
                source_quote=(
                    "On February 20, 2025, Plaintiff Amara Okafor reported to Meridian's "
                    "Director of Human Resources that her supervisor had instructed "
                    "warehouse staff to falsify driver hours-of-service logs."
                ),
            ),
            LegalNode(
                id="allegation-p18-retaliation",
                name=(
                    "Meridian issued Okafor a written warning on March 3, 2025 in "
                    "retaliation for her February 20, 2025 report"
                ),
                type="Allegation",
                description=(
                    "Paragraph 18 of the Complaint: Okafor pleads retaliation and no "
                    "legitimate business reason for the warning."
                ),
                statement_type=StatementType.ALLEGATION,
                polarity=Polarity.POSITIVE,
                asserted_by="okafor",
                applicable_time="2025-03-03",
                report_date="2026-03-10",
                scope="the written warning for insubordination dated March 3, 2025",
                source_quote=(
                    "the written warning was issued in retaliation for her February 20, 2025 report"
                ),
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="okafor",
                target_node_id="complaint",
                relationship_name="party_to",
                description="Amara Okafor is the plaintiff in the Complaint.",
            ),
            KGEdge(
                source_node_id="meridian",
                target_node_id="complaint",
                relationship_name="party_to",
                description="Meridian Logistics, Inc. is the defendant in the Complaint.",
            ),
            KGEdge(
                source_node_id="okafor",
                target_node_id="meridian",
                relationship_name="employed_by",
                description="Amara Okafor worked for Meridian Logistics, Inc.",
            ),
            KGEdge(
                source_node_id="allegation-p18-retaliation",
                target_node_id="written-warning",
                relationship_name="about",
                description=(
                    "The retaliation allegation is about the written warning issued to "
                    "Amara Okafor."
                ),
            ),
        ],
    )


def _answer_p17_denial() -> LegalKnowledgeGraph:
    """A denial stays a denial: the affirmative proposition, negative stance.

    Nothing is re-emitted as a positive statement of the opposite fact, and the recited
    Paragraph 18 allegation and its denial share one name.
    """
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="okafor",
                name="Amara Okafor",
                type="Person",
                description="Plaintiff.",
            ),
            LegalNode(
                id="meridian",
                name="Meridian Logistics, Inc.",
                type="Company",
                description="Defendant, the answering party.",
            ),
            LegalNode(
                id="complaint",
                name="Complaint in Okafor v. Meridian Logistics, Inc.",
                type="Complaint",
                description="The pleading this Answer responds to.",
            ),
            LegalNode(
                id="answer",
                name="Answer of Meridian Logistics, Inc.",
                type="Answer",
                description="Answer filed April 7, 2026 in Case No. 26-cv-00814.",
            ),
            LegalNode(
                id="delacroix",
                name="Corinne Delacroix",
                type="Person",
                description="Counsel for Defendant Meridian Logistics, Inc.",
            ),
            LegalNode(
                id="denial-p17",
                name="The allegations of Paragraph 17 of the Complaint are true",
                type="Denial",
                description=(
                    "Meridian's answer to Paragraph 17. The passage states no proposition "
                    "of its own, so the denial is of the paragraph's allegations; it does "
                    "not assert that the opposite is true."
                ),
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="Complaint ¶17",
                source_quote=(
                    "Defendant denies each and every allegation of Paragraph 17 of the Complaint"
                ),
            ),
            LegalNode(
                id="allegation-p18-recited",
                name=(
                    "Meridian issued Okafor a written warning on March 3, 2025 in "
                    "retaliation for her February 20, 2025 report"
                ),
                type="Allegation",
                description=(
                    "The allegation of Paragraph 18 of the Complaint, as the Answer recites "
                    "it before denying it. Okafor is the one making it."
                ),
                statement_type=StatementType.ALLEGATION,
                polarity=Polarity.POSITIVE,
                asserted_by="okafor",
                applicable_time="2025-03-03",
                source_quote=(
                    "the written warning issued to Ms. Okafor on March 3, 2025 was issued "
                    "in retaliation for her February 20, 2025 report"
                ),
            ),
            LegalNode(
                id="denial-p18",
                name=(
                    "Meridian issued Okafor a written warning on March 3, 2025 in "
                    "retaliation for her February 20, 2025 report"
                ),
                type="Denial",
                description=(
                    "Meridian's answer to the recited Paragraph 18 allegation: the same "
                    "proposition, denied."
                ),
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="allegation-p18-recited",
                source_quote="Denied.",
            ),
            LegalNode(
                id="statement-audit",
                name=(
                    "An internal audit of Meridian's driver hours-of-service records "
                    "completed on February 14, 2025 identified falsified entries"
                ),
                type="Statement",
                description=(
                    "A factual statement Meridian makes on its own account, negating the "
                    "proposition for the period the audit reviewed."
                ),
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                applicable_time="2025-02-14",
                report_date="2026-04-07",
                scope="the period reviewed by the audit",
                source_quote=(
                    "an internal audit of its driver hours-of-service records, completed on "
                    "February 14, 2025, identified no falsified entries"
                ),
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="answer",
                target_node_id="complaint",
                relationship_name="responds_to",
                description="The Answer of Meridian Logistics, Inc. responds to the Complaint.",
            ),
            KGEdge(
                source_node_id="meridian",
                target_node_id="answer",
                relationship_name="party_to",
                description="Meridian Logistics, Inc. filed the Answer.",
            ),
            KGEdge(
                source_node_id="delacroix",
                target_node_id="answer",
                relationship_name="signed",
                description="Corinne Delacroix signed the Answer for Meridian Logistics, Inc.",
            ),
            KGEdge(
                source_node_id="okafor",
                target_node_id="complaint",
                relationship_name="party_to",
                description="Amara Okafor is the plaintiff in the Complaint.",
            ),
        ],
    )


def _answer_p2_partial() -> LegalKnowledgeGraph:
    """One paragraph, six answers: each admission and denial is its own node."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="okafor",
                name="Amara Okafor",
                type="Person",
                description="Plaintiff.",
            ),
            LegalNode(
                id="meridian",
                name="Meridian Logistics, Inc.",
                type="Company",
                description="Defendant, the answering party.",
            ),
            LegalNode(
                id="complaint",
                name="Complaint in Okafor v. Meridian Logistics, Inc.",
                type="Complaint",
                description="The pleading this Answer responds to.",
            ),
            LegalNode(
                id="admission-employment",
                name=(
                    "Meridian employed Okafor as a dispatch coordinator from June 3, 2019 "
                    "through April 11, 2025"
                ),
                type="Admission",
                description="Meridian admits the employment dates pleaded in Paragraph 2.",
                statement_type=StatementType.ADMISSION,
                polarity=Polarity.POSITIVE,
                asserted_by="meridian",
                applies_from="2019-06-03",
                applies_to="2025-04-11",
                report_date="2026-04-07",
                precision=Precision.EXACT,
                responds_to="Complaint ¶2",
                source_quote=(
                    "it employed Plaintiff as a dispatch coordinator from June 3, 2019 "
                    "through April 11, 2025"
                ),
            ),
            LegalNode(
                id="admission-salary",
                name="Okafor's annual salary was $68,400 as of January 1, 2025",
                type="Admission",
                description="Meridian admits the salary figure pleaded in Paragraph 2.",
                statement_type=StatementType.ADMISSION,
                polarity=Polarity.POSITIVE,
                asserted_by="meridian",
                applicable_time="2025-01-01",
                report_date="2026-04-07",
                precision=Precision.EXACT,
                responds_to="Complaint ¶2",
                source_quote="Plaintiff's annual salary was $68,400 as of January 1, 2025",
            ),
            LegalNode(
                id="denial-title",
                name="Okafor held the title of operations manager",
                type="Denial",
                description="Meridian denies the job title pleaded in Paragraph 2.",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="Complaint ¶2",
                source_quote="denies that Plaintiff ever held the title of operations manager",
            ),
            LegalNode(
                id="denial-supervision",
                name="Okafor supervised Meridian employees",
                type="Denial",
                description="Meridian denies the supervisory role pleaded in Paragraph 2.",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="Complaint ¶2",
                source_quote="denies that Plaintiff supervised any Meridian employee",
            ),
            LegalNode(
                id="denial-pay-reduction",
                name="Okafor's compensation was reduced during her employment",
                type="Denial",
                description="Meridian denies the pay reduction pleaded in Paragraph 2.",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="Complaint ¶2",
                source_quote=(
                    "denies that Plaintiff's compensation was reduced at any time during "
                    "her employment"
                ),
            ),
            LegalNode(
                id="denial-residual",
                name="The remaining allegations of Paragraph 2 of the Complaint are true",
                type="Denial",
                description="The catch-all denial closing the partial answer.",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                report_date="2026-04-07",
                responds_to="Complaint ¶2",
                conditions=["Except as expressly admitted herein"],
                source_quote=(
                    "Defendant denies each and every remaining allegation of Paragraph 2 "
                    "of the Complaint"
                ),
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="okafor",
                target_node_id="meridian",
                relationship_name="employed_by",
                description=(
                    "Amara Okafor worked for Meridian Logistics, Inc. as a dispatch coordinator."
                ),
            ),
            KGEdge(
                source_node_id="meridian",
                target_node_id="complaint",
                relationship_name="party_to",
                description="Meridian Logistics, Inc. is the defendant in the Complaint.",
            ),
        ],
    )


def _appraisals_opposing() -> LegalKnowledgeGraph:
    """Opposing appraisals are two authors' opinions, not successive facts."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="adams",
                name="Harriet Adams",
                type="Person",
                description="Plaintiff, the condemnee.",
            ),
            LegalNode(
                id="city-of-clifton",
                name="City of Clifton",
                type="City",
                description="Defendant condemnor in the eminent domain action.",
            ),
            LegalNode(
                id="vance",
                name="Dolores Vance",
                type="Person",
                description="Appraiser, MAI, retained by the plaintiff.",
            ),
            LegalNode(
                id="baptiste",
                name="Terrence Baptiste",
                type="Person",
                description="Appraiser retained by the City of Clifton.",
            ),
            LegalNode(
                id="parcel",
                name="0.62-acre parcel at 118 Harlow Street",
                type="Parcel",
                description="The condemned parcel whose fair market value is in dispute.",
            ),
            LegalNode(
                id="vance-appraisal",
                name="Appraisal of Dolores Vance dated September 8, 2025",
                type="Appraisal",
                description="The plaintiff's appraisal report.",
            ),
            LegalNode(
                id="baptiste-appraisal",
                name="Appraisal of Terrence Baptiste dated October 2, 2025",
                type="Appraisal",
                description="The City of Clifton's appraisal report.",
            ),
            LegalNode(
                id="opinion-vance-value",
                name=(
                    "The fair market value of the 118 Harlow Street parcel as of "
                    "July 1, 2024 was $1,480,000"
                ),
                type="Opinion",
                description=(
                    "Vance's valuation opinion, offered by Adams, derived from three closed "
                    "comparable sales within one mile."
                ),
                statement_type=StatementType.OPINION,
                polarity=Polarity.POSITIVE,
                asserted_by="adams",
                attributed_to="vance",
                applicable_time="2024-07-01",
                report_date="2025-09-08",
                precision=Precision.EXACT,
                scope="the 0.62-acre parcel at 118 Harlow Street as of the date of taking",
                source_quote=(
                    "the fair market value of the parcel as of the July 1, 2024 date of "
                    "taking was $1,480,000"
                ),
            ),
            LegalNode(
                id="opinion-baptiste-value",
                name=(
                    "The fair market value of the 118 Harlow Street parcel as of "
                    "July 1, 2024 was approximately $865,000"
                ),
                type="Opinion",
                description=(
                    "Baptiste's valuation opinion, offered by the City, after a deduction of "
                    "roughly $210,000 for wetlands remediation that Vance did not apply."
                ),
                statement_type=StatementType.OPINION,
                polarity=Polarity.POSITIVE,
                asserted_by="city-of-clifton",
                attributed_to="baptiste",
                applicable_time="2024-07-01",
                report_date="2025-10-02",
                precision=Precision.APPROXIMATE,
                scope=(
                    "the 0.62-acre parcel at 118 Harlow Street as of the date of taking, net "
                    "of a wetlands remediation deduction"
                ),
                source_quote=(
                    "the fair market value of the same parcel as of July 1, 2024 was "
                    "approximately $865,000"
                ),
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="vance",
                target_node_id="vance-appraisal",
                relationship_name="signed",
                description="Dolores Vance authored the appraisal dated September 8, 2025.",
            ),
            KGEdge(
                source_node_id="baptiste",
                target_node_id="baptiste-appraisal",
                relationship_name="signed",
                description="Terrence Baptiste authored the appraisal dated October 2, 2025.",
            ),
            KGEdge(
                source_node_id="vance-appraisal",
                target_node_id="parcel",
                relationship_name="about",
                description=(
                    "The Vance appraisal values the 0.62-acre parcel at 118 Harlow Street."
                ),
            ),
            KGEdge(
                source_node_id="baptiste-appraisal",
                target_node_id="parcel",
                relationship_name="about",
                description=(
                    "The Baptiste appraisal values the 0.62-acre parcel at 118 Harlow Street."
                ),
            ),
        ],
    )


def _lease_amendment() -> LegalKnowledgeGraph:
    """An amendment supersedes a term without erasing it; both periods survive.

    The passage names no speaker for these terms and no pleading party appears in it,
    so the prompt's rule — speaker not named, pleading party absent, leave
    ``asserted_by`` null rather than guess — applies. It is the one fixture here that
    derives no ``asserted_by`` edge.
    """
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="harbor-point",
                name="Harbor Point Holdings LLC",
                type="Company",
                description="Landlord under the Lease.",
            ),
            LegalNode(
                id="brightleaf",
                name="Brightleaf Coffee Company",
                type="Company",
                description="Tenant under the Lease.",
            ),
            LegalNode(
                id="suite-210",
                name="Suite 210, 44 Canal Street",
                type="Suite",
                description="The leased premises.",
            ),
            LegalNode(
                id="lease",
                name="Lease Agreement for Suite 210, 44 Canal Street",
                type="Lease",
                description="The original lease between Harbor Point Holdings and Brightleaf.",
            ),
            LegalNode(
                id="first-amendment",
                name="First Amendment to Lease dated June 14, 2024",
                type="Amendment",
                description="Amendment raising the base rent effective July 1, 2024.",
            ),
            LegalNode(
                id="term-original-rent",
                name=(
                    "Tenant shall pay base rent of $4,000.00 per month for the term "
                    "commencing January 1, 2023 and ending December 31, 2027"
                ),
                type="Term",
                description=(
                    "Section 3.1 of the Lease: monthly base rent payable on the first day "
                    "of each month."
                ),
                statement_type=StatementType.TERM,
                polarity=Polarity.POSITIVE,
                applies_from="2023-01-01",
                applies_to="2027-12-31",
                precision=Precision.EXACT,
                scope="base rent for Suite 210, 44 Canal Street",
                source_quote=(
                    "Tenant shall pay base rent of $4,000.00 per month, payable on the "
                    "first day of each month, for the term commencing January 1, 2023 and "
                    "ending December 31, 2027."
                ),
            ),
            LegalNode(
                id="term-amended-rent",
                name=(
                    "Effective July 1, 2024 Tenant shall pay base rent of $4,400.00 per "
                    "month for the remainder of the term"
                ),
                type="Term",
                description=(
                    "Section 2 of the First Amendment: the raised base rent. It supersedes "
                    "the original rent from July 1, 2024 without erasing it."
                ),
                statement_type=StatementType.TERM,
                polarity=Polarity.POSITIVE,
                applies_from="2024-07-01",
                precision=Precision.EXACT,
                scope="base rent for Suite 210, 44 Canal Street for the remainder of the term",
                source_quote=(
                    "Tenant shall pay base rent of $4,400.00 per month for the remainder "
                    "of the term"
                ),
            ),
            LegalNode(
                id="term-lease-survives",
                name="All terms of the Lease other than the amended base rent remain in effect",
                type="Term",
                description="The First Amendment's savings clause.",
                statement_type=StatementType.TERM,
                polarity=Polarity.POSITIVE,
                conditions=["Except as amended by this First Amendment"],
                source_quote="all terms of the Lease remain in full force and effect",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="harbor-point",
                target_node_id="lease",
                relationship_name="party_to",
                description="Harbor Point Holdings LLC is the landlord under the Lease.",
            ),
            KGEdge(
                source_node_id="brightleaf",
                target_node_id="lease",
                relationship_name="party_to",
                description="Brightleaf Coffee Company is the tenant under the Lease.",
            ),
            KGEdge(
                source_node_id="lease",
                target_node_id="suite-210",
                relationship_name="about",
                description="The Lease demises Suite 210, 44 Canal Street.",
            ),
            KGEdge(
                source_node_id="first-amendment",
                target_node_id="lease",
                relationship_name="about",
                description=(
                    "The First Amendment to Lease dated June 14, 2024 amends the Lease "
                    "Agreement for Suite 210, 44 Canal Street."
                ),
            ),
            KGEdge(
                source_node_id="term-amended-rent",
                target_node_id="term-original-rent",
                relationship_name="supersedes",
                description=(
                    "The $4,400.00 monthly base rent supersedes the $4,000.00 monthly base "
                    "rent from July 1, 2024."
                ),
            ),
        ],
    )


def _deposition_qa() -> LegalKnowledgeGraph:
    """Hedged testimony keeps its hedge: approximate is not exact, unseen is not absent."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="ellerbee",
                name="Raymond Ellerbee",
                type="Person",
                description="Plaintiff and deponent.",
            ),
            LegalNode(
                id="northgate",
                name="Northgate Transit Authority",
                type="GovernmentAgency",
                description="Defendant transit authority.",
            ),
            LegalNode(
                id="deposition",
                name="Deposition of Raymond Ellerbee taken May 22, 2025",
                type="Deposition",
                description="Deposition transcript in Docket No. NT-2025-0442.",
            ),
            LegalNode(
                id="testimony-distance",
                name=(
                    "The bus was roughly forty to fifty feet from the crosswalk when "
                    "Ellerbee first saw it"
                ),
                type="Testimony",
                description=(
                    "Ellerbee's hedged estimate of the distance; he says he is not certain "
                    "and that it was raining hard with his hood up."
                ),
                statement_type=StatementType.TESTIMONY,
                polarity=Polarity.POSITIVE,
                asserted_by="ellerbee",
                report_date="2025-05-22",
                precision=Precision.APPROXIMATE,
                conditions=["I'm not certain"],
                scope="the distance from the crosswalk when the witness first saw the bus",
                source_quote="Maybe forty or fifty feet. I'm not certain.",
            ),
            LegalNode(
                id="testimony-position",
                name=(
                    "Ellerbee was standing on the northeast corner by the mailbox when the "
                    "bus started to move"
                ),
                type="Testimony",
                description="Ellerbee's account of where he was standing.",
                statement_type=StatementType.TESTIMONY,
                polarity=Polarity.POSITIVE,
                asserted_by="ellerbee",
                report_date="2025-05-22",
                precision=Precision.EXACT,
                source_quote="On the northeast corner, by the mailbox.",
            ),
            LegalNode(
                id="testimony-driver-glance",
                name=("Ellerbee saw the driver look to his right before the bus started to move"),
                type="Testimony",
                description=(
                    "Ellerbee says he did not see this — a statement about what the witness "
                    "observed, not about what the driver did: he was watching the crosswalk "
                    "rather than the cab."
                ),
                statement_type=StatementType.TESTIMONY,
                polarity=Polarity.NEGATIVE,
                asserted_by="ellerbee",
                report_date="2025-05-22",
                scope="what the witness observed, not what the driver did",
                source_quote="No. I was watching the crosswalk, not the cab.",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="deposition",
                target_node_id="northgate",
                relationship_name="about",
                description=(
                    "The deposition of Raymond Ellerbee was taken in his action against the "
                    "Northgate Transit Authority."
                ),
            ),
        ],
    )


def _ambiguous_names() -> LegalKnowledgeGraph:
    """Three municipal bodies stay three nodes, and Mr. Smith is not Smith Holdings LLC."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="clifton-planning-board",
                name="Clifton Planning Board",
                type="PlanningBoard",
                description="The board that heard site plan application SP-2025-19.",
            ),
            LegalNode(
                id="clifton-municipal-council",
                name="Clifton Municipal Council",
                type="MunicipalCouncil",
                description="The council that must act on the application within 45 days.",
            ),
            LegalNode(
                id="city-of-clifton",
                name="City of Clifton",
                type="City",
                description="The municipality itself, distinct from its board and council.",
            ),
            LegalNode(
                id="mr-smith",
                name="Mr. Smith",
                type="Person",
                description="Appeared at the meeting on behalf of the applicant.",
            ),
            LegalNode(
                id="smith-holdings",
                name="Smith Holdings LLC",
                type="Company",
                description="The applicant of record, a different entity from Mr. Smith.",
            ),
            LegalNode(
                id="property-300-harlow",
                name="300 Harlow Street",
                type="Property",
                description="Site of the proposed 14-unit building.",
            ),
            LegalNode(
                id="minutes",
                name="Minutes of the Clifton Planning Board meeting of April 9, 2025",
                type="Minutes",
                description="The record of the regular meeting.",
            ),
            LegalNode(
                id="finding-site-plan-recommendation",
                name=(
                    "The Clifton Planning Board voted 5-2 to recommend approval of site "
                    "plan application SP-2025-19"
                ),
                type="Finding",
                description=(
                    "The board's conditional recommendation, tied to the engineer's "
                    "memorandum of March 28, 2025."
                ),
                statement_type=StatementType.FINDING,
                polarity=Polarity.POSITIVE,
                asserted_by="clifton-planning-board",
                report_date="2025-04-09",
                conditions=[
                    "subject to the conditions in the engineer's memorandum of March 28, 2025"
                ],
                scope="site plan application SP-2025-19 for the 14-unit building",
                source_quote="voted 5-2 to recommend approval of site plan application SP-2025-19",
            ),
            LegalNode(
                id="statement-smith-sidewalk",
                name=(
                    "The applicant would fund the full cost of the Harlow Street sidewalk extension"
                ),
                type="Statement",
                description="Stated by Mr. Smith, appearing on behalf of the applicant.",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.POSITIVE,
                asserted_by="mr-smith",
                report_date="2025-04-09",
                source_quote=(
                    "the applicant would fund the full cost of the Harlow Street sidewalk extension"
                ),
            ),
            LegalNode(
                id="record-council-deadline",
                name=(
                    "The Clifton Municipal Council must act on site plan application "
                    "SP-2025-19 within 45 days"
                ),
                type="Record",
                description=(
                    "Recorded by the board: its own recommendation is advisory only and the "
                    "council decides."
                ),
                statement_type=StatementType.RECORD,
                polarity=Polarity.POSITIVE,
                asserted_by="clifton-planning-board",
                report_date="2025-04-09",
                source_quote="the Clifton Municipal Council must act on the application within 45 days",
            ),
            LegalNode(
                id="statement-no-permit",
                name="The City of Clifton has issued a building permit for 300 Harlow Street",
                type="Statement",
                description=(
                    "Recorded in the minutes: the board states that no such permit has issued."
                ),
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.NEGATIVE,
                asserted_by="clifton-planning-board",
                applicable_time="2025-04-09",
                report_date="2025-04-09",
                source_quote=(
                    "The City of Clifton has not issued any building permit for 300 Harlow Street"
                ),
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="smith-holdings",
                target_node_id="property-300-harlow",
                relationship_name="owns",
                description=(
                    "Smith Holdings LLC acquired 300 Harlow Street from the City of Clifton "
                    "in 2022."
                ),
            ),
            KGEdge(
                source_node_id="finding-site-plan-recommendation",
                target_node_id="property-300-harlow",
                relationship_name="about",
                description=(
                    "The Clifton Planning Board's recommendation concerns the 14-unit "
                    "building at 300 Harlow Street."
                ),
            ),
            KGEdge(
                source_node_id="minutes",
                target_node_id="clifton-planning-board",
                relationship_name="about",
                description=(
                    "The April 9, 2025 minutes record the regular meeting of the Clifton "
                    "Planning Board."
                ),
            ),
        ],
    )


def _email_proposal() -> LegalKnowledgeGraph:
    """An indicative offer is a conditional Proposal, never an agreed Term."""
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="raghunathan",
                name="Priya Raghunathan",
                type="Person",
                description="Author of the email, writing for Stonebridge Capital Partners.",
            ),
            LegalNode(
                id="okoye",
                name="Devon Okoye",
                type="Person",
                description="Recipient of the email, at Arcadia Instruments.",
            ),
            LegalNode(
                id="stonebridge",
                name="Stonebridge Capital Partners",
                type="Company",
                description="The prospective buyer.",
            ),
            LegalNode(
                id="arcadia",
                name="Arcadia Instruments, Inc.",
                type="Company",
                description="The target company.",
            ),
            LegalNode(
                id="email",
                name="Email from Priya Raghunathan to Devon Okoye dated February 18, 2025",
                type="Email",
                description="Indicative terms sent after a call.",
            ),
            LegalNode(
                id="proposal-equity-offer",
                name=(
                    "Stonebridge Capital Partners will pay $38,500,000 in cash for 100% of "
                    "the equity of Arcadia Instruments, Inc."
                ),
                type="Proposal",
                description=(
                    "An indicative offer, not an agreed term: it is conditioned on "
                    "diligence, on a definitive agreement, and on the stated working "
                    "capital and liabilities assumptions."
                ),
                statement_type=StatementType.PROPOSAL,
                polarity=Polarity.POSITIVE,
                asserted_by="raghunathan",
                report_date="2025-02-18",
                precision=Precision.EXACT,
                conditions=[
                    "subject to diligence",
                    "to the negotiation of a definitive agreement",
                    "assumes net working capital of at least $2,100,000 at closing",
                    "no undisclosed liabilities",
                ],
                scope="100% of the equity of Arcadia Instruments, Inc.",
                source_quote=(
                    "prepared to offer $38,500,000 in cash for 100% of the equity of "
                    "Arcadia Instruments, Inc."
                ),
            ),
            LegalNode(
                id="statement-nonbinding",
                name="The letter of interest creates a binding obligation on either party",
                type="Statement",
                description=("The email says in terms that it is an expression of interest only."),
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.NEGATIVE,
                asserted_by="raghunathan",
                report_date="2025-02-18",
                source_quote="does not create any binding obligation on either party",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="raghunathan",
                target_node_id="email",
                relationship_name="signed",
                description=("Priya Raghunathan sent the February 18, 2025 email to Devon Okoye."),
            ),
            KGEdge(
                source_node_id="email",
                target_node_id="arcadia",
                relationship_name="about",
                description=(
                    "The February 18, 2025 email sets out indicative terms for acquiring "
                    "Arcadia Instruments, Inc."
                ),
            ),
            KGEdge(
                source_node_id="proposal-equity-offer",
                target_node_id="arcadia",
                relationship_name="about",
                description=(
                    "Stonebridge Capital Partners' offer is for the equity of Arcadia "
                    "Instruments, Inc."
                ),
            ),
        ],
    )


EXPECTED: dict[str, LegalKnowledgeGraph] = {
    "complaint_p17_p18_warning": _complaint_p17_p18_warning(),
    "answer_p17_denial": _answer_p17_denial(),
    "answer_p2_partial": _answer_p2_partial(),
    "appraisals_opposing": _appraisals_opposing(),
    "lease_amendment": _lease_amendment(),
    "deposition_qa": _deposition_qa(),
    "ambiguous_names": _ambiguous_names(),
    "email_proposal": _email_proposal(),
}
