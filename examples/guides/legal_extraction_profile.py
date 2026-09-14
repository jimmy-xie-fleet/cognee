import asyncio

import cognee
from cognee.domains.legal import legal_profile
from cognee.modules.search.types import SearchType

TEXT = (
    "COMPLAINT. Paragraph 12. On March 3, 2025, Plaintiff Jonas Weir alleges that "
    "Northbridge Freight Co. failed to pay overtime wages owed for the pay period "
    "ending February 28, 2025.\n\n"
    "ANSWER. Paragraph 12. Denied. Defendant Northbridge Freight Co. denies each and "
    "every allegation of Paragraph 12 of the Complaint. Further answering, Defendant "
    "states that a payroll audit completed on April 1, 2025 found no unpaid overtime "
    "for the period in question."
)


async def main():
    await cognee.forget(everything=True)

    await cognee.remember(
        TEXT,
        dataset_name="legal_demo",
        self_improvement=False,
        **legal_profile(),
    )

    # To rebuild the graph over data already ingested (e.g. after tuning the profile),
    # run cognify directly instead of remember:
    # await cognee.cognify(datasets=["legal_demo"], **legal_profile())

    results = await cognee.recall(
        "What did the defendant say about the overtime allegation?",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=["legal_demo"],
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
