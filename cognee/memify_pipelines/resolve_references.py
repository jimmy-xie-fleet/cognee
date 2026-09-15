"""Memify pipeline that resolves dangling assertion references across a whole dataset.

The ingest tail (``resolve_assertion_references`` with ``scope="touched"``) can only see
the documents already in the graph when a document is cognified, so a pair ingested
concurrently leaves both references dangling. This pipeline is the sweep that closes them:
one detect (extraction) phase plans every resolution in the graph, one apply (enrichment)
phase writes the edges and node patches.
"""

from typing import Optional

from cognee import memify
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.resolve_assertion_references import (
    DEFAULT_CONFIDENCE_FLOOR,
    apply_reference_resolutions,
    detect_dangling_references,
)

logger = get_logger("resolve_references_pipeline")


async def resolve_references_pipeline(
    force: bool = False,
    dry_run: bool = False,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    enable_prose_lookup: bool = False,
    user: Optional[User] = None,
    dataset: str = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
):
    """Resolve every dangling ``responds_to`` / ``attributed_to`` in a dataset's graph.

    Args:
        force: Re-resolve references a previous pass already answered, reading the
            original wording back out of ``<field>_text``.
        dry_run: Plan and log the resolutions without writing anything.
        confidence_floor: Resolutions scoring below this are left dangling.
        enable_prose_lookup: Opt into the BM25 chunk lookup for locator-less references.
        user: Acting user; the default user is used when omitted.
        dataset: Dataset name (or id) whose graph to resolve.
        run_in_background: Forwarded to ``memify``.

    Returns:
        The ``memify`` pipeline result.
    """
    if (
        isinstance(confidence_floor, bool)
        or not isinstance(confidence_floor, (int, float))
        or not 0 < confidence_floor <= 1
    ):
        raise CogneeValidationError(
            message="confidence_floor must be in the range (0, 1]", log=False
        )

    if user is None:
        user = await get_default_user()

    datasets = await get_authorized_existing_datasets([dataset], "write", user)
    if not datasets:
        raise CogneeValidationError(
            message=f"User (id: {user.id}) has no write access to dataset: {dataset}",
            log=False,
        )
    target = datasets[0]

    extraction_tasks = [
        Task(
            detect_dangling_references,
            scope="all",
            force=force,
            confidence_floor=confidence_floor,
            enable_prose_lookup=enable_prose_lookup,
        )
    ]
    enrichment_tasks = [Task(apply_reference_resolutions, dry_run=dry_run)]

    # No set_database_global_context_variables scope around memify: the pipeline enters it
    # itself under the dataset lock. Holding the scope's queue slot while memify waits on
    # that lock inverts the canonical order (dataset lock -> queue slot) and can deadlock
    # the process (SDK-483).
    result = await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # placeholder seed; the tasks read the assertions from the graph
        dataset=target.id,
        user=user,
        run_in_background=run_in_background,
    )

    logger.info(
        "resolve_references pipeline finished (dataset=%s, dry_run=%s, floor=%.2f).",
        target.id,
        dry_run,
        confidence_floor,
    )
    return result
