"""Memify pipeline that resolves dangling assertion references across a whole dataset.

The ingest tail (``resolve_assertion_references`` with ``scope="touched"``,
``allow_llm=False``) answers only what it can answer for free: a field that already holds
an id, and a reference that names one entity. Everything else -- including every forward
reference, where the document being pointed at was not in the graph yet -- is left
dangling for this pipeline. It is the sweep that closes them: one detect (extraction)
phase seeds and traces every remaining reference, one apply (enrichment) phase writes the
edges and node patches.

This is where the LLM budget is spent (decision D1), so the pipeline is also where it is
set: ``llm_max_calls`` caps the whole run and ``tracer_max_iter`` caps one reference.
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
    apply_reference_resolutions,
    detect_dangling_references,
)

logger = get_logger("resolve_references_pipeline")


def _require_threshold(name: str, value: Optional[float]) -> None:
    """A confidence threshold is a probability: ``(0, 1]``, and never a bool."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise CogneeValidationError(message=f"{name} must be in the range (0, 1]", log=False)


def _require_count(name: str, value: Optional[int], *, minimum: int) -> None:
    """A call or step count is a plain int at or above ``minimum`` (``True`` is not one)."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CogneeValidationError(message=f"{name} must be an integer >= {minimum}", log=False)


async def resolve_references_pipeline(
    force: bool = False,
    dry_run: bool = False,
    llm_max_calls: Optional[int] = None,
    tracer_max_iter: Optional[int] = None,
    llm_confidence_threshold: Optional[float] = None,
    infer_unstated: Optional[bool] = None,
    infer_confidence_threshold: Optional[float] = None,
    user: Optional[User] = None,
    dataset: str = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
):
    """Resolve every dangling ``responds_to`` / ``attributed_to`` in a dataset's graph.

    Args:
        force: Re-resolve references a previous pass already answered, reading the
            structured reference (or the ``<field>_text``) it preserved.
        dry_run: Plan and log the resolutions without writing anything. Traces still run,
            so the plan shows what the agent would have linked.
        llm_max_calls: Calls this run may spend across every reference it traces.
            ``None`` takes ``REFERENCE_LLM_MAX_CALLS``; ``0`` seeds without spending.
        tracer_max_iter: Steps one reference's trace may take. ``None`` takes
            ``REFERENCE_TRACER_MAX_ITER``.
        llm_confidence_threshold: Below this the agent's answer is recorded but never
            linked. ``None`` takes ``REFERENCE_LLM_CONFIDENCE_THRESHOLD``.
        infer_unstated: Opt into inferring unstated denial/allegation links (decision
            D2). Accepted and forwarded now; the inference pass itself lands with the
            ``llm_inferred`` strategy, so today the option changes nothing.
        infer_confidence_threshold: The higher bar an inferred link is held to. Same
            status as ``infer_unstated``.
        user: Acting user; the default user is used when omitted.
        dataset: Dataset name (or id) whose graph to resolve.
        run_in_background: Forwarded to ``memify``.

    Returns:
        The ``memify`` pipeline result.
    """
    _require_count("llm_max_calls", llm_max_calls, minimum=0)
    _require_count("tracer_max_iter", tracer_max_iter, minimum=1)
    _require_threshold("llm_confidence_threshold", llm_confidence_threshold)
    _require_threshold("infer_confidence_threshold", infer_confidence_threshold)

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
            allow_llm=True,
            force=force,
            llm_max_calls=llm_max_calls,
            tracer_max_iter=tracer_max_iter,
            llm_confidence_threshold=llm_confidence_threshold,
            infer_unstated=infer_unstated,
            infer_confidence_threshold=infer_confidence_threshold,
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
        "resolve_references pipeline finished (dataset=%s, dry_run=%s, llm_max_calls=%s).",
        target.id,
        dry_run,
        "config default" if llm_max_calls is None else llm_max_calls,
    )
    return result
