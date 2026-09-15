"""Import-surface check for the Task 10 module split.

``cognee.tasks.graph.reference_pass`` now owns the agentic trace pass that used to live
inline in ``resolve_assertion_references``. This test only checks the seam: the task
module must still expose the same objects (not copies), so every existing caller -- the
``scripts/legal`` reports, the memify pipeline, and the tests that reach for a private
name -- keeps working unchanged.
"""

import importlib

# ``cognee/tasks/graph/__init__.py`` rebinds the package attribute of that name to the
# task function, shadowing the submodule, so the module object comes from importlib.
reference_pass = importlib.import_module("cognee.tasks.graph.reference_pass")
resolve_assertion_references = importlib.import_module(
    "cognee.tasks.graph.resolve_assertion_references"
)

_SHARED_NAMES = (
    "CIRCUIT_BREAKER_FAILURES",
    "NOTE_EDGES_EXIST",
    "NOTE_FORCE_KEPT_PRIOR",
    "NOTE_LLM_ABSTAINED",
    "NOTE_LLM_BELOW_THRESHOLD",
    "NOTE_LLM_BUDGET_EXHAUSTED",
    "NOTE_LLM_CIRCUIT_BROKEN",
    "NOTE_LLM_ITERATION_CAP",
    "NOTE_LLM_MALFORMED_STEP",
    "NOTE_LLM_SELF_REFERENCE",
    "NOTE_LLM_UNKNOWN_LABEL",
    "NOTE_UNSTATED",
    "PATCH_FULL",
    "PATCH_NONE",
    "PATCH_RESOLUTION_ONLY",
    "SEED_DOCUMENT_LIMIT",
    "TRACE_ARGS_MAX_CHARS",
    "_Outcome",
    "_Pending",
    "_TraceAnswer",
    "_answer_to_outcome",
    "_bump",
    "_build_answer",
    "_combine_seed",
    "_count",
    "_default_patch_mode",
    "_document_name",
    "_edge_precheck",
    "_edge_precheck_outcome",
    "_finalize",
    "_narrow_to_quoted_assertions",
    "_negative_record",
    "_order_key",
    "_prior_attempt",
    "_record_outcome",
    "_seed_reference",
    "_trace_dicts",
    "_trace_pending",
)


def test_task_module_reexports_are_the_same_objects_as_the_new_module():
    for name in _SHARED_NAMES:
        moved = getattr(reference_pass, name)
        reexported = getattr(resolve_assertion_references, name)
        assert reexported is moved, f"{name} is no longer re-exported from the task module"


def test_the_pass_module_never_imports_the_task_module_back():
    """The split only works one way round: a cycle here would break every importer."""
    source = importlib.util.find_spec("cognee.tasks.graph.reference_pass").origin
    with open(source, encoding="utf-8") as handle:
        assert "resolve_assertion_references import" not in handle.read()
