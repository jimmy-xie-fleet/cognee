"""Import-surface check for the Task 6 module split.

``cognee.tasks.graph.reference_graph_view`` now owns the graph-reading layer that used
to live inline in ``resolve_assertion_references``. This test only checks the seam: the
task module must still expose the same objects (not copies) so existing callers --
including the ``scripts/legal`` reports -- keep working unchanged.
"""

import importlib

# ``cognee/tasks/graph/__init__.py`` does ``from .resolve_assertion_references import
# resolve_assertion_references``, which rebinds the *package's* attribute of that name to
# the function, shadowing the submodule. ``importlib.import_module`` sidesteps that and
# always returns the module object.
reference_graph_view = importlib.import_module("cognee.tasks.graph.reference_graph_view")
resolve_assertion_references = importlib.import_module(
    "cognee.tasks.graph.resolve_assertion_references"
)

_SHARED_NAMES = (
    "GraphView",
    "DocumentTextCache",
    "_load_graph_view",
    "DOCUMENT_NODE_TYPES",
    "VIEW_NODE_TYPES",
    "_read_processed_text",
    "_raw_locations",
    "_text_of",
    "_as_uuid",
    "_node_label",
    "_chunk_index",
)


def test_task_module_reexports_are_the_same_objects_as_the_new_module():
    for name in _SHARED_NAMES:
        moved = getattr(reference_graph_view, name)
        reexported = getattr(resolve_assertion_references, name)
        assert reexported is moved, f"{name} is no longer re-exported from the task module"


def test_resolve_references_report_import_names_still_resolve():
    """Mirrors scripts/legal/resolve_references_report.py:125-132."""
    from cognee.tasks.graph.resolve_assertion_references import (  # noqa: F401
        REFERENCE_FIELDS,
        REFERENCE_RESOLUTION_DATA_ID,
        DocumentTextCache,
        _load_graph_view,
        plan_resolutions,
        write_resolutions,
    )


def test_find_disputes_import_name_still_resolves():
    """Mirrors scripts/legal/find_disputes.py:42."""
    from cognee.tasks.graph.resolve_assertion_references import DOCUMENT_NODE_TYPES  # noqa: F401
