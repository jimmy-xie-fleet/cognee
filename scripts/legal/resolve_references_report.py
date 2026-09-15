"""Audit and (optionally) apply the deterministic reference resolver over a dataset.

Dry run by default: builds the graph view and document-text cache the same way
``resolve_assertion_references`` does, plans every resolution with
``plan_resolutions``, and prints what would happen -- the summary counters, one line
per resolution the plan proposes, and one line per reference that stayed dangling
(a non-UUID ``responds_to``/``attributed_to`` value the plan did not touch).

``--apply`` writes the plan (edges, then node patches) via ``write_resolutions``.
``--strict`` additionally runs the acceptance gate pinned for ``adams_family_legal``
(see CLAUDE.md's "Legal Extraction Profile" section) and exits 1 if it fails, whether
or not ``--apply`` was also given.

Usage:
    python scripts/legal/resolve_references_report.py adams_family_legal
    python scripts/legal/resolve_references_report.py adams_family_legal --apply --strict
"""

import argparse
import asyncio
import os
from pathlib import Path
from uuid import UUID

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

# Pinned by the live inventory on adams_family_legal (see CLAUDE.md): of the 33
# "Complaint ¶N" references, 31 resolving by document_locator is the accepted floor.
_PARAGRAPH_LOCATOR_FLOOR = 31

# These two references have no document or paragraph anchor in the corpus and must
# never resolve -- if they start resolving, something in the cascade got looser.
_MUST_STAY_UNRESOLVED = ("2014 master plan", "unit c")

_DOCUMENT_STRATEGIES = ("document_locator", "document_only")


def _is_uuid_like(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _is_complaint_paragraph_reference(text: str) -> bool:
    """True for a reference like ``Complaint ¶ 5`` -- the pinned spot-check pattern."""
    return "complaint" in text.lower() and "¶" in text


def _target_label(view, node_id):
    """The name (or a text preview) an edge's target is known by, for the report line."""
    if not node_id:
        return "-"
    props = view.node_props(node_id)
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    text = props.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())[:80]
    return str(node_id)


def _resolution_line(view, resolution) -> str:
    label = _target_label(view, resolution.anchor_id)
    return (
        f'{resolution.field}: "{resolution.reference_text}" → '
        f"{resolution.strategy} {resolution.confidence:.2f} {resolution.anchor_type} {label}"
    )


def _dangling_line(field_name: str, reference_text: str) -> str:
    return f'{field_name}: "{reference_text}" → unresolved 0.00 - -'


def _dangling_entries(view, reference_fields, resolved_keys):
    """Non-UUID reference values the plan left untouched -- what did not resolve.

    ``plan_resolutions`` only enumerates what it resolved; a reference it left
    dangling (unresolved or ambiguous) is recovered here by walking every assertion's
    reference fields and dropping anything already an id (already_resolved, out of
    scope for this report) or already covered by the plan.
    """
    entries = []
    for assertion_id, props in view.assertions.items():
        for field_name in reference_fields:
            value = props.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if _is_uuid_like(value):
                continue
            if (assertion_id, field_name) in resolved_keys:
                continue
            entries.append((field_name, value))
    return entries


def _paragraph_gate(resolutions, dangling, *, floor: int):
    """Rule (a): "Complaint ¶N" references resolved by document_locator, >= floor."""
    resolved_matches = [
        r for r in resolutions if _is_complaint_paragraph_reference(r.reference_text)
    ]
    dangling_matches = [
        text for _field, text in dangling if _is_complaint_paragraph_reference(text)
    ]
    resolved_by_locator = sum(1 for r in resolved_matches if r.strategy == "document_locator")
    total = len(resolved_matches) + len(dangling_matches)
    ok = resolved_by_locator >= floor
    return ok, (
        f"Complaint paragraph references: {resolved_by_locator}/{total} resolved by "
        f"document_locator (floor {floor})"
    )


def _document_checks(view, resolutions, parse_reference, document_profile):
    """One row per document_locator/document_only resolution: doc name + type-word overlap.

    Doubles as the spot-check table and the evidence the dated/spot-check gates score.
    """
    checks = []
    for resolution in resolutions:
        if resolution.strategy not in _DOCUMENT_STRATEGIES:
            continue
        doc_props = view.documents.get(resolution.document_id, {}) if resolution.document_id else {}
        profile = document_profile(resolution.document_id or "", doc_props.get("name") or "")
        parsed = parse_reference(resolution.reference_text)
        overlap = bool(parsed.hint_type_words & profile.type_words)
        checks.append((resolution, doc_props.get("name") or "?", overlap))
    return checks


def _dated_gate(resolutions, dangling, overlap_by_key, has_full_date):
    """Rule (b): a reference with a full date must resolve, by document match, 0 misses."""
    misses = []
    for resolution in resolutions:
        if not has_full_date(resolution.reference_text):
            continue
        if resolution.strategy not in _DOCUMENT_STRATEGIES:
            misses.append(resolution.reference_text)
            continue
        if not overlap_by_key.get((resolution.assertion_id, resolution.field), False):
            misses.append(resolution.reference_text)
    for _field, text in dangling:
        if has_full_date(text):
            misses.append(text)
    ok = not misses
    reason = (
        "Dated references: 0 misses"
        if ok
        else f"Dated references: {len(misses)} miss(es) (expected 0): {misses}"
    )
    return ok, reason


def _spot_check_gate(checks):
    """Rule (c): any document match sharing no type word with its reference is a mismatch."""
    mismatches = [doc_name for _resolution, doc_name, overlap in checks if not overlap]
    ok = not mismatches
    reason = (
        "Spot-check: 0 mismatches"
        if ok
        else f"Spot-check: {len(mismatches)} mismatch(es) (expected 0): {mismatches}"
    )
    return ok, reason


def _unresolved_gate(resolutions, normalize_reference_text):
    """Rule (d): references normalizing to the pinned unresolved forms must stay unresolved."""
    offenders = [
        resolution.reference_text
        for resolution in resolutions
        if normalize_reference_text(resolution.reference_text) in _MUST_STAY_UNRESOLVED
    ]
    ok = not offenders
    reason = (
        "References that must stay unresolved: 0 resolved"
        if ok
        else f"References that must stay unresolved: resolved anyway: {offenders}"
    )
    return ok, reason


def _run_strict_gate(
    view,
    resolutions,
    dangling,
    *,
    floor: int,
    parse_reference,
    document_profile,
    normalize_reference_text,
) -> bool:
    checks = _document_checks(view, resolutions, parse_reference, document_profile)

    print("\nspot-check:")
    for resolution, doc_name, overlap in checks:
        status = "OK" if overlap else "MISMATCH"
        print(f'  "{resolution.reference_text}" → {doc_name} → {status}')

    def has_full_date(text: str) -> bool:
        parsed = parse_reference(text)
        return any(
            date.year is not None and date.month is not None and date.day is not None
            for date in parsed.hint_dates
        )

    overlap_by_key = {
        (resolution.assertion_id, resolution.field): overlap
        for resolution, _doc_name, overlap in checks
    }
    gate_results = [
        _paragraph_gate(resolutions, dangling, floor=floor),
        _dated_gate(resolutions, dangling, overlap_by_key, has_full_date),
        _spot_check_gate(checks),
        _unresolved_gate(resolutions, normalize_reference_text),
    ]

    print("\nstrict gate:")
    ok = True
    for passed, reason in gate_results:
        print(f"  {'PASS' if passed else 'FAIL'}: {reason}")
        ok = ok and passed
    return ok


async def run(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance.write_context import (
        graph_provenance_write_kwargs,
    )
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.graph.utils.reference_resolution import (
        document_profile,
        normalize_reference_text,
        parse_reference,
    )
    from cognee.tasks.graph.resolve_assertion_references import (
        REFERENCE_FIELDS,
        REFERENCE_RESOLUTION_DATA_ID,
        DocumentTextCache,
        _load_graph_view,
        plan_resolutions,
        write_resolutions,
    )

    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        dataset = (
            await session.execute(select(Dataset).where(Dataset.name == args.dataset))
        ).scalar_one_or_none()

    if dataset is None:
        print(f"No such dataset: {args.dataset}")
        return 2

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        view = await _load_graph_view(graph_engine)
        texts = DocumentTextCache(view, dataset_id=dataset.id)
        resolutions, summary = await plan_resolutions(
            view,
            texts,
            force=args.force,
            confidence_floor=args.confidence_floor,
            enable_prose_lookup=args.enable_prose_lookup,
        )

        print(f"dataset={args.dataset} id={dataset.id}")
        print(
            f"scanned={summary['scanned']} already_resolved={summary['already_resolved']} "
            f"resolved={summary['resolved']} unresolved={summary['unresolved']} "
            f"ambiguous={summary['ambiguous']} failed={summary['failed']}"
        )
        print(f"resolved_by_strategy={summary['resolved_by_strategy']}")
        print(f"anchor_types={summary['anchor_types']}")
        if summary.get("notes"):
            print(f"notes={summary['notes']}")

        print("\nplanned resolutions:")
        for resolution in resolutions:
            print(_resolution_line(view, resolution))

        resolved_keys = {(resolution.assertion_id, resolution.field) for resolution in resolutions}
        dangling = _dangling_entries(view, REFERENCE_FIELDS, resolved_keys)
        print("\nunresolved / ambiguous references:")
        for field_name, reference_text in dangling:
            print(_dangling_line(field_name, reference_text))

        if args.apply:
            provenance = await graph_provenance_write_kwargs(
                graph_engine,
                None,
                dataset_id=dataset.id,
                fallback_data_id=REFERENCE_RESOLUTION_DATA_ID,
            )
            write_summary = await write_resolutions(
                graph_engine, view, resolutions, provenance_kwargs=provenance
            )
            print(
                f"\napplied: edges_written={write_summary['edges_written']} "
                f"nodes_patched={write_summary['nodes_patched']}"
            )
            if write_summary.get("notes"):
                print(f"notes={write_summary['notes']}")

        if args.strict:
            ok = _run_strict_gate(
                view,
                resolutions,
                dangling,
                floor=_PARAGRAPH_LOCATOR_FLOOR,
                parse_reference=parse_reference,
                document_profile=document_profile,
                normalize_reference_text=normalize_reference_text,
            )
            return 0 if ok else 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit (and optionally apply) the reference resolver over a dataset."
    )
    parser.add_argument("dataset", help="Dataset name to resolve references over.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the planned resolutions (default: dry run, plan only).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run the pinned acceptance gate and exit 1 if it fails.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-resolve references a previous pass already answered.",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.6,
        help="Resolutions below this confidence are left dangling (default: 0.6).",
    )
    parser.add_argument(
        "--enable-prose-lookup",
        action="store_true",
        help="Opt into the BM25 chunk lookup for locator-less references.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(parsed_args)))
