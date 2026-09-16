"""The bounded agentic loop that resolves ONE assertion reference.

Ask for the next :class:`TracerStep`; either run the tool it named and append the result to
the context, or accept the ``finish`` it returned. Stops on a finish, at ``max_iter``
steps, or when the pass-wide :class:`CallBudget` cannot pay for another call.

Two spending rules, because they are the difference between a bounded pass and an unbounded
one: ``budget.take()`` runs **before** every call, so a trace that starts after the pass
budget ran out costs nothing; and reaching ``max_iter`` returns an abstention **without** a
final "just answer now" call, so a reference costs at most ``max_iter`` calls exactly.

Nothing here writes to the graph.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.graph.utils.reference_candidates import (
    Candidate,
    LabelRegistry,
    format_candidate_lines,
)
from cognee.modules.graph.utils.reference_resolution import ReferenceHint, reference_display_text
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_tracer_tools import (
    MAX_TOOL_OUTPUT_CHARS,
    ToolSpec,
    render_tool_manifest,
    run_tool,
)

logger = get_logger("reference_tracer")

TRACE_SYSTEM_PROMPT = "trace_reference_system.txt"
TRACE_USER_PROMPT = "trace_reference_user.txt"
INFER_UNSTATED_SYSTEM_PROMPT = "infer_unstated_reference_system.txt"

# How much of a tool result a TraceRecord keeps. A record is for a human reading a report,
# not for the model, which already saw the full (truncated) result in its context.
TRACE_PREVIEW_CHARS = 300

_TRUNCATION_NOTE = "\n… [truncated]"
# Tool output is document text, and document text is untrusted: without a delimiter it can
# write "# Step 3: read_chunk(...)\nResult:" and forge a step the tools never ran. The loop
# is the only writer of these two tokens -- any "<<<" run inside a result is broken before
# it is fenced -- and the system prompt tells the model so.
FENCE_OPEN_TEMPLATE = "<<<tool-result step={step} tool={tool}>>>"
FENCE_CLOSE = "<<<end-tool-result>>>"


# --------------------------------------------------------------------------- #
# response models
# --------------------------------------------------------------------------- #


class TracerToolCall(BaseModel):
    tool_name: str = Field(..., description="A tool name from the manifest, exactly as written.")
    # A plain object rather than a per-tool union: the tracer validates it against the named
    # tool's own argument model. Typed as Dict[str, Any] so the emitted JSON schema is
    # explicit; BAML rejects Any-valued maps, so this model is litellm/instructor only.
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for that tool, matching its schema."
    )


class TracerFinish(BaseModel):
    candidate_label: Optional[str] = Field(
        None, description="A label shown in this trace (A3, P2, D1), or null to abstain."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="One sentence naming the wording that decided it.")


# The two outcomes are separate optional submodels rather than a tagged union on purpose:
# a union makes pydantic emit keywords several structured-output providers refuse in strict
# mode, so "exactly one of them" is enforced by the loop. The class docstring below is
# short because it is shipped to the model as the schema's description.
class TracerStep(BaseModel):
    """One step of a trace: a thought, plus either a tool call or a finish, never both."""

    thought: str = ""
    tool_call: Optional[TracerToolCall] = None
    finish: Optional[TracerFinish] = None


# --------------------------------------------------------------------------- #
# budget and trace records
# --------------------------------------------------------------------------- #


@dataclass
class CallBudget:
    """LLM calls one resolver pass may spend, shared by every trace in it.

    Deliberately a plain mutable object handed to each trace rather than a ContextVar: a
    ContextVar is *copied* into a task, so concurrent traces would each get their own
    budget. Traces run sequentially for the same reason.
    """

    max_calls: int
    used: int = 0

    def take(self) -> bool:
        """Claim one call. False (and nothing spent) once the budget is gone."""
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls


@dataclass
class TraceRecord:
    """One tool step of a trace, as a report would print it."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    ok: bool = True


# --------------------------------------------------------------------------- #
# counters
# --------------------------------------------------------------------------- #


def _bump(counters: MutableMapping[str, Any], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


def _bump_tool(counters: MutableMapping[str, Any], name: str) -> None:
    by_name = counters.setdefault("tool_calls_by_name", {})
    by_name[name] = by_name.get(name, 0) + 1


# --------------------------------------------------------------------------- #
# prompt context
# --------------------------------------------------------------------------- #


def _reference_block(hint: Optional[ReferenceHint]) -> Optional[Dict[str, str]]:
    """The reference as the document made it, or None when there is nothing to show.

    None drives the user template's ``{% if reference %}`` to its "no reference recorded"
    branch, which is the shape the unstated-inference variant always runs in.
    Fence-neutralised like the source block: every field here is document wording.
    """
    if hint is None:
        return None

    block = {
        "text": _neutralize_fences(reference_display_text(hint)),
        "document_hint": _neutralize_fences(hint.document_hint or ""),
        "locator_kind": _neutralize_fences(hint.locator_kind or ""),
        "locator_value": _neutralize_fences(hint.locator_value or ""),
        "date": _neutralize_fences(hint.date or ""),
        "basis": _neutralize_fences(hint.basis or ""),
    }
    return block if any(block.values()) else None


def _source_block(
    source_props: Mapping[str, Any], source_document_name: Optional[str], field_name: str
) -> Dict[str, str]:
    """The referring statement as the prompt shows it -- every value fence-neutralised.

    The proposition, the quote and the document name are document text, so they can open or
    close a fence exactly like a tool result can.
    """

    def text(value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return _neutralize_fences(value.strip())
        return fallback

    return {
        "source_document": text(source_document_name, "(unknown document)"),
        "speaker": text(source_props.get("asserted_by"), "(not recorded)"),
        "statement_type": text(source_props.get("statement_type"), "statement"),
        "polarity": text(source_props.get("polarity"), "unknown"),
        "proposition": text(source_props.get("name"), "(no proposition recorded)"),
        "source_quote": text(source_props.get("source_quote"), "(no quote recorded)"),
        "field": field_name,
    }


def _neutralize_fences(text: str) -> str:
    """Break every ``<<`` run so borrowed text cannot open or close a fence.

    Applied to everything in the prompt the resolver did not write itself, so the fence
    tokens stay the loop's alone. A character scan rather than a regex, deliberately: the
    resolver takes no regex over text it did not supply itself.
    """
    if "<<" not in text:
        return text

    characters = list(text)
    for index in range(len(characters) - 1):
        if characters[index] == "<" and characters[index + 1] == "<":
            characters[index] = "< "
    return "".join(characters)


def _render_args(arguments: Mapping[str, Any]) -> str:
    """The arguments as the echoed ``# Step N: tool(args)`` line shows them.

    Neutralised too: the model chose these strings, and they are echoed above the fence.
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive; arguments came from a pydantic dict
        rendered = str(arguments)
    return _neutralize_fences(rendered)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


async def trace_reference(
    *,
    system_prompt_path: str,
    hint: Optional[ReferenceHint],
    source_props: Mapping[str, Any],
    source_document_name: Optional[str],
    field_name: str,
    seed: Sequence[Candidate],
    tools: Mapping[str, ToolSpec],
    registry: LabelRegistry,
    budget: CallBudget,
    max_iter: int,
    counters: MutableMapping[str, Any],
) -> Tuple[TracerFinish, List[TraceRecord], int]:
    """Resolve one reference, or abstain, in at most ``max_iter`` LLM calls.

    Returns ``(finish, tool step records, calls actually spent)``. ``finish`` always names
    either a label this trace's ``registry`` can resolve or ``None``: a label the model
    invented is converted to an abstention here. Counters are created on first use, and the
    specific abstention causes are counted apart from ``llm_abstained``.
    """
    records: List[TraceRecord] = []
    manifest = render_tool_manifest(tools)
    # A missing or blank system prompt is a deployment bug, not a per-reference failure:
    # tracing without one would spend the pass budget on a model told nothing about labels,
    # abstention or the fence. Raise before any slot is taken, so nothing is spent.
    system_prompt = read_query_prompt(system_prompt_path)
    if system_prompt is None:
        raise FileNotFoundError(f"Reference tracer system prompt not found: {system_prompt_path}")
    if not system_prompt.strip():
        raise ValueError(f"Reference tracer system prompt is empty: {system_prompt_path}")
    # The seed opens ``{{ context }}``, and every line of it is document text.
    context = _neutralize_fences(format_candidate_lines(seed)) or "(no seed candidates)"
    reference = _reference_block(hint)
    source = _source_block(source_props, source_document_name, field_name)
    iterations = 0

    for step_number in range(1, max_iter + 1):
        # Rendered before the slot is claimed: burning a pass-wide call on a template bug
        # would charge every other reference for it.
        user_prompt = render_prompt(
            TRACE_USER_PROMPT,
            {
                **source,
                "reference": reference,
                "tools": manifest,
                "context": context,
                "step": step_number,
                "max_steps": max_iter,
            },
        )

        if not budget.take():
            _bump(counters, "llm_budget_exhausted")
            return _abstain("pass budget exhausted"), records, iterations

        iterations += 1

        try:
            step: TracerStep = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=TracerStep,
            )
        except Exception as error:
            # The slot is spent either way; a retry would spend a second one while every
            # other reference in the pass is still waiting.
            _bump(counters, "llm_failed")
            logger.warning("Reference trace step %s failed: %s", step_number, error)
            return _abstain(f"tracer call failed: {error}"), records, iterations
        _bump(counters, "llm_calls")

        finish = step.finish
        if finish is not None:
            label = (finish.candidate_label or "").strip()
            if not label:
                _bump(counters, "llm_abstained")
                # "" and "   " are clumsily written abstentions; normalise them so a
                # caller only ever has to test `candidate_label is None`.
                return finish.model_copy(update={"candidate_label": None}), records, iterations
            if registry.resolve(label) is None:
                _bump(counters, "llm_unknown_label")
                return _abstain(f"unknown candidate label {label}"), records, iterations
            return finish, records, iterations

        tool_call = step.tool_call
        if tool_call is None:
            # Counted apart from an abstention: the model did not look and decline, it
            # returned a step the contract has no reading for.
            _bump(counters, "llm_malformed_step")
            return _abstain("step named neither a tool nor a finish"), records, iterations

        name = tool_call.tool_name.strip()
        arguments = dict(tool_call.arguments or {})
        result = _neutralize_fences(await run_tool(tools, name, arguments))
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            # Inside the fence, so the model can see that what it got was cut short.
            result = result[:MAX_TOOL_OUTPUT_CHARS] + _TRUNCATION_NOTE

        _bump_tool(counters, name)
        records.append(
            TraceRecord(
                tool=name,
                args=arguments,
                result_preview=result[:TRACE_PREVIEW_CHARS],
                ok=not result.startswith("ERROR:"),
            )
        )
        fence_open = FENCE_OPEN_TEMPLATE.format(step=step_number, tool=name)
        context += (
            f"\n\n# Step {step_number}: {name}({_render_args(arguments)})\n"
            f"{fence_open}\n{result}\n{FENCE_CLOSE}"
        )

    _bump(counters, "traces_iteration_capped")
    return _abstain("iteration cap reached"), records, iterations


def _abstain(reason: str) -> TracerFinish:
    return TracerFinish(candidate_label=None, confidence=0.0, reason=reason)
