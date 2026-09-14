"""``remember(session_id=...)`` must reject extraction options, not drop them.

The session branch returns before ``cognify()`` ever runs, and the background bridge
re-cognifies the session with the defaults, so a profile splatted into a session
``remember()`` was silently ignored: the caller believed the legal extraction had built
the graph while the default extraction had. The guard runs before any I/O, so these
tests need nothing patched.
"""

import sys
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.remember.remember import remember
from cognee.domains.legal import legal_profile

remember_module = sys.modules["cognee.api.v1.remember.remember"]

PROFILE_OPTIONS = ("graph_model", "custom_prompt", "config", "chunk_size")


@pytest.mark.asyncio
async def test_session_remember_rejects_the_whole_legal_profile():
    with pytest.raises(ValueError) as raised:
        await remember(
            "Smith denies the payment was late.",
            session_id="s1",
            **legal_profile(),
        )

    message = str(raised.value)
    assert "session_id" in message
    assert "default extraction" in message
    for option in PROFILE_OPTIONS:
        assert option in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option,value",
    [
        ("graph_model", object()),
        ("custom_prompt", "extract legal assertions"),
        ("config", {"ontology_config": {"ontology_resolver": None}}),
        ("chunk_size", 512),
    ],
)
async def test_session_remember_rejects_each_extraction_option(option, value):
    with pytest.raises(ValueError, match="session_id"):
        await remember("Smith denies it.", session_id="s1", **{option: value})


@pytest.mark.asyncio
async def test_session_remember_without_extraction_options_is_untouched():
    with patch.object(remember_module, "_remember_inner", new=AsyncMock()) as inner:
        await remember("Smith denies it.", session_id="s1")

    assert inner.await_count == 1
    assert inner.await_args.kwargs["session_id"] == "s1"


@pytest.mark.asyncio
async def test_permanent_remember_still_accepts_the_profile():
    profile = legal_profile()

    with patch.object(remember_module, "_remember_inner", new=AsyncMock()) as inner:
        await remember("Smith denies it.", **profile)

    assert inner.await_count == 1
    forwarded = inner.await_args.kwargs
    assert forwarded["session_id"] is None
    assert forwarded["chunk_size"] == profile["chunk_size"]
    assert forwarded["custom_prompt"] == profile["custom_prompt"]
    assert forwarded["graph_model"] is profile["graph_model"]
    assert forwarded["config"] is profile["config"]
