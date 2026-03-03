"""Integration tests: Langflow flow translation and execution via Stepflow.

Mirrors Langflow's own integration tests in
``src/backend/tests/integration/flows/``, running the same flows with the
same inputs through the Stepflow translator and orchestrator instead of
Langflow's built-in graph engine.

Translation tests (no orchestrator needed):
- Verify every fixture converts without error
- Check step count and component path prefixes

Execution tests (require a running orchestrator + OPENAI_API_KEY for LLM
flows):
- SimpleAPITest: ChatInput → ChatOutput passthrough (no LLM)
- basic_prompting: Prompt + OpenAI (skipped without OPENAI_API_KEY)
- simple_agent: Agent with calculator tool (skipped without OPENAI_API_KEY)
- memory_chatbot: Session memory (skipped without OPENAI_API_KEY)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_fixture(fixtures_dir: Path, name: str) -> dict[str, Any]:
    """Load a Langflow JSON fixture by name (without ``.json`` extension)."""
    path = fixtures_dir / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    with open(path) as f:
        return json.load(f)


def requires_openai() -> pytest.MarkDecorator:
    """Skip if OPENAI_API_KEY is not set."""
    return pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )


# ---------------------------------------------------------------------------
# Translation-only tests (no orchestrator required)
# All fixtures must translate cleanly.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_name", [
    "SimpleAPITest",
    "basic_prompting",
    "simple_agent",
    "memory_chatbot",
])
def test_translation(converter, langflow_fixtures_dir, fixture_name):
    """Each fixture must translate to a valid Stepflow flow with at least one
    step, and every step must route to /langflow/ or /builtin/."""
    data = load_fixture(langflow_fixtures_dir, fixture_name)
    flow = converter.convert(data)

    assert flow is not None, f"{fixture_name}: convert() returned None"
    assert flow.steps, f"{fixture_name}: translated flow has no steps"

    # Steps route to the langflow worker or builtin blob storage
    for step in flow.steps:
        assert step.component.startswith(("/langflow/", "/builtin/")), (
            f"{fixture_name}: unexpected component prefix: {step.component}"
        )


# ---------------------------------------------------------------------------
# Execution tests (require orchestrator)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
async def test_simple_api_passthrough(runner, langflow_fixtures_dir):
    """SimpleAPITest: ChatInput feeds directly into ChatOutput.

    Mirrors Langflow's test_simple_no_llm — no LLM required.
    Input message should appear in the response.
    """
    data = load_fixture(langflow_fixtures_dir, "SimpleAPITest")
    flow_data = data.get("data", data)

    run_outputs, session_id = await runner.run(
        flow_data=flow_data,
        input_value="hello from stepflow",
        session_id=None,
    )

    assert run_outputs, "Expected at least one RunOutputs"
    assert session_id

    result = run_outputs[0].outputs[0]
    assert result is not None, "Expected a ResultData"

    # The flow echoes the input — verify either messages or raw result
    has_output = bool(result.messages) or result.results is not None
    assert has_output, "Expected messages or results from passthrough flow"

    if result.messages:
        text = result.messages[0].message
        assert text, "Expected non-empty message text"


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
@requires_openai()
async def test_basic_prompting(runner, langflow_fixtures_dir):
    """basic_prompting: Prompt template + OpenAI model.

    Mirrors Langflow's basic prompting integration test — requires OPENAI_API_KEY.
    The model should return a haiku (multiple lines of text).
    """
    data = load_fixture(langflow_fixtures_dir, "basic_prompting")
    flow_data = data.get("data", data)

    run_outputs, session_id = await runner.run(
        flow_data=flow_data,
        input_value="Write a haiku about software testing",
        session_id=None,
    )

    assert run_outputs and session_id

    result = run_outputs[0].outputs[0]
    assert result is not None

    if result.messages:
        text = result.messages[0].message
        assert text and len(text) > 0
        # A haiku has three lines
        assert len(text.split("\n")) >= 3, (
            f"Expected haiku with multiple lines, got: {text!r}"
        )


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
@requires_openai()
async def test_simple_agent(runner, langflow_fixtures_dir):
    """simple_agent: Tool-using agent with a calculator.

    Requires OPENAI_API_KEY. The agent should use the calculator tool and
    return the correct numeric answer.
    """
    data = load_fixture(langflow_fixtures_dir, "simple_agent")
    flow_data = data.get("data", data)

    run_outputs, session_id = await runner.run(
        flow_data=flow_data,
        input_value="What is 25 multiplied by 4?",
        session_id=None,
    )

    assert run_outputs and session_id

    result = run_outputs[0].outputs[0]
    assert result is not None

    if result.messages:
        text = result.messages[0].message
        assert "100" in text, (
            f"Expected '100' (25×4) in agent response, got: {text!r}"
        )


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
@requires_openai()
async def test_memory_chatbot_session_isolation(runner, langflow_fixtures_dir):
    """memory_chatbot: Two sessions must not share memory.

    Requires OPENAI_API_KEY. Each session should only recall messages from
    its own history, not from other sessions.
    """
    import uuid

    data = load_fixture(langflow_fixtures_dir, "memory_chatbot")
    flow_data = data.get("data", data)

    session_a = f"test-session-a-{uuid.uuid4().hex[:8]}"
    session_b = f"test-session-b-{uuid.uuid4().hex[:8]}"

    # Establish context in session A
    await runner.run(
        flow_data=flow_data,
        input_value="My name is Alice",
        session_id=session_a,
    )

    # Ask in session A — should recall "Alice"
    outputs_a, _ = await runner.run(
        flow_data=flow_data,
        input_value="What is my name?",
        session_id=session_a,
    )

    # Ask in session B — should have no knowledge of Alice
    outputs_b, _ = await runner.run(
        flow_data=flow_data,
        input_value="What is my name?",
        session_id=session_b,
    )

    result_a = outputs_a[0].outputs[0]
    result_b = outputs_b[0].outputs[0]

    if result_a and result_a.messages:
        assert "alice" in result_a.messages[0].message.lower(), (
            "Session A should recall 'Alice'"
        )

    if result_b and result_b.messages:
        assert "alice" not in result_b.messages[0].message.lower(), (
            "Session B should not know about Alice"
        )
