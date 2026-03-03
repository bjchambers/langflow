"""Pytest configuration for integration tests.

Integration tests use module-scoped async fixtures to share the Stepflow
orchestrator across tests for performance. They require:

- ``OPENAI_API_KEY``: for tests that call the OpenAI API (skipped otherwise)
- ``LANGFLOW_STEPFLOW_FIXTURES`` (optional): path to a directory of Langflow
  fixture JSON files. Defaults to the sibling ``stepflow`` repo at
  ``../../../../../../stepflow/integrations/langflow/tests/fixtures/langflow``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from langflow_stepflow.translation.translator import LangflowConverter


def _find_fixtures_dir() -> Path:
    """Locate the Langflow fixture JSON directory.

    Resolution order:
    1. ``LANGFLOW_STEPFLOW_FIXTURES`` env var
    2. ``tests/fixtures/langflow/`` bundled alongside this package (preferred)
    3. Sibling ``stepflow`` repo at ``../../../../../../../stepflow/…``
    """
    # Env var takes precedence
    if env_dir := os.environ.get("LANGFLOW_STEPFLOW_FIXTURES"):
        p = Path(env_dir)
        if p.exists():
            return p

    # Local bundled fixtures (this file is at tests/integration/conftest.py)
    here = Path(__file__).resolve()
    local = here.parents[1] / "fixtures" / "langflow"
    if local.exists():
        return local

    # Fallback: sibling stepflow repo at the same level as the langflow repo.
    langflow_repo = here.parents[4]
    candidate = (
        langflow_repo.parent
        / "stepflow"
        / "integrations"
        / "langflow"
        / "tests"
        / "fixtures"
        / "langflow"
    )
    if candidate.exists():
        return candidate

    pytest.skip(
        "Langflow fixture directory not found. "
        "Set LANGFLOW_STEPFLOW_FIXTURES to the path of fixture JSON files, "
        "or ensure tests/fixtures/langflow/ exists."
    )


@pytest.fixture(scope="module")
def langflow_fixtures_dir() -> Path:
    """Path to the directory containing Langflow fixture JSON files."""
    return _find_fixtures_dir()


@pytest.fixture(scope="module")
def converter() -> LangflowConverter:
    """Shared LangflowConverter instance."""
    return LangflowConverter()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def stepflow_orchestrator():
    """Start a local Stepflow orchestrator with the Langflow worker plugin."""
    from stepflow_orchestrator import OrchestratorConfig, StepflowOrchestrator

    config = OrchestratorConfig(
        config={
            "plugins": {
                "builtin": {"type": "builtin"},
                "langflow": {
                    "type": "stepflow",
                    "command": sys.executable,
                    "args": ["-m", "langflow_stepflow.worker"],
                },
            },
            "routes": {
                "/langflow/{*component}": [{"plugin": "langflow"}],
                "/builtin/{*component}": [{"plugin": "builtin"}],
            },
        },
        startup_timeout=60.0,
    )

    async with StepflowOrchestrator.start(config) as orch:
        yield orch


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def runner(stepflow_orchestrator):
    """StepflowRunner connected to the module-scoped orchestrator."""
    from langflow_stepflow.executor.runner import StepflowRunner

    return StepflowRunner(stepflow_orchestrator.url)
