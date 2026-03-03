"""Embedded Stepflow orchestrator management.

Starts a local Stepflow orchestrator process pre-configured with the
Langflow worker plugin, for use when LANGFLOW_STEPFLOW_URL is not set.
"""

from __future__ import annotations

import atexit
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stepflow_orchestrator import StepflowOrchestrator

logger = logging.getLogger(__name__)

_embedded_url: str | None = None
_orchestrator: StepflowOrchestrator | None = None


async def start_embedded() -> str:
    """Start an embedded Stepflow orchestrator with the Langflow worker.

    Configures the orchestrator with a ``langflow`` plugin that runs
    ``python -m langflow_stepflow.worker`` as a subprocess component server,
    then starts the orchestrator process.

    On first call the orchestrator is started and its URL cached. Subsequent
    calls return the cached URL immediately. The process is terminated when
    the Python interpreter exits via ``atexit``.

    Returns:
        URL of the running orchestrator (e.g., ``http://127.0.0.1:PORT``)

    Raises:
        RuntimeError: If ``stepflow-orchestrator`` is not installed or the
            orchestrator fails to start.
    """
    global _embedded_url, _orchestrator

    if _embedded_url is not None:
        return _embedded_url

    try:
        from stepflow_orchestrator import OrchestratorConfig, StepflowOrchestrator
    except ImportError as exc:
        raise RuntimeError(
            "The 'stepflow-orchestrator' package is required to run an embedded "
            "Stepflow orchestrator. Install it with: pip install stepflow-orchestrator\n"
            "Alternatively, start a Stepflow orchestrator separately and set "
            "LANGFLOW_STEPFLOW_URL to its URL."
        ) from exc

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
        }
    )

    orch = StepflowOrchestrator(config)
    await orch.__aenter__()
    _orchestrator = orch
    _embedded_url = orch.url

    def _shutdown() -> None:
        import asyncio

        if _orchestrator is not None:
            try:
                asyncio.run(_orchestrator.__aexit__(None, None, None))
            except Exception:
                pass

    atexit.register(_shutdown)
    logger.info("Embedded Stepflow orchestrator started at %s", _embedded_url)
    return _embedded_url
