"""Stepflow flow runner for Langflow integration.

Translates Langflow flows to Stepflow format, submits them to the
Stepflow orchestrator, and converts the results back to Langflow's
RunOutputs format.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from lfx.graph.schema import ResultData, RunOutputs
from lfx.utils.schemas import ChatOutputResponse

from langflow_stepflow.translation.translator import LangflowConverter

logger = logging.getLogger(__name__)

_runner_cache: dict[str, StepflowRunner] = {}


async def get_runner(stepflow_url: str | None) -> StepflowRunner:
    """Return a cached StepflowRunner for the given URL.

    If ``stepflow_url`` is None, starts an embedded Stepflow orchestrator
    on first call and caches its URL for subsequent calls.

    Args:
        stepflow_url: URL of an external Stepflow orchestrator, or None
            to use an embedded one.

    Returns:
        A ``StepflowRunner`` connected to the orchestrator.
    """
    if stepflow_url is None:
        from .embedded import start_embedded

        stepflow_url = await start_embedded()

    if stepflow_url not in _runner_cache:
        _runner_cache[stepflow_url] = StepflowRunner(stepflow_url)

    return _runner_cache[stepflow_url]


class StepflowRunner:
    """Executes Langflow flows via a Stepflow orchestrator.

    Translates the (already-tweaked) Langflow flow JSON to a Stepflow flow
    definition, stores it via the flows API, submits a synchronous run, and
    converts the result back into Langflow's ``RunOutputs`` format.
    """

    def __init__(self, stepflow_url: str) -> None:
        self.url = stepflow_url.rstrip("/")
        self._converter = LangflowConverter()

    async def run(
        self,
        flow_data: dict[str, Any],
        input_value: str | None,
        session_id: str | None,
    ) -> tuple[list[RunOutputs], str]:
        """Execute a Langflow flow via Stepflow.

        Args:
            flow_data: The ``flow.data`` dict (nodes + edges), already tweaked
                by Langflow's ``process_tweaks()``.
            input_value: The user's input string (e.g., the chat message).
            session_id: Optional session ID; one is generated if not provided.

        Returns:
            Tuple of ``(list[RunOutputs], session_id)``, matching the return
            type of ``run_graph_internal()``.
        """
        # Wrap in the envelope that LangflowConverter.convert() expects
        flow = self._converter.convert({"data": flow_data})
        # exclude_none: Rust serde rejects explicit null for optional fields;
        # strip them until stepflow-ai/stepflow#707 is resolved.
        flow_dict = flow.model_dump(by_alias=True, exclude_unset=True, exclude_none=True)

        async with httpx.AsyncClient(base_url=self.url, timeout=300.0) as client:
            # Store the flow definition via the flows API
            store_resp = await client.post("/api/v1/flows", json={"flow": flow_dict})
            store_resp.raise_for_status()
            flow_id = store_resp.json()["flowId"]

            # Langflow flows accept {"message": <input>} for chat-style input
            run_input = [{"message": input_value}] if input_value is not None else [{}]

            run_resp = await client.post(
                "/api/v1/runs",
                json={"flowId": flow_id, "input": run_input, "wait": True},
            )
            run_resp.raise_for_status()
            run_data = run_resp.json()

        session_id = session_id or str(uuid.uuid4())
        return self._to_run_outputs(run_data, input_value, session_id), session_id

    def _to_run_outputs(
        self,
        run_data: dict[str, Any],
        input_value: str | None,
        session_id: str,
    ) -> list[RunOutputs]:
        """Convert a Stepflow run result to a list of Langflow RunOutputs."""
        result_value: Any = None
        if run_data.get("results"):
            item = run_data["results"][0]
            outcome = item.get("result", {})
            if outcome.get("outcome") == "success":
                result_value = outcome.get("result")

        messages: list[ChatOutputResponse] = []
        text = self._extract_text(result_value)
        if text:
            messages.append(
                ChatOutputResponse(
                    message=text,
                    sender="Machine",
                    sender_name="AI",
                    session_id=session_id,
                    type="text",
                )
            )

        return [
            RunOutputs(
                inputs={"input_value": input_value or ""},
                outputs=[
                    ResultData(
                        results=result_value,
                        messages=messages,
                    )
                ],
            )
        ]

    def _extract_text(self, result: Any) -> str | None:
        """Extract a plain-text string from a Stepflow result value."""
        if result is None:
            return None
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if result.get("__langflow_type__") == "Message":
                return result.get("text")
            if "text" in result:
                return result["text"]
        return None
