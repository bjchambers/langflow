"""Test utilities for building tweaks with environment variable support."""

import os
from typing import Any


class TweaksBuilder:
    """Builder utility for creating tweaks with environment variable support."""

    def __init__(self):
        self.tweaks: dict[str, dict[str, Any]] = {}
        self.missing_env_vars: list[str] = []

    def add_tweak(self, component_id: str, field_name: str, value: Any) -> "TweaksBuilder":
        if component_id not in self.tweaks:
            self.tweaks[component_id] = {}
        self.tweaks[component_id][field_name] = value
        return self

    def add_env_tweak(self, component_id: str, field_name: str, env_var: str) -> "TweaksBuilder":
        env_value = os.environ.get(env_var)
        if env_value is not None:
            self.add_tweak(component_id, field_name, env_value)
        else:
            self.missing_env_vars.append(env_var)
        return self

    def add_astradb_tweaks(
        self,
        component_id: str,
        endpoint_env: str = "ASTRA_DB_API_ENDPOINT",
        **kwargs: Any,
    ) -> "TweaksBuilder":
        self.add_env_tweak(component_id, "api_endpoint", endpoint_env)
        if "database_name" not in kwargs:
            kwargs["database_name"] = "langflow-test"
        if "collection_name" not in kwargs:
            kwargs["collection_name"] = "test_collection"
        for field_name, value in kwargs.items():
            self.add_tweak(component_id, field_name, value)
        return self

    def build(self) -> dict[str, dict[str, Any]]:
        if self.missing_env_vars:
            missing_vars = ", ".join(self.missing_env_vars)
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        return dict(self.tweaks)

    def build_or_skip(self) -> dict[str, dict[str, Any]]:
        if self.missing_env_vars:
            import pytest
            missing_vars = ", ".join(self.missing_env_vars)
            pytest.skip(f"Missing required environment variables: {missing_vars}")
        return dict(self.tweaks)
