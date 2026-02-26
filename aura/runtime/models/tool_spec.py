from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .spec_common import SpecBase, _clean_non_empty_str, _dedupe_str_list


class ToolKind(StrEnum):
    LOCAL = "local"
    MCP_PROXY = "mcp_proxy"
    COMPOSITE = "composite"


class ToolEntrypointType(StrEnum):
    PYTHON_CALLABLE = "python_callable"
    SHELL = "shell"
    MCP = "mcp"
    DAG = "dag"
    UNKNOWN = "unknown"


class SideEffectLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class ToolEntrypoint(BaseModel):
    type: ToolEntrypointType = ToolEntrypointType.UNKNOWN
    ref: str | None = None

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="entrypoint.ref")


class ToolEffectProfile(BaseModel):
    side_effect_level: SideEffectLevel = SideEffectLevel.NONE
    idempotent: bool = True


class ToolRetryPolicy(BaseModel):
    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff: str | None = None

    @field_validator("backoff")
    @classmethod
    def _validate_backoff(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="runtime.retry.backoff")


class ToolRuntimePolicy(BaseModel):
    timeout_sec: int = Field(default=30, ge=1, le=3600)
    retry: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)


class ToolAccessPolicy(BaseModel):
    approval_required: bool = False
    approval_level: str | None = None
    scope: list[str] = Field(default_factory=list)

    @field_validator("approval_level")
    @classmethod
    def _validate_approval_level(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="policy.approval_level")

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)


class McpToolBinding(BaseModel):
    server_id: str
    remote_tool: str

    @field_validator("server_id", "remote_tool")
    @classmethod
    def _validate_required(cls, v: str, info) -> str:
        return _clean_non_empty_str(v, field_name=str(info.field_name))


class ToolSpec(SpecBase):
    """
    Declarative tool contract used by the agent-market registry.
    """

    kind: ToolKind = ToolKind.LOCAL
    runtime_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    entrypoint: ToolEntrypoint = Field(default_factory=ToolEntrypoint)
    params_schema_ref: str | None = None
    params_schema: dict[str, Any] | None = None
    result_schema_ref: str | None = None
    result_schema: dict[str, Any] | None = None

    effects: ToolEffectProfile = Field(default_factory=ToolEffectProfile)
    runtime: ToolRuntimePolicy = Field(default_factory=ToolRuntimePolicy)
    policy: ToolAccessPolicy = Field(default_factory=ToolAccessPolicy)
    mcp_binding: McpToolBinding | None = None

    @field_validator("runtime_name", "description", "params_schema_ref", "result_schema_ref")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

    @model_validator(mode="after")
    def _validate_kind_and_binding(self) -> "ToolSpec":
        if self.kind is ToolKind.MCP_PROXY and self.mcp_binding is None:
            raise ValueError("mcp_proxy tools must define mcp_binding.")
        if self.entrypoint.type is ToolEntrypointType.MCP and self.mcp_binding is None:
            raise ValueError("MCP entrypoint requires mcp_binding.")
        return self

