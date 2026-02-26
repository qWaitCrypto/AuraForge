from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from .spec_common import SpecBase, _clean_non_empty_str, _dedupe_str_list


class AgentExecutionMode(StrEnum):
    SUBAGENT_PRESET = "subagent_preset"
    NATIVE = "native"


class AgentRoutingPolicy(BaseModel):
    priority: int = Field(default=5, ge=0, le=10)
    max_concurrency: int = Field(default=1, ge=1, le=64)
    selection_weight: float = Field(default=1.0, gt=0, le=100)
    fallback_agent_ids: list[str] = Field(default_factory=list)

    @field_validator("fallback_agent_ids")
    @classmethod
    def _validate_fallback_ids(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)


class AgentRiskPolicy(BaseModel):
    approval_level: str | None = None
    risk_level: str | None = None

    @field_validator("approval_level", "risk_level")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))


class AgentCostProfile(BaseModel):
    model_tier: str | None = None
    token_budget: int | None = Field(default=None, ge=1)
    cost_hint_usd: float | None = Field(default=None, ge=0)

    @field_validator("model_tier")
    @classmethod
    def _validate_model_tier(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="cost.model_tier")


class AgentExecutionSpec(BaseModel):
    mode: AgentExecutionMode = AgentExecutionMode.NATIVE
    preset_name: str | None = None
    default_allowlist: list[str] = Field(default_factory=list)
    default_max_turns: int | None = Field(default=None, ge=1, le=50)
    default_max_tool_calls: int | None = Field(default=None, ge=1, le=200)
    safe_shell_prefixes: list[str] = Field(default_factory=list)
    auto_approve_tools: list[str] = Field(default_factory=list)

    @field_validator("preset_name")
    @classmethod
    def _validate_preset_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="execution.preset_name")

    @field_validator("default_allowlist", "safe_shell_prefixes", "auto_approve_tools")
    @classmethod
    def _validate_lists(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "AgentExecutionSpec":
        if self.mode is AgentExecutionMode.SUBAGENT_PRESET and self.preset_name is None:
            raise ValueError("subagent_preset execution mode requires preset_name.")
        return self


class AgentSpec(SpecBase):
    """
    Declarative agent contract for routing and policy enforcement.
    """

    summary: str | None = None
    role: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    input_schema_ref: str | None = None
    output_schema_ref: str | None = None

    skill_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    mcp_server_ids: list[str] = Field(default_factory=list)

    routing: AgentRoutingPolicy = Field(default_factory=AgentRoutingPolicy)
    policy: AgentRiskPolicy = Field(default_factory=AgentRiskPolicy)
    cost: AgentCostProfile = Field(default_factory=AgentCostProfile)
    execution: AgentExecutionSpec = Field(default_factory=AgentExecutionSpec)

    @field_validator("summary", "role", "input_schema_ref", "output_schema_ref")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @field_validator("capabilities", "skill_ids", "tool_ids", "mcp_server_ids")
    @classmethod
    def _validate_lists(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

