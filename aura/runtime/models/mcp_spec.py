from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from .spec_common import SpecBase, _clean_non_empty_str, _dedupe_str_list


class McpTransport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"
    WS = "ws"
    UNKNOWN = "unknown"


class McpAuthType(StrEnum):
    NONE = "none"
    TOKEN = "token"
    OAUTH = "oauth"
    CUSTOM = "custom"


class McpHealthcheck(BaseModel):
    interval_sec: int = Field(default=60, ge=1, le=3600)
    timeout_sec: int = Field(default=3, ge=1, le=120)


class McpAuthSpec(BaseModel):
    type: McpAuthType = McpAuthType.NONE
    scopes: list[str] = Field(default_factory=list)

    @field_validator("scopes")
    @classmethod
    def _validate_scopes(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)


class McpProvidedToolSpec(BaseModel):
    remote_name: str
    tool_id: str | None = None
    description: str | None = None

    @field_validator("remote_name")
    @classmethod
    def _validate_remote_name(cls, v: str) -> str:
        return _clean_non_empty_str(v, field_name="provides_tools.remote_name")

    @field_validator("tool_id", "description")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))


class McpServerSpec(SpecBase):
    """
    Declarative MCP server contract.
    """

    enabled: bool = True
    transport: McpTransport = McpTransport.UNKNOWN
    endpoint: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    timeout_sec: int = Field(default=60, ge=1, le=3600)
    auth: McpAuthSpec = Field(default_factory=McpAuthSpec)
    healthcheck: McpHealthcheck = Field(default_factory=McpHealthcheck)
    protocol_version: str | None = None
    provides_tools: list[McpProvidedToolSpec] = Field(default_factory=list)

    @field_validator("endpoint", "command", "cwd", "protocol_version")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @field_validator("args")
    @classmethod
    def _validate_args(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "McpServerSpec":
        if self.transport is McpTransport.STDIO and self.command is None:
            raise ValueError("stdio MCP server requires command.")
        if self.transport in {McpTransport.HTTP, McpTransport.WS} and self.endpoint is None:
            raise ValueError("http/ws MCP server requires endpoint.")
        return self

    def provided_tool_ids(self) -> list[str]:
        out: list[str] = []
        for item in self.provides_tools:
            if item.tool_id is None:
                continue
            out.append(item.tool_id)
        return out

