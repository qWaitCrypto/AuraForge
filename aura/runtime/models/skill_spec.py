from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from .spec_common import SpecBase, _clean_non_empty_str, _dedupe_str_list


class SkillEntryType(StrEnum):
    TEMPLATE = "template"
    SCRIPT = "script"
    PROMPT = "prompt"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class SkillEntrySpec(BaseModel):
    type: SkillEntryType = SkillEntryType.UNKNOWN
    ref: str

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, v: str) -> str:
        return _clean_non_empty_str(v, field_name="entry.ref")


class SkillOutputContract(BaseModel):
    format: str | None = None
    sections: list[str] = Field(default_factory=list)

    @field_validator("format")
    @classmethod
    def _validate_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="output_contract.format")

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)


class SkillSpec(SpecBase):
    """
    Declarative skill contract.
    """

    description: str | None = None
    source_path: str | None = None
    triggers: list[str] = Field(default_factory=list)
    context_requirements: list[str] = Field(default_factory=list)
    entry: SkillEntrySpec
    requires_tool_ids: list[str] = Field(default_factory=list)
    requires_capabilities: list[str] = Field(default_factory=list)
    output_contract: SkillOutputContract = Field(default_factory=SkillOutputContract)

    @field_validator("description", "source_path")
    @classmethod
    def _validate_optional_strs(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @field_validator("triggers", "context_requirements", "requires_tool_ids", "requires_capabilities")
    @classmethod
    def _validate_lists(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

