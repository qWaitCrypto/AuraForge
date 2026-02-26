from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SpecLifecycle(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


def _clean_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


def _dedupe_str_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class SpecBase(BaseModel):
    """
    Shared envelope for registry-managed specs.

    This is declarative metadata only. Runtime execution lives elsewhere.
    """

    spec_version: str = "1.0"
    id: str
    name: str
    vendor: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: SpecLifecycle = SpecLifecycle.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        cleaned = _clean_non_empty_str(v, field_name="id")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._:-")
        if not cleaned[0].isalnum():
            raise ValueError("id must start with an alphanumeric character.")
        if any(ch.lower() not in allowed for ch in cleaned.lower()):
            raise ValueError("id contains unsupported characters.")
        if len(cleaned) < 3 or len(cleaned) > 160:
            raise ValueError("id length must be between 3 and 160 characters.")
        return cleaned

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return _clean_non_empty_str(v, field_name="name")

    @field_validator("vendor")
    @classmethod
    def _validate_vendor(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _clean_non_empty_str(v, field_name="vendor")

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str]) -> list[str]:
        return _dedupe_str_list(v)

