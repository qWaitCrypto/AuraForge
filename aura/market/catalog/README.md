# Aura Market Catalog

This directory is the built-in agent market source for Aura.

## Goals
- Market assets are part of project source control.
- Runtime loading must not depend on temporary `dev/` folders.
- User-specific additions are loaded from `.aura/market/custom`.

## Layout
- `tools/` -> `ToolSpec` JSON files
- `mcp_servers/` -> `McpServerSpec` JSON files
- `skills/` -> `SkillSpec` JSON files
- `agents/` -> `AgentSpec` JSON files
- `index.json` -> optional explicit manifest of file paths

## File format
Each JSON file can be one of:
1. A single spec object
2. A list of spec objects
3. An object with `items` (list)

## Loading order
1. Built-in catalog (`aura/market/catalog`)
2. User catalog (`.aura/market/custom`)

Later sources override earlier sources on the same `id`.
