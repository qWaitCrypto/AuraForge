# Linear MCP Setup

This project expects Linear access through MCP, not hardcoded API calls.

## 1) Configure `.aura/config/mcp.json`

Add or enable a `linear` server entry:

```json
{
  "mcpServers": {
    "linear": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.linear.app/mcp"],
      "env": {},
      "cwd": "",
      "timeout_s": 60
    }
  }
}
```

Notes:
- The preferred endpoint is `https://mcp.linear.app/mcp`.
- `/sse` should be treated as fallback compatibility mode only.

## 2) Restart Aura session

MCP tools are discovered at runtime. After editing config, restart the session so `mcp__linear__*` tools are rebuilt.

## 3) Verify in runtime

At startup, check that:
- the `linear` server is present in loaded MCP config,
- `mcp__linear__list_issues` (and related tools) appears in the capability surface for agents that include the linear skill/MCP.

## 4) Audit expectation

When agents operate on Linear through MCP:
- `EventLog` records each tool call automatically,
- external references (for example Linear URLs/IDs) are extracted into `external_refs`,
- reviewers can query via `audit__query` / `audit__refs`.
