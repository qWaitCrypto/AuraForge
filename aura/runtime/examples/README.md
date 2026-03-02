# Runtime Examples

## Phase 3 E2E Demo

Run:

```bash
python -m aura.runtime.examples.phase3_e2e_demo
```

Optional flags:
- `--issue-key DEMO-123`
- `--base-branch main`
- `--agents market_fused__python-pro market_fused__backend-developer market_fused__security-auditor`
- `--keep-sandboxes` (do not auto-destroy demo sandboxes)

What it demonstrates:
- 3+ sandboxes created for one issue key (isolation check),
- signal flow: `wake -> task_assigned -> notify`,
- bid/assignment/work completion represented as audit events,
- `audit__query` + `audit__refs` verification output,
- local check that `mcpServers.linear` exists in `.aura/config/mcp.json`.
