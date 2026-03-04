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

## Phase 3 Acceptance Check

Run:

```bash
python -m aura.runtime.examples.phase3_acceptance_check
```

This command validates roadmap acceptance targets across Phase 0-3 and returns a JSON report with per-check PASS/FAIL.

## End-to-End Showcase (Execution + Control Plane)

```bash
python -m aura.runtime.examples.end_to_end_showcase
```

What it demonstrates in one run:

- Execution-plane E2E demo (`run_demo`): bidding signals, task assignment, sandbox work, and audit refs.
- Control-plane E2E via CLI: `dispatch` (WAKE/TASK_ASSIGNED), `status --json`, `status agent`, `status issue`.
- Probe/recovery loop: timeout policy + `probe --auto-recover` + dead letter visibility.
- Circuit breaker loop: OPEN view in `status mcp` and manual reset via `recover force-close`.
