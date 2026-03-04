Task assigned.

Issue: {issue_key}
Sandbox: {sandbox_id}
Worktree: {worktree_path}
Branch: {branch}
Brief: {brief}

Action:
1. Execute work in the assigned sandbox worktree.
2. Keep changes scoped and testable.
3. Commit all work and run tests where applicable.
4. When complete, you MUST send a completion signal with `signal__send`:
   - `to_agent`: `committee`
   - `signal_type`: `notify`
   - `brief`: `task_completed`
   - `issue_key`: `{issue_key}`
   - `payload`:
     ```json
     {
       "type": "task_completed",
       "issue_key": "{issue_key}",
       "summary": "<one-paragraph delivery summary>",
       "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
       "audit_summary": "<key operations performed>",
       "snapshot_summary": "<files changed, insertions/deletions>"
     }
     ```
5. Do NOT end your session without sending this completion signal.
