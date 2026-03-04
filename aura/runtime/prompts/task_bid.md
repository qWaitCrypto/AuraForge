You were woken for issue `{issue_key}`.

1. Read the issue details using Linear MCP.
2. Decide if your skills match this task.
3. If it is a strong fit, submit one bid comment in the issue using this exact structure:

````markdown
## 📋 BID
```json
{
  "agent_id": "{your_agent_id}",
  "confidence": "high|medium|low",
  "approach": "2-5 concise sentences",
  "deliverables": ["deliverable 1", "deliverable 2"],
  "estimated_files": 0,
  "estimated_turns": 0,
  "risks": [],
  "questions": [],
  "relevant_experience": []
}
```
````

4. If this task is not a fit, do nothing and end.

Rules:
- Be honest about confidence and experience.
- Keep bid JSON valid and machine-parseable.
- Do not claim unavailable tools or permissions.
