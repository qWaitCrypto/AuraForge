---
name: gh-address-comments
description: Help address review/issue comments on the open GitHub PR for the current branch using gh CLI; verify gh auth first and prompt the user to authenticate if not logged in.
metadata:
  short-description: Address comments in a GitHub PR review
---

# PR Comment Handler

Guide to find the open PR for the current branch and address its comments with gh CLI.

Prereq: ensure `gh` is authenticated (for example, run `gh auth login` once), then run `gh auth status` (repo + workflow scopes are typically required).
Before running bundled scripts, call `skill__load {"name":"gh-address-comments"}` and use `skill.skill_root` for script paths.

## 1) Inspect comments needing attention
- Run `python "<skill_root>/scripts/fetch_comments.py"` to print all comments and review threads on the PR.

## 2) Ask the user for clarification
- Number all the review threads and comments and provide a short summary of what would be required to apply a fix for it
- Ask the user which numbered comments should be addressed

## 3) If user chooses comments
- Apply fixes for the selected comments

Notes:
- If gh hits auth/rate issues mid-run, prompt the user to re-authenticate with `gh auth login`, then retry.
