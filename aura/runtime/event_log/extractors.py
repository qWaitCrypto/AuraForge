from __future__ import annotations

import json
import re
from typing import Any

_COMMIT_RE = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
_GITHUB_PR_URL_RE = re.compile(r"https://github\.com/[^\s/]+/[^\s/]+/pull/\d+")
_LINEAR_URL_RE = re.compile(r"https://linear\.app/[^\s\"')>]+")


def _compact_text(value: Any, *, max_len: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "…"


def _json_keys_summary(args: dict[str, Any]) -> str:
    keys = sorted(k for k in args.keys() if isinstance(k, str))
    if not keys:
        return "{}"
    return "keys=" + ",".join(keys)


def summarize_tool_args(tool_name: str, args: dict[str, Any], *, max_len: int = 200) -> str:
    if tool_name == "shell__run":
        command = args.get("command")
        if isinstance(command, str) and command.strip():
            return _compact_text(command, max_len=max_len)
        return "shell command"

    if tool_name == "project__read_text":
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            return f"path={path.strip()}"

    if tool_name == "project__apply_edits":
        edits = args.get("edits")
        if isinstance(edits, list):
            paths: set[str] = set()
            for item in edits:
                if isinstance(item, dict):
                    path = item.get("path")
                    if isinstance(path, str) and path.strip():
                        paths.add(path.strip())
            return f"{len(edits)} ops on {len(paths)} file(s)"

    return _compact_text(_json_keys_summary(args), max_len=max_len)


def summarize_tool_result(tool_name: str, result: Any, *, max_len: int = 200) -> str:
    if tool_name == "shell__run" and isinstance(result, dict):
        code = result.get("exit_code")
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        head = stdout if isinstance(stdout, str) and stdout.strip() else stderr
        if isinstance(head, str) and head.strip():
            return _compact_text(f"exit={code} {head}", max_len=max_len)
        return f"exit={code}"

    if tool_name == "project__read_text" and isinstance(result, dict):
        path = result.get("path")
        text = result.get("text")
        if isinstance(path, str) and isinstance(text, str):
            return f"read {path} ({len(text)} chars)"

    if isinstance(result, dict):
        ok = result.get("ok")
        if isinstance(ok, bool):
            if ok:
                return "ok"
            error = result.get("error")
            if isinstance(error, str) and error.strip():
                return _compact_text(f"error: {error}", max_len=max_len)
            return "error"

    return _compact_text(str(result), max_len=max_len)


def _stringify_payload(tool_args: dict[str, Any], tool_result: Any) -> str:
    try:
        result_text = json.dumps(tool_result, ensure_ascii=False, sort_keys=True)
    except Exception:
        result_text = str(tool_result)
    try:
        args_text = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    except Exception:
        args_text = str(tool_args)
    return args_text + "\n" + result_text


def extract_external_refs(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: dict[str, Any] | Any,
) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        if value in seen:
            return
        seen.add(value)
        refs.append(value)

    payload_text = _stringify_payload(tool_args, tool_result)

    if tool_name == "shell__run":
        command = str(tool_args.get("command") or "")
        if "git push" in command:
            m = re.search(r"git\s+push(?:\s+-u)?\s+\S+\s+([\w./:-]+)", command)
            if m is not None:
                _add(f"push:{m.group(1)}")
            else:
                _add("push:unknown")

        if "git commit" in command:
            for sha in _COMMIT_RE.findall(payload_text):
                _add(f"commit:{sha.lower()}")

    tool_name_l = tool_name.lower()
    if "linear" in tool_name_l:
        if isinstance(tool_result, dict):
            for key in ("id", "commentId", "comment_id"):
                value = tool_result.get(key)
                if isinstance(value, str) and value.strip():
                    _add(f"linear:{value.strip()}")

    if "github" in tool_name_l or "gh" in tool_name_l:
        for url in _GITHUB_PR_URL_RE.findall(payload_text):
            _add(f"pr:{url}")

    for url in _GITHUB_PR_URL_RE.findall(payload_text):
        _add(f"pr:{url}")

    for url in _LINEAR_URL_RE.findall(payload_text):
        _add(f"linear:{url}")

    return refs
