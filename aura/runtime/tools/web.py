from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{key}' (expected non-empty string).")
    return value.strip()


def _maybe_int(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid '{key}' (expected int).")
    return value


def _maybe_float(args: dict[str, Any], key: str) -> float | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid '{key}' (expected number).")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Invalid '{key}' (expected number).")


def _maybe_bool(args: dict[str, Any], key: str) -> bool | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid '{key}' (expected boolean).")


def _decode_body(body: bytes, content_type: str | None) -> str:
    charset = None
    if content_type:
        m = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.IGNORECASE)
        if m:
            charset = m.group(1)
    for enc in [charset, "utf-8", "utf-8-sig", "gb18030", "latin-1"]:
        if not enc:
            continue
        try:
            return body.decode(enc, errors="replace")
        except Exception:
            continue
    return body.decode("utf-8", errors="replace")


def _ensure_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported.")
    return url


@dataclass(frozen=True, slots=True)
class WebFetchTool:
    name: str = "web__fetch"
    description: str = (
        "Fetch a web URL and return bounded text content. "
        "This is high-risk (network access) and typically requires user approval."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to fetch."},
                "timeout_s": {"type": "number", "minimum": 0, "description": "Timeout seconds (default 20)."},
                "max_chars": {"type": "integer", "minimum": 1, "description": "Max characters to return (default 24000)."},
                "headers": {
                    "type": "object",
                    "description": "Optional request headers.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root) -> dict[str, Any]:  # project_root unused by design
        url = _ensure_http_url(_require_str(args, "url"))
        timeout_s = _maybe_float(args, "timeout_s")
        if timeout_s is None:
            timeout_s = 20.0
        max_chars = _maybe_int(args, "max_chars") or 24000

        headers_in = args.get("headers")
        headers: dict[str, str] = {}
        if headers_in is not None:
            if not isinstance(headers_in, dict):
                raise ValueError("Invalid 'headers' (expected object).")
            for k, v in headers_in.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ValueError("Invalid 'headers' (expected string->string).")
                headers[k] = v

        headers.setdefault("User-Agent", "Aura/0.1 (+https://example.invalid)")
        headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = int(getattr(resp, "status", 200))
                final_url = str(getattr(resp, "url", url))
                content_type = resp.headers.get("Content-Type")
                body = resp.read()
        except Exception as e:
            return {"ok": False, "url": url, "error": str(e)}

        text = _decode_body(body, content_type)
        truncated = False
        if len(text) > max_chars:
            truncated = True
            text = text[:max_chars]

        return {
            "ok": True,
            "url": url,
            "final_url": final_url,
            "status": status,
            "content_type": content_type,
            "truncated": truncated,
            "content": text,
        }


@dataclass(frozen=True, slots=True)
class WebSearchTool:
    name: str = "web__search"
    description: str = (
        "Run a lightweight web search and return a bounded list of results. "
        "This is high-risk (network access) and typically requires user approval."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "minimum": 1, "description": "Max results (default 5)."},
                "timeout_s": {"type": "number", "minimum": 0, "description": "Timeout seconds (default 20)."},
                "engine": {
                    "type": "string",
                    "enum": ["duckduckgo"],
                    "description": "Search engine (default duckduckgo).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root) -> dict[str, Any]:  # project_root unused by design
        query = _require_str(args, "query")
        max_results = _maybe_int(args, "max_results") or 5
        timeout_s = _maybe_float(args, "timeout_s")
        if timeout_s is None:
            timeout_s = 20.0
        engine = str(args.get("engine") or "duckduckgo")
        if engine != "duckduckgo":
            raise ValueError("Unsupported engine.")

        q = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent": "Aura/0.1 (+https://example.invalid)",
                "Accept": "text/html,*/*;q=0.8",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                html = resp.read()
                content_type = resp.headers.get("Content-Type")
                status = int(getattr(resp, "status", 200))
        except Exception as e:
            return {"ok": False, "query": query, "engine": engine, "error": str(e)}

        text = _decode_body(html, content_type)
        # Minimal HTML extraction for DDG results. (Best-effort, can break if markup changes.)
        # Titles: <a class="result__a" href="...">Title</a>
        # Snippets: <a class="result__snippet">...</a> or <div class="result__snippet">...</div>
        title_re = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
        snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</', re.I | re.S)
        tag_re = re.compile(r"<[^>]+>")
        ws_re = re.compile(r"\s+")

        results: list[dict[str, Any]] = []
        for m in title_re.finditer(text):
            if len(results) >= max_results:
                break
            href = m.group(1)
            title_html = m.group(2)
            title = ws_re.sub(" ", tag_re.sub("", title_html)).strip()
            href = ws_re.sub(" ", href).strip()
            snippet_m = snippet_re.search(text, m.end())
            snippet = ""
            if snippet_m:
                snippet_html = snippet_m.group(1)
                snippet = ws_re.sub(" ", tag_re.sub("", snippet_html)).strip()
            results.append({"title": title, "url": href, "snippet": snippet})

        return {
            "ok": True,
            "query": query,
            "engine": engine,
            "status": status,
            "results": results,
            "debug": {"raw_url": url, "raw_len": len(text)},
        }

