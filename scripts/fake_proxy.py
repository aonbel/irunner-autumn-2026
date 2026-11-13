#!/usr/bin/env python3
"""A tiny OpenAI-compatible mock proxy for local tests.

It is intentionally dependency-free.  Run it with:

    python scripts/fake_proxy.py --port 8000

Environment variables:
    FAKE_RESPONSE      default response text (default: "42")
    FAKE_MAX_CALLS     return HTTP 402 after this many calls (0 = unlimited)
    FAKE_FAIL_FIRST    return HTTP 429 on the first N calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_STATE = {
    "calls": 0,
    "failed": 0,
}
_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "FakeProxy/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        print(f"[fake-proxy] {fmt % args}", file=sys.stderr, flush=True)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self._send_json(200, {"object": "list", "data": [{"id": "qwen3.6-35b-a3b-noreason", "object": "model"}]})
            return
        self._send_json(200, {"ok": True, "calls": _STATE["calls"]})

    def do_POST(self):  # noqa: N802
        path = self.path.rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"message": f"unknown path {path}"}})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._send_json(400, {"error": {"message": "invalid json"}})
            return

        with _LOCK:
            _STATE["calls"] += 1
            call_no = _STATE["calls"]
            fail_first = int(os.environ.get("FAKE_FAIL_FIRST", "0") or 0)
            max_calls = int(os.environ.get("FAKE_MAX_CALLS", "0") or 0)
            should_fail = call_no <= fail_first
            over_budget = max_calls > 0 and call_no > max_calls

        if should_fail:
            self._send_json(429, {"error": {"message": "rate limited"}})
            return
        if over_budget:
            self._send_json(402, {"error": {"message": "budget exhausted"}})
            return

        messages = request.get("messages") or []
        tools = request.get("tools") or []
        use_tool = bool(tools) and os.environ.get("FAKE_TOOL_MODE", "1") != "0"
        content = os.environ.get("FAKE_RESPONSE")
        if content is None and not use_tool:
            content = self._respond(messages, request)
        prompt_tokens = max(1, len(json.dumps(messages)) // 4)
        completion_tokens = max(1, len(content or "") // 4)
        reasoning_tokens = int(os.environ.get("FAKE_REASONING_TOKENS", "0") or 0)
        message = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if use_tool:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{call_no}",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": json.dumps({"answer": os.environ.get("FAKE_TOOL_ANSWER", "42")}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        response = {
            "id": f"chatcmpl-fake-{call_no}",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                **(
                    {"completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
                    if reasoning_tokens
                    else {}
                ),
            },
        }
        self._send_json(200, response)

    @staticmethod
    def _respond(messages, request) -> str:
        last = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last = str(message.get("content") or "")
                break
        lowered = last.casefold()
        if "return only a python expression" in lowered:
            return "100"
        if "return only the final integer" in lowered:
            return "100"
        if "answer the question using only" in lowered:
            return "1983"
        if "write a short python script" in lowered:
            return "import csv\nprint(1)\n"
        if "fix the bugs" in lowered:
            return "<<<FILE: mylib.py>>>\ndef total(xs):\n    return sum(xs)\n<<<END>>>"
        return os.environ.get("FAKE_RESPONSE", "42")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[fake-proxy] listening on {args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
