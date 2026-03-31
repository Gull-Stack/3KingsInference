#!/usr/bin/env python3
"""OpenAI-compatible API server for 3KingsInference.

Usage:
    python server.py --model mlx-community/Qwen3.5-397B-A17B-4bit --port 8080

Then use any OpenAI-compatible client:
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "qwen3.5-397b", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from core.config import InferenceConfig, ModelConfig
from core.pipeline import ThreeKingsPipeline

# Global pipeline reference
_pipeline: Optional[ThreeKingsPipeline] = None


class APIHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests."""

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_error(404, "Not found")

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_error(404, "Not found")

    def _handle_chat_completions(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._send_error(400, "Invalid JSON")
            return

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        if not messages:
            self._send_error(400, "messages is required")
            return

        # Format prompt
        prompt_ids = _pipeline.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt_text = _pipeline.tokenizer.decode(prompt_ids, skip_special_tokens=False)

        if stream:
            self._handle_stream(prompt_text, max_tokens, temperature, top_p)
        else:
            self._handle_blocking(prompt_text, max_tokens, temperature, top_p, body.get("model", "qwen3.5"))

    def _handle_blocking(self, prompt: str, max_tokens: int, temperature: float, top_p: float, model: str):
        full_text = ""
        for token in _pipeline.generate(prompt, max_tokens=max_tokens,
                                         temperature=temperature, top_p=top_p, stream=True):
            full_text += token

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": _pipeline._stats.prompt_tokens,
                "completion_tokens": _pipeline._stats.tokens_generated,
                "total_tokens": _pipeline._stats.prompt_tokens + _pipeline._stats.tokens_generated,
            },
        }
        self._send_json(response)

    def _handle_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        for token in _pipeline.generate(prompt, max_tokens=max_tokens,
                                         temperature=temperature, top_p=top_p, stream=True):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Final chunk
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_models(self):
        self._send_json({
            "object": "list",
            "data": [{
                "id": "qwen3.5-397b",
                "object": "model",
                "owned_by": "3kings",
            }],
        })

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        body = json.dumps({"error": {"message": message, "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    global _pipeline

    parser = argparse.ArgumentParser(description="3Kings API Server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--expert-dir", type=str, default="./packed_experts")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--machine-id", type=int, default=0)
    parser.add_argument("--peer", type=str, default=None)
    parser.add_argument("--shard-port", type=int, default=5555)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=5)
    args = parser.parse_args()

    n_machines = 2 if args.peer else 1
    peers = [args.peer] if args.peer else []

    config = InferenceConfig(
        model=ModelConfig.qwen_397b(),
        n_machines=n_machines,
        machine_id=args.machine_id,
        peer_addresses=peers,
        shard_port=args.shard_port,
        expert_dir=args.expert_dir,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
    )

    _pipeline = ThreeKingsPipeline(config)
    _pipeline.setup(args.model)

    server = HTTPServer(("0.0.0.0", args.port), APIHandler)
    print(f"3Kings API server running on http://0.0.0.0:{args.port}")
    print(f"  POST /v1/chat/completions — OpenAI-compatible chat")
    print(f"  GET  /v1/models — list models")
    print(f"  GET  /health — health check\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        _pipeline.teardown()
        server.server_close()


if __name__ == "__main__":
    main()
