#!/usr/bin/env python3
"""
VLM Listener — reverse-proxy that sits between the live-vlm-webui
and vLLM, intercepting every VLM response in real-time and piping it
through the Nemotron compliance checker.

Architecture
────────────
  Live WebUI ──► vlm_listener :8001 ──► vLLM (Cosmos) :8000
                      │
                      │  intercepts every response
                      ▼
              compliance_checker (Nemotron via Ollama)
                      │
                      ▼
               reports/ (JSON files)

Setup
─────
  1. Start vLLM Docker as normal (port 8000)
  2. Run:  python vlm_listener.py
  3. Point the live-vlm-webui at port 8001 instead of 8000
     (change the "API Base" field in the webui to http://<dgx-ip>:8001/v1)

That's it — every VLM query flows through this proxy transparently.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
import uvicorn

# ── Local import ────────────────────────────────────────────────────
from compliance_checker import check_compliance

# ── Config ──────────────────────────────────────────────────────────
VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VLM Compliance Proxy")

# Shared async HTTP client (created on startup)
_client: httpx.AsyncClient | None = None

# Counter for reports
_idx = 0


@app.on_event("startup")
async def _startup():
    global _client
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    print(f"🔗 Proxy ready — forwarding to vLLM at {VLLM_BASE}")
    print(f"📁 Reports will be saved to {REPORT_DIR.resolve()}")


@app.on_event("shutdown")
async def _shutdown():
    if _client:
        await _client.aclose()


# ────────────────────────────────────────────────────────────────────
# The key endpoint: intercept /v1/chat/completions
# ────────────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _idx
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    print(f"[{ts}] 📨 Incoming request → forwarding to vLLM…")

    # Forward the request exactly as-is to vLLM
    try:
        resp = await _client.post(
            "/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(f"[{ts}] ❌ vLLM unreachable: {e}")
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=502,
            media_type="application/json",
        )

    # Parse the vLLM response
    try:
        vllm_data = resp.json()
    except Exception:
        # Not JSON — just pass through transparently
        print(f"[{ts}] ⚠️  Non-JSON response from vLLM, passing through")
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/json"))

    # Extract the VLM's text output
    vlm_text = ""
    try:
        vlm_text = vllm_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass

    if vlm_text:
        print(f"[{ts}] 🔍 VLM says: {vlm_text[:150]}…")

        # Try to parse VLM output as JSON
        observation = _parse_vlm_output(vlm_text)

        # Run compliance in background so we don't slow down the webui
        asyncio.create_task(_run_compliance_async(observation, _idx, ts))
        _idx += 1

    # Return the original vLLM response to the webui unchanged
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ────────────────────────────────────────────────────────────────────
# Catch-all: proxy everything else (model list, health, etc.)
# ────────────────────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_passthrough(request: Request, path: str):
    url = f"/{path}"
    body = await request.body()

    try:
        resp = await _client.request(
            method=request.method,
            url=url,
            content=body,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "content-length")},
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=502,
            media_type="application/json",
        )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _parse_vlm_output(text: str) -> dict:
    """Try to parse VLM text as JSON; wrap in a dict if it fails."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "raw_description": text,
                "timestamp": datetime.now().isoformat(),
            }


async def _run_compliance_async(observation: dict, idx: int, ts: str):
    """Run the compliance check in a thread pool (it's sync/blocking)."""
    try:
        print(f"[{ts}] ⚖️  Running compliance check (background)…")
        # Run blocking Ollama call in a thread so we don't block the event loop
        report = await asyncio.to_thread(check_compliance, observation, None)

        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        risk = report.get("risk_score", "?")

        icon = "🚨" if violations else "✅"
        print(f"[{ts}] {icon} Compliance: {status} | Risk: {risk}/100 | Violations: {len(violations)}")

        if violations:
            for v in violations:
                print(f"       ⛔ {v.get('rule', '?')}: {v.get('description', '')[:100]}")

        # Save report
        save_path = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:04d}.json"
        combined = {
            "observation": observation,
            "compliance_report": report,
        }
        save_path.write_text(json.dumps(combined, indent=2))
        print(f"[{ts}] 💾 Saved → {save_path}")

    except Exception as e:
        print(f"[{ts}] ❌ Compliance check failed: {e}")


# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
def main():
    # FIX: Declare global immediately at the start of the function
    global VLLM_BASE 

    parser = argparse.ArgumentParser(
        description="VLM Compliance Proxy — intercepts vLLM responses and runs Nemotron compliance checks"
    )
    parser.add_argument("--port", type=int, default=PROXY_PORT,
                        help=f"Proxy listen port (default: {PROXY_PORT})")
    
    # Now this usage of VLLM_BASE is legal because we declared it global above
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE,
                        help=f"vLLM backend URL (default: {VLLM_BASE})")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    # Update the global config with the argument
    VLLM_BASE = args.vllm_url

    print("╔══════════════════════════════════════════════════╗")
    print("║       VLM Compliance Proxy                       ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Proxy:   http://{args.host}:{args.port}               ║")
    print(f"║  vLLM:    {VLLM_BASE:<38} ║")
    print(f"║  Reports: {str(REPORT_DIR.resolve()):<38} ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Point your live-vlm-webui at this proxy port!   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()