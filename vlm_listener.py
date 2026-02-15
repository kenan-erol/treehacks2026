#!/usr/bin/env python3
"""
VLM Listener — reverse-proxy with Rate Limiting.
Prevents flooding Ollama by skipping frames if the previous check is still running.
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
VLLM_BASE = ""  # Set in main()
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VLM Compliance Proxy")

# Shared async HTTP client
_client: httpx.AsyncClient | None = None
_idx = 0

# 🔴 RATE LIMITING FLAG: Prevents flooding Ollama
COMPLIANCE_BUSY = False

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

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _idx, COMPLIANCE_BUSY
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    # 1. Forward request to vLLM (Cosmos)
    try:
        resp = await _client.post(
            "/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(f"[{ts}] ❌ vLLM unreachable: {e}")
        return Response(content=json.dumps({"error": str(e)}), status_code=502)

    # 2. Extract VLM Response
    vlm_text = ""
    try:
        vllm_data = resp.json()
        vlm_text = vllm_data["choices"][0]["message"]["content"]
    except Exception:
        pass  # Failed to parse, probably empty or error

    # 3. Trigger Compliance Check (ONLY IF IDLE)
    if vlm_text:
        # Check if we are already running a compliance check
        if COMPLIANCE_BUSY:
            print(f"[{ts}] ⏳ Skipping compliance (previous check still running)...")
        else:
            print(f"[{ts}] 🔍 VLM Output: {vlm_text[:50]}...")
            observation = _parse_vlm_output(vlm_text)
            
            # Mark as busy and start task
            COMPLIANCE_BUSY = True
            asyncio.create_task(_run_compliance_safe(observation, _idx, ts))
            _idx += 1

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests (like model listing) untouched."""
    url = f"/{path}"
    body = await request.body()
    try:
        resp = await _client.request(
            method=request.method,
            url=url,
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=502)

def _parse_vlm_output(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {"raw_description": text}

async def _run_compliance_safe(observation: dict, idx: int, ts: str):
    """Wrapper to run compliance check and ALWAYS release the busy flag."""
    global COMPLIANCE_BUSY
    try:
        print(f"[{ts}] ⚖️  Starting Compliance Check on Nemotron...")
        
        # Run the blocking function in a separate thread
        report = await asyncio.to_thread(check_compliance, observation, None)

        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        icon = "✅" if status == "compliant" else "🚨"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} RESULT: {status.upper()} | Violations: {len(violations)}")

        if violations:
            for v in violations:
                print(f"       ⛔ {v.get('rule', 'Rule')}: {v.get('description', '')[:80]}")

        # Save to disk
        save_path = REPORT_DIR / f"report_{idx:04d}.json"
        save_path.write_text(json.dumps({"obs": observation, "report": report}, indent=2))

    except Exception as e:
        print(f"[{ts}] ❌ Compliance Check Error: {e}")
    finally:
        # CRITICAL: Always release the lock so the next frame can be processed
        COMPLIANCE_BUSY = False

def main():
    global VLLM_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    VLLM_BASE = args.vllm_url
    print(f"🚀 Proxy listening on {args.host}:{args.port} -> vLLM {VLLM_BASE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")

if __name__ == "__main__":
    main()