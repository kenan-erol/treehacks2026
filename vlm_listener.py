#!/usr/bin/env python3
"""
VLM Listener — Minimalist.
Only filters EMPTY lists. Does not touch data fields.
"""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
import uvicorn

# ── Local import ────────────────────────────────────────────────────
from compliance_checker import check_compliance

# ── Config ──────────────────────────────────────────────────────────
VLLM_BASE = ""
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# BATCH CONFIGURATION
BATCH_INTERVAL = 2.0
MAX_BATCH_SIZE = 15
DEDUPLICATE = True

event_queue = asyncio.Queue()
_client: httpx.AsyncClient | None = None
_idx = 0

# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    print(f"🔗 Proxy ready — forwarding to vLLM at {VLLM_BASE}")
    print("🚀 Filter Mode: MINIMAL (Only dropping [] empty lists)")
    asyncio.create_task(batch_processor())
    yield
    if _client:
        await _client.aclose()

app = FastAPI(title="VLM Batching Proxy", lifespan=lifespan)

# ── Safe Helpers ────────────────────────────────────────────────────
def _fix_counts_only(obs):
    """
    Only fixes the '1000 people' bug. Does NOT delete any data.
    """
    if isinstance(obs, dict):
        if "people" in obs and isinstance(obs["people"], list):
            # Overwrite count with actual list length
            obs["people_count"] = len(obs["people"])
    return obs

# ── Routes ──────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    # Forward to vLLM
    try:
        resp = await _client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=502)

    # Extract VLM Output
    vlm_text = ""
    try:
        vllm_data = resp.json()
        vlm_text = vllm_data["choices"][0]["message"]["content"]
    except:
        pass

    if vlm_text:
        # 1. Parse
        raw_obs = _parse_vlm_output(vlm_text)
        
        # 2. Fix Count Only (Safe)
        final_obs = _fix_counts_only(raw_obs)
        
        # 3. Queue (Send everything, let batch processor handle deduping)
        event_queue.put_nowait((ts, json.dumps(final_obs)))

    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_passthrough(request: Request, path: str):
    url = f"/{path}"
    body = await request.body()
    try:
        resp = await _client.request(method=request.method, url=url, content=body, headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")})
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=502)

# ── Batch Processor ─────────────────────────────────────────────────
async def batch_processor():
    last_processed_text = None
    
    while True:
        first_item = await event_queue.get()
        batch = [first_item]
        
        deadline = asyncio.get_event_loop().time() + BATCH_INTERVAL
        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0: break
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError: break
        
        unique_observations = []
        
        for ts, text_obs in batch:
            # 1. Skip strictly EMPTY lists (Fixes latency/hallucination)
            if text_obs == "[]" or text_obs == "{}" or text_obs == "null":
                continue
            
            # 2. Deduplicate
            if DEDUPLICATE and text_obs == last_processed_text:
                continue
            
            last_processed_text = text_obs
            
            # 3. Add to batch
            unique_observations.append({
                "time": ts,
                "observation": json.loads(text_obs)
            })

        if not unique_observations:
            continue

        print(f"📦 Batch: {len(batch)} frames -> {len(unique_observations)} unique events")
        
        # Construct timeline
        timeline_obs = {
            "type": "timeline_batch",
            "start_time": unique_observations[0]["time"],
            "end_time": unique_observations[-1]["time"],
            "events": unique_observations
        }

        asyncio.create_task(_run_compliance_batch(timeline_obs))

async def _run_compliance_batch(observation: dict):
    global _idx
    try:
        report = await asyncio.to_thread(check_compliance, observation, None)
        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        # Logging
        if violations:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 {status.upper()} | {len(violations)} Violations")
            for v in violations:
                print(f"   ⛔ {v.get('rule', '?')}: {v.get('description', '')[:80]}")
        else:
             print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Compliant")

        _idx += 1
        save_path = REPORT_DIR / f"batch_report_{_idx:04d}.json"
        save_path.write_text(json.dumps({"obs": observation, "report": report}, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")

def _parse_vlm_output(text: str):
    try:
        return json.loads(text)
    except:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {}

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