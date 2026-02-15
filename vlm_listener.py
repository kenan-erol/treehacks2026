#!/usr/bin/env python3
"""
VLM Listener — High-Performance Batching Proxy.
Implements Producer-Consumer architecture to handle high-speed VLM input
without dropping data or overloading the slow Nemotron reasoning agent.
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
VLLM_BASE = ""  # Set in main()
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# BATCH CONFIGURATION
BATCH_INTERVAL = 2.0  # Seconds to wait to accumulate frames
MAX_BATCH_SIZE = 15   # Max frames to pack into one Nemotron request
DEDUPLICATE = True    # If True, identical sequential frames are merged

# Global Queue
event_queue = asyncio.Queue()

# Shared async HTTP client
_client: httpx.AsyncClient | None = None
_idx = 0

# ── Lifespan (Startup/Shutdown) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    
    # 1. Setup vLLM Client
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    print(f"🔗 Proxy ready — forwarding to vLLM at {VLLM_BASE}")
    
    # 2. Start the Background Batch Processor
    print("⚙️  Starting Background Batch Processor...")
    asyncio.create_task(batch_processor())

    yield
    
    # Shutdown
    if _client:
        await _client.aclose()

app = FastAPI(title="VLM Batching Proxy", lifespan=lifespan)

# ── Routes ──────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _idx
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    # 1. Forward request to vLLM (Cosmos) - The "Producer"
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
        pass

    # 3. Push to Queue (Non-Blocking)
    if vlm_text:
        # We push a tuple of (timestamp, text_data)
        event_queue.put_nowait((ts, vlm_text))
        q_size = event_queue.qsize()
        if q_size % 10 == 0:
            print(f"[{ts}] 📥 Buffered frame (Queue size: {q_size})")

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_passthrough(request: Request, path: str):
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

# ── Background Worker (The Consumer) ────────────────────────────────
async def batch_processor():
    """
    Infinite loop that:
    1. Waits for data in the queue
    2. Accumulates data for BATCH_INTERVAL seconds
    3. Deduplicates identical sequential frames
    4. Sends one consolidated 'Timeline' report to Nemotron
    """
    last_processed_text = None
    
    while True:
        # Wait for the first item (don't burn CPU if empty)
        first_item = await event_queue.get()
        batch = [first_item]
        
        # Now gather any other items that arrive within our interval
        # or until we hit MAX_BATCH_SIZE
        deadline = asyncio.get_event_loop().time() + BATCH_INTERVAL
        while len(batch) < MAX_BATCH_SIZE:
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        # We now have a batch of raw frames. Let's process them.
        unique_observations = []
        
        for ts, text in batch:
            # Parse JSON if possible
            obs = _parse_vlm_output(text)
            
            # Deduplication: If this frame is exactly the same as the last unique one, skip it
            # (But assume the timestamp updated implicitly)
            current_sig = json.dumps(obs, sort_keys=True)
            if DEDUPLICATE and current_sig == last_processed_text:
                continue
            
            last_processed_text = current_sig
            unique_observations.append({
                "time": ts,
                "observation": obs
            })

        # If everything was a duplicate, we might have nothing to send
        if not unique_observations:
            print(f"💤 All {len(batch)} frames were duplicates. Skipping check.")
            continue

        # Construct the "Timeline Observation" for Nemotron
        timeline_obs = {
            "type": "timeline_batch",
            "start_time": unique_observations[0]["time"],
            "end_time": unique_observations[-1]["time"],
            "event_count": len(unique_observations),
            "events": unique_observations
        }

        # Run Compliance Check (Blocking call in thread)
        print(f"📦 Processing Batch: {len(batch)} raw -> {len(unique_observations)} unique events")
        asyncio.create_task(_run_compliance_batch(timeline_obs))


async def _run_compliance_batch(observation: dict):
    global _idx
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        # Run blocking check in thread
        report = await asyncio.to_thread(check_compliance, observation, None)

        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        icon = "✅" if status == "compliant" else "🚨"
        print(f"[{ts}] {icon} BATCH RESULT: {status.upper()} | Violations: {len(violations)}")
        
        if violations:
             for v in violations:
                print(f"       ⛔ {v.get('rule', 'Rule')}: {v.get('description', '')[:80]}")

        # Save to disk
        _idx += 1
        save_path = REPORT_DIR / f"batch_report_{_idx:04d}.json"
        save_path.write_text(json.dumps({"obs": observation, "report": report}, indent=2))

    except Exception as e:
        print(f"[{ts}] ❌ Batch Check Error: {e}")

# ── Helpers ─────────────────────────────────────────────────────────
def _parse_vlm_output(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {"raw_description": text}

def main():
    global VLLM_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    VLLM_BASE = args.vllm_url
    print(f"🚀 Proxy listening on {args.host}:{args.port} -> vLLM {VLLM_BASE}")
    print(f"⏱️  Batch Interval: {BATCH_INTERVAL}s | Max Batch: {MAX_BATCH_SIZE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")

if __name__ == "__main__":
    main()