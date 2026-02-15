#!/usr/bin/env python3
"""
VLM Listener — with Data Sanitization & Hallucination Filters.
Filters out "null" people, fixes fake counts, and dedups frames.
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
BATCH_INTERVAL = 2.0
MAX_BATCH_SIZE = 15
DEDUPLICATE = True

# Global Queue
event_queue = asyncio.Queue()
_client: httpx.AsyncClient | None = None
_idx = 0

# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    print(f"🔗 Proxy ready — forwarding to vLLM at {VLLM_BASE}")
    print("🧹 Data Sanitizer: ACTIVE (Filtering nulls & fake counts)")
    asyncio.create_task(batch_processor())
    yield
    if _client:
        await _client.aclose()

app = FastAPI(title="VLM Batching Proxy", lifespan=lifespan)

# ── Cleaning Logic (New!) ───────────────────────────────────────────
def _clean_vlm_data(raw_data):
    """
    Sanitizes VLM output to remove hallucinations and nulls.
    Returns CLEANED data (or None if empty/invalid).
    """
    # 1. Handle primitive types (None, strings)
    if not raw_data:
        return []
    
    # If it's a list (common for object detection outputs)
    if isinstance(raw_data, list):
        clean_list = []
        for item in raw_data:
            cleaned_item = _clean_vlm_data(item)
            if cleaned_item: # Only keep valid items
                clean_list.append(cleaned_item)
        return clean_list

    # If it's a dictionary (a specific person/object)
    if isinstance(raw_data, dict):
        # 2. Filter "Ghost" People (Null/Unknown Names)
        # Adjust these keys based on your VLM's actual output format
        bad_values = ["null", "unknown", "none", "n/a", ""]
        
        name = str(raw_data.get("first_name", "")).lower()
        badge = str(raw_data.get("badge_color", "")).lower()
        
        # Heuristic: If name is null AND badge is unknown -> It's a hallucination. Throw it away.
        if name in bad_values and badge in bad_values:
            return None 

        # 3. Clean individual fields
        clean_obj = {}
        for k, v in raw_data.items():
            if str(v).lower() in bad_values:
                clean_obj[k] = None # Standardize to actual Python None
            else:
                clean_obj[k] = v
        
        return clean_obj

    return raw_data

def _validate_observation(obs):
    """
    Fixes count inconsistencies (e.g., VLM says '1000 people' but list has 1).
    """
    if isinstance(obs, list):
        # If the output is just a list of people, that IS the observation.
        return obs
    
    if isinstance(obs, dict):
        # Fix people_count if it exists
        if "people" in obs and isinstance(obs["people"], list):
            actual_count = len(obs["people"])
            if "people_count" in obs:
                # OVERWRITE the hallucinated count with the real count
                obs["people_count"] = actual_count
        return obs
    
    return obs

# ── Routes ──────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _idx
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    # Forward to vLLM
    try:
        resp = await _client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), status_code=502)

    # Extract & Clean
    vlm_text = ""
    try:
        vllm_data = resp.json()
        vlm_text = vllm_data["choices"][0]["message"]["content"]
    except:
        pass

    if vlm_text:
        # Parse -> Clean -> Validate -> Queue
        raw_obs = _parse_vlm_output(vlm_text)
        cleaned_obs = _clean_vlm_data(raw_obs)
        final_obs = _validate_observation(cleaned_obs)
        
        # Convert back to JSON string for the queue/deduplicator
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
            # DEDUPLICATION:
            # Since we cleaned the data *before* queuing, "null" frames 
            # became "[]" (empty list). 
            # If we get 10 frames of "[]", this logic sees them as identical 
            # and drops 9 of them.
            if DEDUPLICATE and text_obs == last_processed_text:
                continue
            
            last_processed_text = text_obs
            
            # Don't send empty lists to Nemotron (further reduces load)
            if text_obs == "[]" or text_obs == "{}":
                continue

            unique_observations.append({
                "time": ts,
                "observation": json.loads(text_obs)
            })

        if not unique_observations:
            # If batch was full of empty/duplicate data, skip compliance check
            continue

        timeline_obs = {
            "type": "timeline_batch",
            "start_time": unique_observations[0]["time"],
            "end_time": unique_observations[-1]["time"],
            "events": unique_observations
        }

        print(f"📦 Batch: {len(batch)} frames -> {len(unique_observations)} valid events")
        asyncio.create_task(_run_compliance_batch(timeline_obs))

async def _run_compliance_batch(observation: dict):
    global _idx
    try:
        report = await asyncio.to_thread(check_compliance, observation, None)
        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        # Only print violations to keep terminal clean
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