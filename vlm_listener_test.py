#!/usr/bin/env python3
"""
VLM Listener — Optimized Batching Proxy.
Fixes:
1. [] Hallucinations -> Filters out empty frames before LLM.
2. Fake Counts -> Recalculates people_count from list length.
3. Data Loss -> Uses an infinite queue so no frames are dropped.
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
    
    # 1. Setup vLLM Client (The Eyes)
    _client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    print(f"[PROXY] Ready, forwarding to vLLM at {VLLM_BASE}")
    
    # Auto-initialize Nemotron
    print("[PROXY] Loading Nemotron (30B model, may take a minute)...")
    try:
        async with httpx.AsyncClient(timeout=300.0) as ollama:
            resp = await ollama.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "nemotron-3-nano:30b",
                    "prompt": "system check", 
                    "stream": False
                }
            )
            if resp.status_code == 200:
                print("[PROXY] Nemotron loaded.")
            else:
                print(f"[PROXY] Ollama warning: {resp.text}")
    except Exception as e:
        print(f"[PROXY] Could not initialize Nemotron: {type(e).__name__}: {e}")
        print("         Make sure 'ollama serve' is running.")

    # Start background batch processor
    asyncio.create_task(batch_processor())

    yield
    
    # Shutdown
    if _client:
        await _client.aclose()

app = FastAPI(title="VLM Batching Proxy", lifespan=lifespan)

# ── Helper: Fix Counts Safely ───────────────────────────────────────
def _fix_counts_only(obs):
    """
    Fixes the '1000 people' bug by overwriting people_count 
    with the actual number of items in the list. 
    Does NOT delete any data/fields.
    """
    if isinstance(obs, dict):
        if "people" in obs and isinstance(obs["people"], list):
            obs["people_count"] = len(obs["people"])
    return obs

# ── Routes ──────────────────────────────────────────────────────────
@app.post("/v1/analyze_frame_sync")
async def analyze_frame_sync(request: Request):
    """
    Synchronous endpoint for laptop client.
    Workflow: Laptop → Cosmos → Nemotron → Report back to Laptop
    """
    global _idx
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")
    
    # 1. Forward to Cosmos (vLLM)
    try:
        resp = await _client.post(
            "/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(f"[{ts}] ERROR: vLLM unreachable: {e}")
        return Response(
            content=json.dumps({"error": f"Cosmos unreachable: {str(e)}"}), 
            status_code=502,
            media_type="application/json"
        )
    
    # 2. Extract Cosmos VLM Response
    vlm_text = ""
    try:
        vllm_data = resp.json()
        vlm_text = vllm_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[{ts}] ERROR: Failed to parse Cosmos response: {e}")
        return Response(
            content=json.dumps({"error": f"Invalid Cosmos response: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )
    
    if not vlm_text:
        return Response(
            content=json.dumps({"error": "Empty response from Cosmos"}),
            status_code=500,
            media_type="application/json"
        )
    
    # 3. Log Cosmos output
    print(f"\n[{ts}] COSMOS OUTPUT:")
    print(vlm_text)
    
    # 4. Parse and fix counts
    raw_obs = _parse_vlm_output(vlm_text)
    fixed_obs = _fix_counts_only(raw_obs)
    
    # If Cosmos returned a list, wrap it into expected format
    if isinstance(fixed_obs, list):
        fixed_obs = {"people_count": len(fixed_obs), "people": fixed_obs}
    
    # If parsing failed (free text, not JSON), pass raw text as-is
    if not fixed_obs or fixed_obs == {}:
        fixed_obs = {"raw_description": vlm_text}
    
    # 5. Run Nemotron Compliance Check
    print(f"[{ts}] Running compliance check...")
    try:
        report = await asyncio.to_thread(check_compliance, fixed_obs, None)
        
        _idx += 1
        save_path = REPORT_DIR / f"batch_report_{_idx:04d}.json"
        save_path.write_text(json.dumps({
            "obs": fixed_obs, 
            "report": report,
            "source": "laptop_sync"
        }, indent=2))
        
        # 6. Log result
        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        if violations:
            print(f"[{ts}] {status.upper()} | {len(violations)} violation(s)")
            for v in violations:
                who = v.get('subject', 'Unknown')
                rule = v.get('rule', 'Rule')
                desc = v.get('description', '')[:60]
                print(f"  [{who}] {rule}: {desc}")
        else:
            print(f"[{ts}] {status.upper()} | No violations")
        
        # 7. Return report to laptop
        return Response(
            content=json.dumps(report, indent=2),
            status_code=200,
            media_type="application/json"
        )
        
    except Exception as e:
        print(f"[{ts}] ERROR: Compliance check failed: {e}")
        return Response(
            content=json.dumps({"error": f"Nemotron compliance check failed: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global _idx
    body = await request.body()
    ts = datetime.now().strftime("%H:%M:%S")

    try:
        resp = await _client.post(
            "/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(f"[{ts}] ERROR: vLLM unreachable: {e}")
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
        # Parse & Fix Counts immediately
        raw_obs = _parse_vlm_output(vlm_text)
        fixed_obs = _fix_counts_only(raw_obs)
        
        # Serialize back to string for the queue
        event_queue.put_nowait((ts, json.dumps(fixed_obs)))
        
        q_size = event_queue.qsize()
        if q_size % 50 == 0:
            print(f"[{ts}] Buffered frame (queue: {q_size})")

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
    """Background worker: batches frames, deduplicates, sends to Nemotron."""
    last_processed_text = None
    
    while True:
        # Wait for the first item
        first_item = await event_queue.get()
        batch = [first_item]
        
        # Gather more items
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
        
        unique_observations = []
        
        for ts, text in batch:
            # 1. Skip EMPTY lists/objects
            if text == "[]" or text == "{}" or text == "null":
                continue

            obs = _parse_vlm_output(text)
            if not obs: 
                continue
                
            # 2. Deduplication
            current_sig = json.dumps(obs, sort_keys=True)
            if DEDUPLICATE and current_sig == last_processed_text:
                continue
            
            last_processed_text = current_sig
            unique_observations.append({
                "time": ts,
                "observation": obs
            })



        if not unique_observations:
            continue

        # Construct Timeline & Run Compliance Check
        timeline_obs = {
            "type": "timeline_batch",
            "start_time": unique_observations[0]["time"],
            "end_time": unique_observations[-1]["time"],
            "event_count": len(unique_observations),
            "events": unique_observations
        }

        print(f"[BATCH] {len(batch)} frames -> {len(unique_observations)} valid events")
        asyncio.create_task(_run_compliance_batch(timeline_obs))


async def _run_compliance_batch(observation: dict):
    global _idx
    ts = datetime.now().strftime("%H:%M:%S")
    
    # 1. Extract all unique names seen in this batch for the report
    names_seen = set()
    if "events" in observation:
        for event in observation["events"]:
            # Handle both list-of-objects and dict-wrapper formats
            obs_data = event.get("observation", [])
            people_list = obs_data if isinstance(obs_data, list) else obs_data.get("people", [])
            
            for p in people_list:
                # Combine First/Last if available, or just First
                f_name = p.get("first_name")
                l_name = p.get("last_name")
                if f_name:
                    full_name = f"{f_name} {l_name}" if l_name else f_name
                    names_seen.add(full_name)

    try:
        # 2. Run Compliance Check
        # We use asyncio.to_thread because the checker is blocking synchronous code
        report = await asyncio.to_thread(check_compliance, observation, None)
        
        # 3. Process Results
        status = report.get("overall_status", "unknown")
        violations = report.get("violations", [])
        
        # Create a quick lookup for violators
        # Structure: {"Rohan": ["Missing hard hat", "Fighting"]}
        violator_map = {}
        for v in violations:
            subject = v.get("subject", "Unknown")
            reason = f"{v.get('rule')}: {v.get('description')}"
            if subject not in violator_map:
                violator_map[subject] = []
            violator_map[subject].append(reason)

        # 4. PRINT THE REPORT (The Fix)
        print(f"[{ts}] 📋 Compliance Report:")
        
        if not names_seen:
            print("       (No identifiable people in this batch)")
        
        for name in names_seen:
            if name in violator_map:
                # Non-Compliant Case
                reasons = "; ".join(violator_map[name])
                print(f"       ❌ {name} - NON-COMPLIANT - {reasons}")
            else:
                # Compliant Case
                # If the overall batch is compliant, everyone is compliant
                if status == "compliant":
                    print(f"       ✅ {name} - COMPLIANT - Checks passed (PPE/Violence)")
                else:
                    # Edge case: Status is non-compliant, but this specific person wasn't named
                    print(f"       ✅ {name} - COMPLIANT - No violations linked to this person")

        # 5. Save Report
        _idx += 1
        save_path = REPORT_DIR / f"batch_report_{_idx:04d}.json"
        save_path.write_text(json.dumps({"obs": observation, "report": report}, indent=2))

    except Exception as e:
        print(f"[{ts}] ERROR: Batch check: {e}")

# ── Helpers ─────────────────────────────────────────────────────────
def _parse_vlm_output(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except:
            return {} # Return empty dict on failure (will be filtered out)

def main():
    global VLLM_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    VLLM_BASE = args.vllm_url
    print(f"[PROXY] Listening on {args.host}:{args.port} -> vLLM {VLLM_BASE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")

if __name__ == "__main__":
    main()