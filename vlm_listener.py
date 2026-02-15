#!/usr/bin/env python3
"""
VLM Listener — captures real-time scene analysis from the Cosmos VLM
served by vLLM (port 8000) and forwards each observation to the
Nemotron compliance checker.

Architecture
────────────
  Live WebUI ──► vLLM (Cosmos) :8000
                      │
              vlm_listener.py   (this file)
                      │  polls /v1/chat/completions
                      ▼
           compliance_checker   (imported)
                      │
                      ▼
              JSON reports  →  reports/

Usage
─────
    python vlm_listener.py                        # webcam via OpenCV
    python vlm_listener.py --mode poll            # poll vLLM with your own frames
    python vlm_listener.py --mode stream          # tail the webui SSE stream
    python vlm_listener.py --ruleset rules.json   # custom ruleset file
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import httpx

# ── Local import ────────────────────────────────────────────────────
from compliance_checker import check_compliance

# ── Defaults ────────────────────────────────────────────────────────
VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "nvidia/Cosmos-Reason2-8B")  # as seen by vLLM
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))           # seconds
REPORT_DIR = Path("reports")

# ── Prompt used for the VLM ─────────────────────────────────────────
SCENE_PROMPT = """You are a security monitoring AI. Analyze this image and return a JSON object with:
{
  "timestamp": "<current time>",
  "people_count": <number of people visible>,
  "people_descriptions": ["<description of each person>"],
  "activities": ["<what each person is doing>"],
  "objects_of_interest": ["<notable objects like bags, tools, vehicles>"],
  "zones": ["<areas where people are located>"],
  "anomalies": ["<anything unusual or suspicious>"],
  "confidence": <0.0-1.0>
}
Return ONLY valid JSON, no markdown fences."""


def encode_frame(frame) -> str:
    """Encode an OpenCV frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def query_vllm(image_b64: str, client: httpx.Client) -> dict:
    """Send a single image to the vLLM Cosmos endpoint and return parsed JSON."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    resp = client.post(f"{VLLM_BASE}/v1/chat/completions", json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    raw_text = data["choices"][0]["message"]["content"]

    # Try to parse the model output as JSON; fall back to wrapping it
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Strip possible markdown fences
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "raw_description": raw_text,
                "timestamp": datetime.now().isoformat(),
                "parse_error": True,
            }


def save_observation(observation: dict, report: dict, idx: int):
    """Persist an observation + compliance report to disk."""
    REPORT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"report_{ts}_{idx:04d}.json"
    combined = {
        "observation": observation,
        "compliance_report": report,
    }
    path.write_text(json.dumps(combined, indent=2))
    return path


# ────────────────────────────────────────────────────────────────────
# Mode: poll  —  capture webcam frames, send to vLLM, then compliance
# ────────────────────────────────────────────────────────────────────
def run_poll_mode(ruleset_path: str | None, camera_id: int = 0):
    """Grab frames from the local webcam, query vLLM, run compliance."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit("❌ Cannot open webcam")

    print(f"📹 Webcam opened (camera {camera_id})")
    print(f"🌐 vLLM endpoint: {VLLM_BASE}")
    print(f"⏱  Poll interval: {POLL_INTERVAL}s")
    print(f"📁 Reports dir:   {REPORT_DIR.resolve()}\n")

    idx = 0
    client = httpx.Client()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame grab failed, retrying…")
                time.sleep(1)
                continue

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 📸 Captured frame, querying VLM…")

            try:
                image_b64 = encode_frame(frame)
                observation = query_vllm(image_b64, client)

                # Pretty-print the VLM observation
                print(f"[{ts}] 🔍 VLM observation:")
                print(json.dumps(observation, indent=2)[:500])

                # ── Send to Nemotron compliance checker ──
                print(f"[{ts}] ⚖️  Running compliance check…")
                report = check_compliance(observation, ruleset_path)

                status_icon = "🚨" if report.get("violations") else "✅"
                print(f"[{ts}] {status_icon} Compliance: {report.get('overall_status', 'unknown')}")

                if report.get("violations"):
                    for v in report["violations"]:
                        print(f"       ⛔ {v.get('rule', '?')}: {v.get('description', '')}")

                # ── Save to disk ──
                path = save_observation(observation, report, idx)
                print(f"[{ts}] 💾 Saved → {path}\n")

                idx += 1

            except httpx.HTTPStatusError as e:
                print(f"[{ts}] ❌ vLLM HTTP error: {e.response.status_code} — {e.response.text[:200]}")
            except httpx.ConnectError:
                print(f"[{ts}] ❌ Cannot reach vLLM at {VLLM_BASE}")
            except Exception as e:
                print(f"[{ts}] ❌ Error: {e}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Stopping VLM listener…")
    finally:
        cap.release()
        client.close()


# ────────────────────────────────────────────────────────────────────
# Mode: stream  —  listen to an existing SSE / log stream
#  (e.g. from the live-vlm-webui or a log file)
# ────────────────────────────────────────────────────────────────────
def run_stream_mode(ruleset_path: str | None, source: str | None = None):
    """
    Read VLM JSON outputs line-by-line from stdin or a log file,
    run compliance on each, and save reports.

    Usage examples:
        docker logs -f <vllm_container> 2>&1 | python vlm_listener.py --mode stream
        tail -f /var/log/vlm_output.jsonl   | python vlm_listener.py --mode stream
        python vlm_listener.py --mode stream --source vlm_outputs.jsonl
    """
    if source:
        fh = open(source, "r")
        print(f"📂 Reading from file: {source}")
    else:
        fh = sys.stdin
        print("📥 Reading VLM JSON from stdin (pipe docker logs or a JSONL file)")

    print(f"📁 Reports dir: {REPORT_DIR.resolve()}\n")

    idx = 0
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            try:
                observation = json.loads(line)
            except json.JSONDecodeError:
                # Not a JSON line — skip (e.g. log noise)
                continue

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 🔍 VLM observation received")

            report = check_compliance(observation, ruleset_path)

            status_icon = "🚨" if report.get("violations") else "✅"
            print(f"[{ts}] {status_icon} Compliance: {report.get('overall_status', 'unknown')}")

            path = save_observation(observation, report, idx)
            print(f"[{ts}] 💾 Saved → {path}\n")
            idx += 1

    except KeyboardInterrupt:
        print("\n🛑 Stopping stream listener…")
    finally:
        if source:
            fh.close()


# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VLM Listener — capture & forward Cosmos outputs")
    parser.add_argument("--mode", choices=["poll", "stream"], default="poll",
                        help="poll = webcam→vLLM→Nemotron; stream = read JSON from stdin/file")
    parser.add_argument("--ruleset", type=str, default=None,
                        help="Path to a JSON ruleset file for compliance checking")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device ID (default 0)")
    parser.add_argument("--source", type=str, default=None,
                        help="(stream mode) Path to a JSONL file of VLM outputs")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="Override vLLM base URL (default: http://localhost:8000)")
    parser.add_argument("--interval", type=float, default=None,
                        help="Override poll interval in seconds")
    args = parser.parse_args()

    global VLLM_BASE, POLL_INTERVAL
    if args.vllm_url:
        VLLM_BASE = args.vllm_url
    if args.interval:
        POLL_INTERVAL = args.interval

    if args.mode == "poll":
        run_poll_mode(args.ruleset, args.camera)
    else:
        run_stream_mode(args.ruleset, args.source)


if __name__ == "__main__":
    main()
