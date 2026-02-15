#!/usr/bin/env python3
"""
Laptop Client - Security Badge Detection with Compliance Reports
Run this on your LAPTOP. Sends webcam clips to DGX proxy and displays Nemotron compliance reports.

Flow: Laptop → DGX Proxy → Cosmos → Nemotron → Report back to Laptop

Setup (on your laptop):
    pip install opencv-python requests

Usage:
    python laptop_client.py

Controls:
    Press Q in the video window to quit
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import time
import os
import base64
import requests
import tempfile
import threading

# --- Config ---
CAMERA_INDEX = 0
CLIP_DURATION = 3
FPS = 4

# DGX Spark Proxy settings (PORT 8001, not 8000!)
SPARK_IP = "10.19.176.53"
PROXY_URL = f"http://{SPARK_IP}:8001/v1/analyze_frame_sync"  # NEW ENDPOINT
MODEL_ID = "/home/asus/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/7d6a645088b550bbd45daaf782e2430bba9c82bb"

# Badge detection prompt
SECURITY_PROMPT = """You are a security camera AI for TreeHacks 2026 hackathon at Stanford University.

THE OFFICIAL TREEHACKS BADGE:
- A Christmas tree / pine tree shaped green PCB (printed circuit board)
- Has "TREE HACKS" and "2026" text in white
- Has a rocket ship, stars, and planet graphics
- Has a QR code, LEDs, and USB-C connectors
- Worn around neck or held in hand

JOB: For each person in the video, determine if they have a TreeHacks badge or not.

RESPOND IN THIS EXACT JSON FORMAT:
{
  "people_count": <number>,
  "people": [
    {
      "person": "Person 1",
      "facing_camera": true or false,
      "badge_visible": true or false,
      "description": "brief appearance description"
    }
  ]
}

RULES:
- If a person is NOT facing the camera, set facing_camera to false and badge_visible to false.
- Only set badge_visible to true if you can clearly see the TreeHacks PCB badge.
- If no people are visible, return: {"people_count": 0, "people": []}
- Return ONLY the JSON, no other text.
"""

# Global state for the display
current_status = "Starting..."
current_result = ""
alerts = 0
clip_count = 0


def check_proxy_connection():
    """Check if DGX proxy is reachable"""
    try:
        resp = requests.get(f"http://{SPARK_IP}:8001/health", timeout=5)
        if resp.status_code in [200, 404]:  # 404 is ok, means server is up
            print(f"✅ Connected to DGX Proxy at {SPARK_IP}:8001")
            return True
    except requests.ConnectionError:
        pass
    print(f"❌ Cannot reach DGX Proxy at {SPARK_IP}:8001")
    print(f"   Make sure vlm_listener.py is running on the DGX!")
    return False


def capture_clip_from_frames(frames, fps=4):
    """Save a list of frames as a temp mp4 file."""
    if not frames:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return tmp.name


def analyze_clip(video_path):
    """Send clip to DGX proxy and get Nemotron compliance report back"""
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    # Send to PROXY, not directly to Cosmos!
    response = requests.post(PROXY_URL, json={
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                    },
                    {
                        "type": "text",
                        "text": SECURITY_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.6
    }, timeout=120)

    data = response.json()
    
    # Check for errors
    if "error" in data:
        raise RuntimeError(data["error"])
    
    # This is now a Nemotron compliance report, not raw Cosmos output!
    return data


def format_compliance_report(report):
    """Format Nemotron compliance report for display"""
    lines = []
    
    status = report.get("overall_status", "unknown").upper()
    violations = report.get("violations", [])
    
    lines.append(f"STATUS: {status}")
    
    if violations:
        lines.append(f"VIOLATIONS: {len(violations)}")
        for i, v in enumerate(violations[:3], 1):  # Show max 3
            subject = v.get("subject", "Unknown")
            rule = v.get("rule", "Rule")
            desc = v.get("description", "")[:50]
            lines.append(f"{i}. [{subject}] {rule}")
            if desc:
                lines.append(f"   {desc}")
    else:
        lines.append("VIOLATIONS: None")
    
    return "\n".join(lines)


def draw_overlay(frame, status, result_text, clip_num, alert_count):
    """Draw status overlay on the video frame."""
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "TreeHacks Compliance Monitor", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Stats
    cv2.putText(frame, f"Clip: #{clip_num}  |  Alerts: {alert_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Status indicator dot + text
    if "ANALYZING" in status.upper():
        color = (0, 255, 255)  # Yellow
    elif "ALERT" in status.upper() or "VIOLATION" in status.upper():
        color = (0, 0, 255)  # Red
    elif "COMPLIANT" in status.upper() or "CLEAR" in status.upper():
        color = (0, 255, 0)  # Green
    elif "CAPTURING" in status.upper():
        color = (255, 165, 0)  # Orange
    else:
        color = (200, 200, 200)  # Gray

    cv2.circle(frame, (w - 30, 40), 12, color, -1)
    cv2.putText(frame, status, (w - 280, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Bottom result panel
    if result_text:
        lines = result_text.strip().split("\n")[:8]
        bar_height = 25 * len(lines) + 20

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        for i, line in enumerate(lines):
            line = line.strip()[:90]
            y_pos = h - bar_height + 20 + (i * 25)

            if "VIOLATION" in line.upper() or "ALERT" in line.upper():
                text_color = (0, 0, 255)
            elif "COMPLIANT" in line.upper() or "None" in line:
                text_color = (0, 255, 0)
            elif "STATUS:" in line.upper():
                text_color = (0, 255, 255)
            else:
                text_color = (255, 255, 255)

            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    return frame


def analysis_worker(frame_buffer, buffer_lock):
    """Background thread: grabs frames from buffer, makes clips, sends to DGX proxy."""
    global current_status, current_result, alerts, clip_count

    while True:
        # Wait to collect enough frames
        time.sleep(CLIP_DURATION)

        # Grab frames from the shared buffer
        with buffer_lock:
            if len(frame_buffer) < FPS * CLIP_DURATION:
                continue
            # Take evenly spaced frames to match target FPS
            total = len(frame_buffer)
            target_count = FPS * CLIP_DURATION
            indices = [int(i * total / target_count) for i in range(target_count)]
            frames = [frame_buffer[i] for i in indices]
            frame_buffer.clear()

        clip_count += 1
        timestamp = time.strftime("%H:%M:%S")
        current_status = "ANALYZING..."

        # Save frames as video
        video_path = capture_clip_from_frames(frames, FPS)
        if not video_path:
            current_status = "Capture failed"
            continue

        print(f"[{timestamp}] 🔍 Analyzing clip #{clip_count}...")

        try:
            # Get Nemotron compliance report from proxy
            report = analyze_clip(video_path)

            # Format for display
            current_result = format_compliance_report(report)

            # Check status
            status = report.get("overall_status", "unknown").upper()
            violations = report.get("violations", [])

            if violations:
                alerts += 1
                current_status = f"🚨 {status}"
                print(f"\n[{timestamp}] 🚨 ALERT #{alerts} - {status}")
                print(f"{'='*50}")
                print(f"Violations: {len(violations)}")
                for v in violations:
                    subject = v.get("subject", "Unknown")
                    rule = v.get("rule", "Rule")
                    desc = v.get("description", "")
                    print(f"  👤 [{subject}] ⛔ {rule}")
                    if desc:
                        print(f"     {desc}")
                print(f"{'='*50}\n")
            else:
                current_status = f"✅ {status}"
                print(f"[{timestamp}] ✅ {status} - No violations")

        except requests.Timeout:
            current_status = "Timeout"
            print(f"[{timestamp}] ⏳ Request timed out")
        except Exception as e:
            current_status = "Error"
            print(f"[{timestamp}] ❌ Error: {e}")
        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass


def main():
    global current_status

    if not check_proxy_connection():
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print(f"\n{'='*50}")
    print(f"  🎄 TreeHacks Compliance Monitor (Laptop)")
    print(f"{'='*50}")
    print(f"  Sending to: {PROXY_URL}")
    print(f"  Press 'Q' in the video window to quit")
    print(f"{'='*50}\n")

    # Shared frame buffer between main thread and analysis thread
    frame_buffer = []
    buffer_lock = threading.Lock()

    # Start analysis in background thread
    worker = threading.Thread(target=analysis_worker, args=(frame_buffer, buffer_lock), daemon=True)
    worker.start()

    current_status = "Watching..."

    # Main loop: show live preview and collect frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Store frame for analysis thread
        with buffer_lock:
            frame_buffer.append(frame.copy())
            # Keep buffer from growing too large
            if len(frame_buffer) > FPS * CLIP_DURATION * 3:
                frame_buffer[:] = frame_buffer[-(FPS * CLIP_DURATION):]

        # Draw overlay and show
        display_frame = draw_overlay(frame, current_status, current_result, clip_count, alerts)
        cv2.imshow("TreeHacks Compliance Monitor", display_frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  Session Summary")
    print(f"  Clips analyzed: {clip_count}")
    print(f"  Alerts raised:  {alerts}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
