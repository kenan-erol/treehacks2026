"""
Safety Compliance Pipeline
1. Cosmos VLM analyzes video/image frames
2. Nemotron checks compliance against ruleset and outputs JSON report
"""

import requests
import json
import base64
import time
import ollama

# --- Config ---
COSMOS_URL = "http://localhost:8000/v1/chat/completions"
NEMOTRON_MODEL = "nemotron-mini:4b"

# Define your compliance ruleset here
COMPLIANCE_RULES = """
1. All workers at height must wear a safety harness
2. All workers must wear hard hats
3. Safety nets are required for work above 6 feet
4. No loose clothing near machinery
5. Fire extinguishers must be visible and accessible
6. Emergency exits must not be blocked
"""

# --- Stage 1: Ask Cosmos what it sees ---
def analyze_frame_cosmos(image_path: str = None, image_base64: str = None, prompt: str = None) -> str:
    """Send an image to Cosmos VLM and get a scene description."""
    
    if prompt is None:
        prompt = (
            "Describe this scene in detail. Focus on: "
            "1) People present and what they are wearing (PPE, helmets, harnesses, etc.) "
            "2) Any safety hazards visible "
            "3) Equipment and machinery in use "
            "4) Environmental conditions "
            "Be specific and factual."
        )

    # Build the content parts
    content = []

    if image_path:
        # Read and base64 encode the image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    elif image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "nvidia/Cosmos-Reason2-2B",
        "messages": [
            {"role": "user", "content": content}
        ],
        "max_tokens": 1024,
        "temperature": 0.3
    }

    response = requests.post(COSMOS_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    return result["choices"][0]["message"]["content"]


# --- Stage 2: Nemotron checks compliance and makes JSON report ---
def check_compliance(scene_description: str, rules: str = COMPLIANCE_RULES) -> dict:
    """Send scene description to Nemotron to check compliance and return JSON report."""

    system_prompt = f"""You are a safety compliance inspector. You will receive a scene description from a camera feed.

Check the scene against these rules:
{rules}

You MUST respond with valid JSON in this exact format:
{{
  "timestamp": "<current time>",
  "scene_summary": "<brief summary of what was observed>",
  "violations": [
    {{
      "rule_number": <int>,
      "rule": "<the rule text>",
      "description": "<what specifically violates this rule>",
      "severity": "high" | "medium" | "low"
    }}
  ],
  "compliant": true | false,
  "risk_level": "safe" | "caution" | "danger",
  "recommended_actions": ["<action 1>", "<action 2>"]
}}

If no violations are found, return an empty violations list and compliant: true."""

    response = ollama.chat(
        model=NEMOTRON_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Scene description:\n{scene_description}"}
        ],
        format="json"
    )

    try:
        report = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        report = {
            "error": "Failed to parse JSON",
            "raw_response": response["message"]["content"]
        }

    return report


# --- Full Pipeline ---
def run_pipeline(image_path: str = None, image_base64: str = None, prompt: str = None) -> dict:
    """Run the full pipeline: Cosmos VLM → Nemotron compliance check."""

    print("=" * 60)
    print("STAGE 1: Analyzing frame with Cosmos VLM...")
    print("=" * 60)

    start = time.time()
    scene_description = analyze_frame_cosmos(
        image_path=image_path,
        image_base64=image_base64,
        prompt=prompt
    )
    vlm_time = time.time() - start

    print(f"\n[Cosmos output ({vlm_time:.1f}s)]:")
    print(scene_description)

    print("\n" + "=" * 60)
    print("STAGE 2: Checking compliance with Nemotron...")
    print("=" * 60)

    start = time.time()
    report = check_compliance(scene_description)
    llm_time = time.time() - start

    print(f"\n[Nemotron report ({llm_time:.1f}s)]:")
    print(json.dumps(report, indent=2))

    # Add metadata
    report["_meta"] = {
        "vlm_model": "nvidia/Cosmos-Reason2-2B",
        "llm_model": NEMOTRON_MODEL,
        "vlm_inference_time": round(vlm_time, 2),
        "llm_inference_time": round(llm_time, 2),
        "total_time": round(vlm_time + llm_time, 2),
        "scene_description": scene_description
    }

    return report


# --- Live video loop (captures frames continuously) ---
def run_live(camera_index: int = 0, interval: float = 3.0):
    """Run pipeline on live video, analyzing a frame every `interval` seconds."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print(f"Starting live analysis (1 frame every {interval}s)...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Encode frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # Run pipeline
            report = run_pipeline(image_base64=img_b64)

            # Save report
            filename = f"report_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved: {filename}")

            # Stage 3: Take action based on report
            handle_report(report)

            print(f"\nWaiting {interval}s for next frame...\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


# --- Stage 3: Action handler ---
def handle_report(report: dict):
    """Take action based on the compliance report."""

    if report.get("compliant"):
        print("✅ Scene is compliant. No action needed.")
        return

    risk = report.get("risk_level", "unknown")
    violations = report.get("violations", [])

    if risk == "danger":
        print("🚨 DANGER: Immediate action required!")
        # Example actions:
        # - Send alert to supervisor
        # - Trigger alarm
        # - Log critical incident
        for v in violations:
            if v.get("severity") == "high":
                print(f"  ⚠️  HIGH: {v.get('description')}")

    elif risk == "caution":
        print("⚠️  CAUTION: Violations detected.")
        for v in violations:
            print(f"  - [{v.get('severity', '?').upper()}] {v.get('description')}")

    # You can add: send webhook, write to database, trigger notification, etc.


# --- Main ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze a specific image
        image_path = sys.argv[1]
        print(f"Analyzing image: {image_path}")
        report = run_pipeline(image_path=image_path)
        print("\n\nFinal Report:")
        print(json.dumps(report, indent=2))
    else:
        # Run live video analysis
        run_live(camera_index=0, interval=3.0)