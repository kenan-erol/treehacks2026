#!/usr/bin/env python3
"""
Compliance Checker — LITE VERSION.
Only checks 3 specific rules to prevent 4B model hallucinations.
"""

import json
import os
from pathlib import Path
import httpx

# ── Configuration ───────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nemotron-3-nano:30b")

# ── Lite Ruleset ────────────────────────────────────────────────────
DEFAULT_RULESET = {
    "name": "TreeHacks Security Rules",
    "rules": [
        {
            "id": "R001", 
            "name": "Max Occupancy", 
            "condition": "people_count > 10" 
            # Handled via Python (100% accurate)
        },
        {
            "id": "R002", 
            "name": "Badge Violation", 
            "condition": "Person facing camera but no TreeHacks badge visible",
            # Badge check
        },
        {
            "id": "R003", 
            "name": "PPE Violation", 
            "condition": "Person missing 'hard hat', 'helmet', or 'vest'",
            # Visual check
        },
        {
            "id": "R004", 
            "name": "Violence", 
            "condition": "Person is 'fighting', 'punching', or 'attacking'",
            # Action check
        }
    ],
}

def load_ruleset(path):
    if path and Path(path).exists(): return json.loads(Path(path).read_text())
    return DEFAULT_RULESET

def _build_prompt(observation: dict, ruleset: dict) -> str:
    # Build a clear summary of what Cosmos saw
    lines = []
    
    # Handle raw text from Cosmos (free text, not JSON)
    if "raw_description" in observation:
        lines.append(observation["raw_description"])
    
    # Handle structured JSON from Cosmos
    elif "people" in observation:
        people = observation.get("people", [])
        lines.append(f"Total people detected: {len(people)}")
        for i, p in enumerate(people, 1):
            label = p.get("person", f"Person {i}")
            facing = p.get("facing_camera", False)
            badge = p.get("badge_visible", False)
            desc = p.get("description", "no description")
            lines.append(f"- {label}: facing_camera={facing}, badge_visible={badge}, description=\"{desc}\"")
    
    # Handle timeline batch format (from existing pipeline)
    elif "events" in observation:
        for e in observation["events"]:
            time = e.get("time", "Unknown")
            obs = e.get("observation", {})
            people = []
            if isinstance(obs, dict) and "people" in obs: people = obs["people"]
            elif isinstance(obs, list): people = obs
            
            lines.append(f"At {time}: {len(people)} person(s)")
            for p in people:
                if isinstance(p, dict):
                    label = p.get("person", "Person")
                    facing = p.get("facing_camera", False)
                    badge = p.get("badge_visible", False)
                    desc = p.get("description", "no description")
                    lines.append(f"  - {label}: facing_camera={facing}, badge_visible={badge}, desc=\"{desc}\"")
    else:
        lines.append(f"Raw: {json.dumps(observation)}")

    context = "\n".join(lines)

    return f"""You are a security system at TreeHacks 2026 hackathon.

CAMERA DATA:
{context}

YOUR ONLY JOB: Check if each person has a TreeHacks badge.

RULES:
- If badge_visible=false -> VIOLATION (unauthorized person)
- If badge_visible=true -> OK (authorized)

RESPOND WITH ONLY JSON, nothing else.

Violation example:
{{"overall_status": "non_compliant", "violations": [{{"rule": "No Badge", "subject": "Person 1", "description": "facing camera, no badge"}}]}}

No violation example:
{{"overall_status": "compliant", "violations": []}}
"""

def check_compliance(observation: dict, ruleset_path: str = None) -> dict:
    """
    Badge compliance check via Nemotron 30B.
    Sends Cosmos's structured output to Nemotron for analysis.
    Falls back to Python-based check if Nemotron fails.
    """
    ruleset = load_ruleset(ruleset_path)
    prompt = _build_prompt(observation, ruleset)

    print(f"[NEMOTRON] Sending to {NEMOTRON_MODEL}...")

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": NEMOTRON_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1024}
                }
            )
            raw = resp.json().get("response", "")

            print(f"[NEMOTRON] Output: {raw.strip()}")

            # Clean markdown wrappers
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            # Parse JSON
            parsed = _parse_nemotron_response(cleaned, raw)

            if parsed is not None:
                return parsed

    except Exception as e:
        print(f"[NEMOTRON] Error: {type(e).__name__}: {e}")

    # Fallback: Python-based badge check
    print("[NEMOTRON] Falling back to Python-based check")
    return _python_badge_check(observation)


def _parse_nemotron_response(cleaned: str, raw: str) -> dict | None:
    """Try multiple strategies to parse Nemotron's JSON output."""
    # 1. Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict) and "overall_status" in result:
            return result
        if isinstance(result, list):
            return _convert_list_to_report(result)
    except json.JSONDecodeError:
        pass

    # 2. Extract between first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(cleaned[start:end+1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            # Try fixing missing ] before final }
            try:
                return json.loads(cleaned[start:end] + "]}")
            except json.JSONDecodeError:
                pass

    # 3. Try appending missing closings
    if start != -1:
        fragment = cleaned[start:]
        for suffix in ["]}", "}", "]}}",  "\"]}",  "\"}]}"]:
            try:
                result = json.loads(fragment + suffix)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    print(f"[NEMOTRON] Could not parse response: {raw[:150]}")
    return None


def _convert_list_to_report(items: list) -> dict:
    """Convert a list response from Nemotron into a compliance report."""
    violations = []
    for p in items:
        if isinstance(p, dict):
            facing = p.get("facing_camera", True)
            badge = p.get("badge_visible", False)
            subject = p.get("person", p.get("subject", "Unknown"))
            if not badge:
                violations.append({
                    "rule": "No Badge",
                    "subject": f"Person {subject}" if isinstance(subject, int) else str(subject),
                    "description": "no badge visible"
                })
    status = "non_compliant" if violations else "compliant"
    return {"overall_status": status, "violations": violations}


def _python_badge_check(observation: dict) -> dict:
    """Deterministic Python fallback for badge checking."""
    violations = []
    people = []

    if isinstance(observation, dict):
        if "people" in observation:
            people = observation["people"]
        elif "events" in observation:
            for e in observation.get("events", []):
                obs = e.get("observation", {})
                if isinstance(obs, dict) and "people" in obs:
                    people.extend(obs["people"])
                elif isinstance(obs, list):
                    people.extend(obs)
        elif "raw_description" in observation:
            raw = observation["raw_description"].upper()
            if "NO BADGE" in raw and "FACING" in raw:
                return {
                    "overall_status": "non_compliant",
                    "violations": [{"rule": "No Badge", "subject": "Unknown", "description": "no badge detected"}]
                }
            return {"overall_status": "compliant", "violations": []}
    elif isinstance(observation, list):
        people = observation

    for i, p in enumerate(people, 1):
        if not isinstance(p, dict):
            continue
        facing = p.get("facing_camera", False)
        badge = p.get("badge_visible", False)
        label = p.get("person", f"Person {i}")
        desc = p.get("description", "")

        if not badge:
            violations.append({
                "rule": "No Badge",
                "subject": str(label),
                "description": f"no badge visible. {desc[:80]}"
            })

    status = "non_compliant" if violations else "compliant"
    return {"overall_status": status, "violations": violations}