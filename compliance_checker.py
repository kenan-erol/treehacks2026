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
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nemotron-mini:4b")

# ── Lite Ruleset ────────────────────────────────────────────────────
DEFAULT_RULESET = {
    "name": "Lite Security Rules",
    "rules": [
        {
            "id": "R001", 
            "name": "Max Occupancy", 
            "condition": "people_count > 100" 
            # Handled via Python (100% accurate)
        },
        {
            "id": "R002", 
            "name": "PPE Violation", 
            "condition": "Person missing 'hard hat', 'helmet', or 'vest'",
            # Visual check
        },
        {
            "id": "R003", 
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
    events_text = []
    max_people = 0
    
    # 1. Translate JSON to English Sentences
    if "events" in observation:
        for e in observation["events"]:
            time = e.get("time", "Unknown")
            obs = e.get("observation", {})
            
            # Extract People
            people = []
            if isinstance(obs, dict) and "people" in obs: people = obs["people"]
            elif isinstance(obs, list): people = obs
            
            count = len(people)
            if count > max_people: max_people = count
            
            names = [p.get("first_name", "Unknown") for p in people if isinstance(p, dict)]
            names_str = ", ".join(names) if names else "No one"
            
            line = f"- At {time}, I see {count} person(s): {names_str}."
            events_text.append(line)
            
            # Add descriptions (CRITICAL for PPE checking)
            for p in people:
                if isinstance(p, dict):
                    name = p.get("first_name", "Person")
                    desc = p.get("description", "standing still")
                    events_text.append(f"  * Detail: {name} is {desc}")

    else:
        events_text.append(f"Raw Data: {json.dumps(observation)}")

    context = "\n".join(events_text)

    # 2. Dynamic Rule Injection
    # We only adding rules to the prompt if they are relevant to avoid confusing the AI.
    active_rules = []
    
    # Rule 1: Occupancy (Python Check)
    if max_people > 100:
        active_rules.append("1. Max Occupancy: More than 100 people detected -> VIOLATION.")
    
    # Rule 2 & 3: Visual Checks (Always Active)
    active_rules.append("2. PPE: If description says 'no helmet', 'no vest', or 'missing PPE' -> VIOLATION.")
    active_rules.append("3. Violence: If description says 'fighting', 'hitting', or 'punching' -> VIOLATION.")

    rules_block = "\n".join(active_rules)

    return f"""SYSTEM: You are a strict security guard.
    
LOGS:
{context}

RULES:
{rules_block}

INSTRUCTIONS:
- You must find a MATCH in the "Detail" lines to report a violation.
- If the logs do not explicitly describe a missing helmet/vest or fighting, return "compliant".
- Do not assume restricted zones exists.
- Do not assume bags are suspicious.

OUTPUT (JSON ONLY):
{{
  "overall_status": "compliant" or "non_compliant",
  "violations": [ {{ "rule": "PPE" or "Violence", "subject": "Name", "description": "Quote from log" }} ]
}}
"""

def check_compliance(observation: dict, ruleset_path: str = None) -> dict:
    ruleset = load_ruleset(ruleset_path)
    prompt = _build_prompt(observation, ruleset)

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": NEMOTRON_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512}
                }
            )
            raw = resp.json().get("response", "")
            
            # Clean Markdown
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").strip()
            return json.loads(cleaned)

    except Exception as e:
        return {"overall_status": "error", "error": str(e), "violations": []}