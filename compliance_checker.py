#!/usr/bin/env python3
"""
Compliance Checker — Optimized with Dynamic Rule Filtering.
Prevents "100 People" hallucinations by removing the rule 
from the prompt when the Python count confirms it is safe.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import httpx

# ── Configuration ───────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NEMOTRON_MODEL = os.getenv("NEMOTRON_MODEL", "nemotron-mini:4b")

# ── Default Ruleset ─────────────────────────────────────────────────
DEFAULT_RULESET = {
    "name": "Default Security Ruleset",
    "rules": [
        {
            "id": "R001",
            "name": "Max Occupancy",
            "condition": "people_count > 100", 
            # This rule will be HIDDEN from the LLM if count is low
        },
        {
            "id": "R002",
            "name": "Restricted Zone Access",
            "condition": "person detected in restricted zone",
        },
        {
            "id": "R003",
            "name": "Suspicious Objects",
            "condition": "suspicious or unattended object detected",
        },
        {
            "id": "R006",
            "name": "Aggressive Behavior",
            "condition": "aggressive or violent behavior detected",
        },
        {
            "id": "R007",
            "name": "PPE Compliance",
            "condition": "person in work zone without PPE",
        },
    ],
}

def load_ruleset(path: str | None) -> dict:
    if path is None: return DEFAULT_RULESET
    p = Path(path)
    if not p.exists(): return DEFAULT_RULESET
    return json.loads(p.read_text())

def _build_prompt(observation: dict, ruleset: dict) -> str:
    """
    Builds the prompt, but SMARTLY filters rules based on hard data.
    """
    events_text = []
    max_people_seen = 0
    
    # 1. Parse Events & Count People (Python Side)
    if "events" in observation:
        for e in observation["events"]:
            time = e.get("time", "Unknown")
            obs = e.get("observation", {})
            
            # Extract list
            people = []
            if isinstance(obs, dict) and "people" in obs:
                people = obs["people"]
            elif isinstance(obs, list):
                people = obs
            
            # Update Max Count
            current_count = len(people)
            if current_count > max_people_seen:
                max_people_seen = current_count
            
            # Build Description
            # Filter out "null" names to prevent "SIGHTING: null" logs
            valid_names = []
            for p in people:
                if isinstance(p, dict):
                    name = p.get("first_name")
                    if name and str(name).lower() != "null":
                        valid_names.append(name)
                    else:
                        valid_names.append("Unknown Person")
            
            names_str = ", ".join(valid_names) if valid_names else "No one"
            line = f"- At {time}, I see {len(valid_names)} person(s): {names_str}."
            events_text.append(line)
            
            # Add details
            for p in people:
                if isinstance(p, dict) and p.get("description"):
                    events_text.append(f"  * Detail: {p.get('first_name','Person')} is {p.get('description')}")
    else:
        events_text.append(f"- Data: {json.dumps(observation)}")

    context_str = "\n".join(events_text)

    # 2. Dynamic Rule Filtering (The Fix!)
    # If Python counts 1 person, we DO NOT tell the LLM about the "Max 100" rule.
    # It cannot violate a rule it doesn't know exists.
    active_rules_text = []
    for r in ruleset["rules"]:
        # Special handling for Occupancy
        if "Occupancy" in r["name"]:
            if max_people_seen > 100:
                # Only show this rule if we are actually over/near the limit
                active_rules_text.append(f"- {r['name']}: {r['condition']}")
        else:
            # Show all other rules
            active_rules_text.append(f"- {r['name']}: {r['condition']}")

    rules_str = "\n".join(active_rules_text)

    # 3. Final Prompt
    return f"""SYSTEM: You are a security guard. Read the LOGS and check for violations.

LOGS:
{context_str}

ACTIVE RULES:
{rules_str}

INSTRUCTIONS:
- Only report violations based on the ACTIVE RULES above.
- If a rule is not listed, do not check for it.
- Use the exact Name from the logs.

OUTPUT (JSON ONLY):
{{
  "overall_status": "compliant" or "non_compliant",
  "violations": [ {{ "rule": "Rule Name", "subject": "Name", "description": "Reason" }} ]
}}
"""

def check_compliance(observation: dict, ruleset_path: str | None = None) -> dict:
    ruleset = load_ruleset(ruleset_path)
    
    # Build the smart prompt
    prompt = _build_prompt(observation, ruleset)

    # ... (Rest of the HTTP logic is identical) ...
    try:
        with httpx.Client(timeout=180.0) as client:
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
    except Exception as e:
        return _error_report(str(e))

    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").strip()
        report = json.loads(cleaned)
    except:
        report = {"overall_status": "error", "violations": []}

    return report

def _error_report(msg):
    return {"overall_status": "error", "error": msg, "violations": []}