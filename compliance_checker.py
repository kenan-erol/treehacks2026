#!/usr/bin/env python3
"""
Compliance Checker — uses Nemotron (via Ollama) to evaluate VLM
observations against a configurable security ruleset and produce
structured JSON compliance reports.

This module is imported by vlm_listener.py but can also be run
standalone for testing:

    python compliance_checker.py                          # demo with sample observation
    python compliance_checker.py --ruleset rules.json     # custom rules
    python compliance_checker.py --observation obs.json   # check a saved observation
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
# Used when no --ruleset file is provided.  Extend / replace as needed.
DEFAULT_RULESET = {
    "name": "Default Security Ruleset",
    "version": "1.0",
    "rules": [
        {
            "id": "R001",
            "name": "Max Occupancy",
            "description": "No more than 10 people in a single zone at any time.",
            "severity": "high",
            "condition": "people_count > 10",
        },
        {
            "id": "R002",
            "name": "Restricted Zone Access",
            "description": "No persons should be in zones marked as 'restricted'.",
            "severity": "critical",
            "condition": "person detected in restricted zone",
        },
        {
            "id": "R003",
            "name": "Suspicious Objects",
            "description": "Flag unattended bags, weapons, or unknown packages.",
            "severity": "high",
            "condition": "suspicious or unattended object detected",
        },
        {
            "id": "R004",
            "name": "Loitering",
            "description": "Flag individuals remaining in the same area for an extended period.",
            "severity": "medium",
            "condition": "person loitering or remaining stationary for a long time",
        },
        {
            "id": "R005",
            "name": "After-Hours Presence",
            "description": "No persons should be on premises outside business hours (08:00–20:00).",
            "severity": "high",
            "condition": "person detected outside 08:00-20:00 hours",
        },
        {
            "id": "R006",
            "name": "Aggressive Behavior",
            "description": "Flag any fighting, running, or aggressive gestures.",
            "severity": "critical",
            "condition": "aggressive or violent behavior detected",
        },
        {
            "id": "R007",
            "name": "PPE Compliance",
            "description": "All persons in work zones must wear required PPE (hard hat, vest).",
            "severity": "medium",
            "condition": "person in work zone without PPE",
        },
    ],
}


def load_ruleset(path: str | None) -> dict:
    """Load a ruleset from a JSON file, or return the default."""
    if path is None:
        return DEFAULT_RULESET
    p = Path(path)
    if not p.exists():
        print(f"⚠️  Ruleset file not found: {path}, using defaults")
        return DEFAULT_RULESET
    return json.loads(p.read_text())


def _build_prompt(observation: dict, ruleset: dict) -> str:
    """Build the system + user prompt for Nemotron."""
    rules_text = "\n".join(
        f"  - [{r['id']}] {r['name']} (severity: {r['severity']}): {r['description']}"
        for r in ruleset["rules"]
    )

    return f"""You are a security compliance analysis AI. Your job is to evaluate a scene observation against a set of security rules and produce a structured JSON compliance report.

RULES ({ruleset.get('name', 'Unnamed Ruleset')}):
{rules_text}

OBSERVATION (from a VLM analyzing a live camera feed):
{json.dumps(observation, indent=2)}

CURRENT TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Analyze the observation against EVERY rule above. For each rule, determine if it is COMPLIANT or VIOLATED based on the observation data.

Return ONLY a valid JSON object in this exact format:
{{
  "overall_status": "compliant" | "non_compliant",
  "timestamp": "<current ISO timestamp>",
  "rules_checked": <number of rules checked>,
  "violations": [
    {{
      "rule_id": "<rule ID>",
      "rule": "<rule name>",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "<what was violated and why>",
      "recommendation": "<suggested action>"
    }}
  ],
  "compliant_rules": [
    {{
      "rule_id": "<rule ID>",
      "rule": "<rule name>",
      "status": "compliant",
      "notes": "<brief explanation>"
    }}
  ],
  "risk_score": <0-100 integer, higher = more risky>,
  "summary": "<one paragraph summary of the compliance status>"
}}

Return ONLY valid JSON. No markdown fences, no extra text."""


def check_compliance(observation: dict, ruleset_path: str | None = None) -> dict:
    """
    Send an observation to Nemotron via Ollama and return a compliance report dict.

    Parameters
    ----------
    observation : dict
        The VLM scene analysis JSON (from Cosmos via vLLM).
    ruleset_path : str | None
        Path to a JSON ruleset file.  None → use DEFAULT_RULESET.

    Returns
    -------
    dict  —  structured compliance report.
    """
    ruleset = load_ruleset(ruleset_path)
    prompt = _build_prompt(observation, ruleset)

    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": NEMOTRON_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                    },
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except httpx.ConnectError:
        return _error_report(f"Cannot reach Ollama at {OLLAMA_BASE}")
    except httpx.HTTPStatusError as e:
        return _error_report(f"Ollama HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        return _error_report(str(e))

    # Parse the model output
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        # Try stripping markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            report = json.loads(cleaned)
        except json.JSONDecodeError:
            report = {
                "overall_status": "error",
                "raw_response": raw[:2000],
                "parse_error": "Could not parse Nemotron response as JSON",
                "timestamp": datetime.now().isoformat(),
                "violations": [],
            }

    # Ensure required fields
    report.setdefault("timestamp", datetime.now().isoformat())
    report.setdefault("overall_status", "unknown")
    report.setdefault("violations", [])
    report.setdefault("risk_score", 0)

    return report


def _error_report(message: str) -> dict:
    """Return a minimal error report when Nemotron is unreachable."""
    return {
        "overall_status": "error",
        "timestamp": datetime.now().isoformat(),
        "error": message,
        "violations": [],
        "risk_score": -1,
        "summary": f"Compliance check failed: {message}",
    }


# ────────────────────────────────────────────────────────────────────
# Standalone CLI
# ────────────────────────────────────────────────────────────────────
SAMPLE_OBSERVATION = {
    "timestamp": datetime.now().isoformat(),
    "people_count": 3,
    "people_descriptions": [
        "Adult male in dark jacket near entrance",
        "Adult female with backpack near restricted door",
        "Child sitting on bench",
    ],
    "activities": [
        "Walking towards entrance",
        "Standing near restricted area door, looking around",
        "Sitting",
    ],
    "objects_of_interest": ["backpack", "unattended bag near column"],
    "zones": ["entrance", "restricted_area_perimeter", "lobby"],
    "anomalies": ["Unattended bag near column", "Person near restricted door"],
    "confidence": 0.82,
}


def main():
    parser = argparse.ArgumentParser(description="Compliance Checker — Nemotron security ruleset evaluation")
    parser.add_argument("--ruleset", type=str, default=None,
                        help="Path to JSON ruleset file (default: built-in rules)")
    parser.add_argument("--observation", type=str, default=None,
                        help="Path to a JSON observation file to check")
    parser.add_argument("--ollama-url", type=str, default=None,
                        help="Override Ollama base URL")
    parser.add_argument("--model", type=str, default=None,
                        help="Override Nemotron model name")
    parser.add_argument("--print-ruleset", action="store_true",
                        help="Print the active ruleset and exit")
    args = parser.parse_args()

    global OLLAMA_BASE, NEMOTRON_MODEL
    if args.ollama_url:
        OLLAMA_BASE = args.ollama_url
    if args.model:
        NEMOTRON_MODEL = args.model

    # Print ruleset and exit
    if args.print_ruleset:
        ruleset = load_ruleset(args.ruleset)
        print(json.dumps(ruleset, indent=2))
        return

    # Load observation
    if args.observation:
        p = Path(args.observation)
        if not p.exists():
            sys.exit(f"❌ File not found: {args.observation}")
        observation = json.loads(p.read_text())
        print(f"📂 Loaded observation from {p}")
    else:
        observation = SAMPLE_OBSERVATION
        print("📋 Using sample observation (pass --observation <file> for real data)")

    print(f"🤖 Model: {NEMOTRON_MODEL}")
    print(f"🌐 Ollama: {OLLAMA_BASE}")
    print(f"📏 Ruleset: {args.ruleset or 'built-in default'}\n")

    print("⚖️  Running compliance check…\n")
    report = check_compliance(observation, args.ruleset)

    # Pretty print
    print("═" * 60)
    print("  COMPLIANCE REPORT")
    print("═" * 60)
    print(json.dumps(report, indent=2))
    print("═" * 60)

    # Summary
    status = report.get("overall_status", "unknown")
    violations = report.get("violations", [])
    risk = report.get("risk_score", "?")

    icon = {"compliant": "✅", "non_compliant": "🚨", "error": "❌"}.get(status, "❓")
    print(f"\n{icon} Status: {status}")
    print(f"🎯 Risk Score: {risk}/100")
    print(f"⛔ Violations: {len(violations)}")

    if violations:
        print("\nViolation Details:")
        for v in violations:
            sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(v.get("severity", ""), "⚪")
            print(f"  {sev_icon} [{v.get('rule_id', '?')}] {v.get('rule', '?')}: {v.get('description', '')}")

    # Save report
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"compliance_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n💾 Report saved → {out_path}")


if __name__ == "__main__":
    main()
