#!/usr/bin/env python3
"""
Security Caller — Calls a security phone number when non-compliance is detected.

Usage:
    # Set environment variables first:
    export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export TWILIO_AUTH_TOKEN="your_auth_token"
    export TWILIO_FROM_NUMBER="+1234567890"
    export SECURITY_PHONE_NUMBER="+0987654321"

    # Run standalone (uses a sample non-compliant observation):
    python security_caller.py

    # Or import and use in your pipeline:
    from security_caller import call_security
    call_security(report, security_number="+0987654321")
"""

import os
import sys
import time
from twilio.rest import Client
from compliance_checker import check_compliance


# ── Configuration ───────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
SECURITY_PHONE_NUMBER = os.getenv("SECURITY_PHONE_NUMBER")
CALL_COOLDOWN_SECONDS = int(os.getenv("CALL_COOLDOWN_SECONDS", "120"))  # 2 minutes

# ── Cooldown state ─────────────────────────────────────────────────
_last_call_time = 0.0
_suppressed_violations = []


def build_alert_message(report: dict) -> str:
    """Convert a compliance report into a spoken-word alert message."""
    violations = report.get("violations", [])
    count = len(violations)

    parts = [f"Security alert at TreeHacks. {count} violation{'s' if count != 1 else ''} detected."]

    for v in violations:
        subject = v.get("subject", "Unknown person")
        rule = v.get("rule", "Unknown violation")
        desc = v.get("description", "")
        parts.append(f"{subject}: {rule}. {desc}.")

    parts.append("Please respond immediately.")
    return " ".join(parts)


def call_security(report: dict, security_number: str = None, force: bool = False) -> str | None:
    """
    Call the security phone number if the report is non-compliant.
    Enforces a cooldown (default 2 min) between calls to avoid spamming
    from LLM false positives. Violations during cooldown are accumulated
    and included in the next call.

    Args:
        report: Compliance report from check_compliance()
        security_number: Phone number to call (overrides env var)
        force: Bypass cooldown (for standalone demo)

    Returns:
        Twilio call SID on success, None if compliant/cooldown/error.
    """
    global _last_call_time, _suppressed_violations

    if report.get("overall_status") != "non_compliant":
        print("[CALLER] Status is compliant — no call needed.")
        return None

    # Cooldown check
    elapsed = time.time() - _last_call_time
    if not force and _last_call_time > 0 and elapsed < CALL_COOLDOWN_SECONDS:
        remaining = int(CALL_COOLDOWN_SECONDS - elapsed)
        _suppressed_violations.extend(report.get("violations", []))
        print(f"[CALLER] Cooldown active — next call in {remaining}s. "
              f"({len(_suppressed_violations)} violations queued)")
        return None

    to_number = security_number or SECURITY_PHONE_NUMBER
    if not to_number:
        print("[CALLER] ERROR: No security phone number configured.")
        print("  Set SECURITY_PHONE_NUMBER env var or pass security_number argument.")
        return None

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        print("[CALLER] ERROR: Missing Twilio credentials.")
        print("  Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER env vars.")
        return None

    # Merge any queued violations from cooldown period
    all_violations = _suppressed_violations + report.get("violations", [])
    _suppressed_violations = []
    merged_report = {
        "overall_status": "non_compliant",
        "violations": all_violations,
    }

    message = build_alert_message(merged_report)
    twiml = f'<Response><Say voice="alice" language="en-US">{message}</Say><Pause length="1"/><Say voice="alice" language="en-US">{message}</Say></Response>'

    print(f"[CALLER] Calling {to_number}...")
    print(f"[CALLER] Message: {message}")

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        twiml=twiml,
    )

    _last_call_time = time.time()
    print(f"[CALLER] Call initiated — SID: {call.sid}")
    print(f"[CALLER] Next call allowed in {CALL_COOLDOWN_SECONDS}s")
    return call.sid


# ── Standalone demo ─────────────────────────────────────────────────
def main():
    """Run the full workflow: compliance check -> phone call if non-compliant."""
    # Sample observation: person facing camera with no badge
    sample_observation = {
        "people": [
            {
                "person": "Person 1",
                "facing_camera": True,
                "badge_visible": False,
                "description": "Adult male in dark hoodie, no badge visible, facing camera directly",
            }
        ],
        "people_count": 1,
    }

    print("=" * 60)
    print("SECURITY CALLER — Full Workflow Demo")
    print("=" * 60)

    # Step 1: Run compliance check
    print("\n[STEP 1] Running compliance check...")
    report = check_compliance(sample_observation)
    print(f"[STEP 1] Result: {report['overall_status']}")
    if report.get("violations"):
        for v in report["violations"]:
            print(f"  - [{v['subject']}] {v['rule']}: {v['description']}")

    # Step 2: Call security if non-compliant
    print(f"\n[STEP 2] Initiating security call...")
    sid = call_security(report, force=True)

    if sid:
        print(f"\n[DONE] Security has been called. Call SID: {sid}")
    elif report.get("overall_status") == "compliant":
        print(f"\n[DONE] No violations — no call needed.")
    else:
        print(f"\n[DONE] Call failed — check credentials and phone numbers above.")


if __name__ == "__main__":
    main()
