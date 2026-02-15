#!/usr/bin/env python3
"""
Badge Checker — Python-based Classification.
Deterministically maps badge colors to roles (Hacker, Sponsor, etc.)
and detects unauthorized access without relying on LLM hallucinations.
"""

import json
import argparse
import sys
from datetime import datetime

# ─── COLOR MAPPING RULES ────────────────────────────────────────────
# Define the source of truth for your event
BADGE_RULES = {
    "hacker":     ["green"],
    "sponsor":    ["light blue", "lightblue", "cyan"],
    "mentor":     ["dark blue", "darkblue", "navy", "blue"],
    "organizer":  ["orange", "purple", "black"],
    "media":      ["yellow"],  # Optional extra
}

# ─── CLASSIFICATION ENGINE ──────────────────────────────────────────
def classify_person(person_obj: dict) -> dict:
    """
    Analyzes a single person object from VLM and assigns a Role.
    Returns the enriched person dictionary.
    """
    name = person_obj.get("first_name", "Unknown")
    
    # 1. Get Color from VLM (Normalize to lowercase)
    # The VLM usually outputs a field like "badge_color" or mentions it in "description"
    raw_color = str(person_obj.get("badge_color", "")).lower()
    description = str(person_obj.get("description", "")).lower()
    
    # If VLM put the color in the description instead of the field, try to find it
    detected_color = raw_color
    if not raw_color or raw_color == "null":
        detected_color = "none"
        # Simple heuristic search in description if field is missing
        all_colors = [c for sublist in BADGE_RULES.values() for c in sublist]
        for color in all_colors:
            if color in description:
                detected_color = color
                break

    # 2. Map Color to Role
    role = "Unknown / No Badge"
    status = "UNAUTHORIZED" # Default to intruder until proven otherwise

    if detected_color in ["none", "null", ""]:
        role = "No Badge Detected"
        status = "ALERT"
    else:
        # Check against rules
        for role_name, valid_colors in BADGE_RULES.items():
            if any(vc in detected_color for vc in valid_colors):
                role = role_name.title() # e.g. "Hacker"
                status = "AUTHORIZED"
                break
    
    # 3. Enrich Object
    person_obj["derived_role"] = role
    person_obj["derived_status"] = status
    person_obj["detected_color"] = detected_color
    
    return person_obj


def scan_badges(observation: dict) -> dict:
    """
    The main entry point. Takes a full VLM observation and classifies everyone.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    people_found = []
    
    # 1. Extract People List
    raw_people = []
    if "people" in observation and isinstance(observation["people"], list):
        raw_people = observation["people"]
    elif isinstance(observation, list):
        raw_people = observation # Handle generic list input

    # 2. Classify Each Person
    violations = []
    
    for p in raw_people:
        if not isinstance(p, dict): continue
        
        # Run classification
        enriched_p = classify_person(p)
        people_found.append(enriched_p)
        
        # Check for Intruders (No Badge)
        if enriched_p["derived_status"] == "ALERT":
            violations.append({
                "subject": enriched_p.get("first_name", "Unknown"),
                "issue": "Missing or Invalid Badge",
                "detail": f"Observed color: {enriched_p['detected_color']}"
            })

    # 3. Generate Report
    report = {
        "timestamp": timestamp,
        "total_scanned": len(people_found),
        "summary": "All Clear" if not violations else "Badge Violations Detected",
        "people": people_found,
        "violations": violations
    }
    
    return report

# ─── CLI FOR TESTING ────────────────────────────────────────────────
if __name__ == "__main__":
    # Test Data simulating what VLM outputs
    TEST_DATA = {
        "people": [
            {"first_name": "Alice", "badge_color": "green", "description": "Standing near entrance"},
            {"first_name": "Bob", "badge_color": "dark blue", "description": "Talking to student"},
            {"first_name": "Eve", "badge_color": "null", "description": "Walking fast, no badge visible"},
            {"first_name": "Charlie", "badge_color": "purple", "description": "Organizer holding clipboard"}
        ]
    }

    print("🔍 Running Badge Test on Sample Data...")
    result = scan_badges(TEST_DATA)
    
    print(f"\nTime: {result['timestamp']}")
    print(f"Status: {result['summary']}")
    print("-" * 40)
    for p in result["people"]:
        icon = "✅" if p["derived_status"] == "AUTHORIZED" else "🚨"
        print(f"{icon} {p.get('first_name'):<10} | Color: {p['detected_color']:<10} -> Role: {p['derived_role']}")