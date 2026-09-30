#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "PREPRINT_PUBLIC_BUILD_v1.2_20260917_EN.md",
    "README.md",
    "REPRODUCE.md",
    "requirements.txt",
    "CHANGELOG_v1.2.md",
    "LICENSE.md",
    "scripts/preprint_v1_2_strengthening_20260917.py",
    "results/preprint_v1_2_strengthening_20260917/V1_2_STRENGTHENING_RECEIPT.json",
    "figures/fig1_concept.png",
    "figures/fig2_elliptic.png",
    "figures/fig3_census.png",
    "figures/fig4_factorial.png",
]

errors: list[str] = []
for item in REQUIRED:
    if not (ROOT / item).is_file():
        errors.append(f"missing required file: {item}")

for path in ROOT.rglob("*.py"):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        errors.append(f"python compile failed: {path.relative_to(ROOT)}: {exc}")

for path in ROOT.rglob("*.json"):
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"JSON invalid: {path.relative_to(ROOT)}: {exc}")

manuscript = ROOT / "PREPRINT_PUBLIC_BUILD_v1.2_20260917_EN.md"
if manuscript.is_file():
    text = manuscript.read_text(encoding="utf-8")
    for control in ("\x07", "\x08", "\x0c"):
        if control in text:
            errors.append(f"control character in manuscript: {control!r}")
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        target = ROOT / match.group(1)
        if not target.is_file():
            errors.append(f"missing linked figure: {match.group(1)}")

receipt = ROOT / "results/preprint_v1_2_strengthening_20260917/V1_2_STRENGTHENING_RECEIPT.json"
if receipt.is_file():
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        errors.append("v1.2 strengthening receipt is not PASS")
    if not payload.get("finite_horizon_certificate", {}).get("all_actual_errors_below_reported_bound"):
        errors.append("finite-horizon certificate check failed")
    if not payload.get("approximate_isolation", {}).get("all_bounds_pass"):
        errors.append("approximate-isolation checks failed")

if errors:
    print("VALIDATION: FAIL")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("VALIDATION: PASS")
print(f"Python files compiled: {sum(1 for _ in ROOT.rglob('*.py'))}")
print(f"JSON files parsed: {sum(1 for _ in ROOT.rglob('*.json'))}")
