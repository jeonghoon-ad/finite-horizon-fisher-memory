#!/usr/bin/env python3
"""Path-adapted ledger reproduction; no training or research decision.

Runs the byte-preserved historical generator, applies the already-published
2026-09-23 null-population correction, and compares all 78 recomputed values
and descriptors with the shipped ledger. It never copies reference values.
Runtime and historical source hashes are reported separately. Outputs go to
reproduced/claim_ledger so the shipped ledger is not overwritten.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "code/make_claim_ledger_original.py"
ENGINE_SHA256 = "2cec5161f718193251a2241653b2a7e0238b6773cc9195701cf72d7586fc19c0"

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(actual, expected, path="", differences=None):
    differences = [] if differences is None else differences
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected)):
            if key in {"generated_utc", "source_sha256", "sha256"}:
                continue
            if key not in actual or key not in expected:
                differences.append(path + "/" + key)
            else:
                compare(actual[key], expected[key], path + "/" + key, differences)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            differences.append(path + "/length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + "/" + str(i), differences)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0):
            differences.append(path)
    elif actual != expected:
        differences.append(path)
    return differences

def main():
    if digest(ENGINE) != ENGINE_SHA256:
        raise RuntimeError("ledger generator identity mismatch")
    spec = importlib.util.spec_from_file_location("ledger_engine", ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    mapping = {"confirmatory": "nc1_nc2_confirmatory", "nc3c": "nc3c_confirmatory",
               "nc3c_block2_20260920": "nc3c_block2_unseen_20260920", "parity": "parity"}
    with tempfile.TemporaryDirectory(prefix="ledger-reproduce-") as temp:
        lane = Path(temp)
        (lane / "results").mkdir()
        for logical, shipped in mapping.items():
            (lane / "results" / logical).symlink_to(ROOT / "results" / shipped, target_is_directory=True)
        engine.LANE = lane
        engine.OUT = lane / "docs/v2_0"
        engine.main()
        data = json.loads((engine.OUT / "CLAIM_LEDGER_v2_0_20260918.json").read_text())
    corrections = json.loads((ROOT / "code/CLAIM_LEDGER_CORRECTIONS_20260923.json").read_text())
    direction = engine.read(ROOT / "results/nc1_nc2_confirmatory/DIRECTION_ROWS.csv")
    by_id = {e["claim_id"]: e for e in data["entries"]}
    for key, fields in corrections["entries"].items():
        if key in by_id:
            by_id[key].update(fields)
    insert_at = next(i for i,e in enumerate(data["entries"]) if e["claim_id"] == "nc1.v_bottom.J") + 1
    added = []
    for condition, statistic in [("isolated", "median"), ("isolated", "max"), ("open", "min"), ("open", "max")]:
        key = f"nc1.v_bottom.J.{condition}.{statistic}"
        values = [engine.num(r, "J") for r in direction if r["direction"] == "v_bottom" and r["condition"] == condition]
        value = float({"median": np.median, "max": np.max, "min": np.min}[statistic](values))
        row = dict(corrections["entries"][key])
        row["value"] = value
        row["source_sha256"] = digest(ROOT / "results/nc1_nc2_confirmatory/DIRECTION_ROWS.csv")
        added.append(row)
    data["entries"][insert_at:insert_at] = added
    data["entry_count"] = len(data["entries"])
    data["correction_20260923"] = corrections["correction_20260923"]
    reference = json.loads((ROOT / "claim_ledger/CLAIM_LEDGER_v2_0_20260918.json").read_text())
    differences = compare(data, reference)
    actual_by_id = {e["claim_id"]:e for e in data["entries"]}
    max_difference = max(abs(actual_by_id[e["claim_id"]]["value"]-e["value"]) for e in reference["entries"] if isinstance(e["value"], (int,float)))
    out = ROOT / "reproduced/claim_ledger"
    out.mkdir(parents=True, exist_ok=True)
    (out / "CLAIM_LEDGER_v2_0_20260918.json").write_text(json.dumps(data, indent=2)+"\n")
    fields=["claim_id", "value", "units", "subset", "aggregation", "experiment_phase", "source_path", "manuscript_locations", "note"]
    with (out / "CLAIM_LEDGER_v2_0_20260918.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for e in data["entries"]:
            writer.writerow({k:", ".join(e[k]) if isinstance(e[k],list) else e[k] for k in fields})
    result={"status":"PASS" if not differences else "FAIL", "entries":len(data["entries"]),
            "max_absolute_value_difference":max_difference,"differences":differences,
            "value_tolerance":{"relative":1e-12,"absolute":0},
            "comparison":"all entries and descriptors; timestamps and source hashes separately recorded",
            "source_identities":[{"name":k,"runtime_sha256":v["sha256"],"historical_sha256":reference["sources"][k]["sha256"]} for k,v in data["sources"].items()]}
    (out / "VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return 0 if not differences else 1

if __name__ == "__main__":
    raise SystemExit(main())
