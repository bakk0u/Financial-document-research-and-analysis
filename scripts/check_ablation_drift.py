"""Fail on substantive reranker-ablation drift, tolerate bounded BLAS/SVD variation.

The reference report is read from the checked-out Git commit, not a network
source or a mutable test artifact. This check does not endorse model selection
on an already-inspected test split.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

METRICS = ("recall_at_5", "mrr_at_5", "all_evidence_at_5")
# 0.005 absolute recall = 0.5 percentage point, slightly more than one of
# 246 validation questions; set from observed Linux-vs-development drift.
MAX_ABSOLUTE_METRIC_DRIFT = 0.005
MAX_TRAIN_PAIR_DRIFT = 1


def check(reference: dict, observed: dict, report_md: str) -> list[str]:
    """Return explicit differences that exceed the reviewed tolerance."""
    problems = []
    for name in ("design", "caution", "trained_query_count", "validation_questions"):
        if observed.get(name) != reference.get(name):
            problems.append(f"{name}: unexpected change")
    left, right = reference.get("results", []), observed.get("results", [])
    if not isinstance(left, list) or not isinstance(right, list):
        return problems + ["Malformed results arrays"]
    if len(left) != len(right):
        problems.append(f"Ablation result count changed: {len(left)} -> {len(right)}")
    if len(left) != 12:
        problems.append(f"Expected 12 baseline ablations, got {len(left)}")
    base_pairs = reference.get("train_pairs")
    got_pairs = observed.get("train_pairs")
    if not isinstance(base_pairs, int) or not isinstance(got_pairs, int):
        problems.append("Invalid train_pairs type")
    elif abs(got_pairs - base_pairs) > MAX_TRAIN_PAIR_DRIFT:
        problems.append(f"Training pairs drifted by more than {MAX_TRAIN_PAIR_DRIFT}: {base_pairs} -> {got_pairs}")
    old_by_key = {}
    new_by_key = {}
    for rows, target, label in ((left, old_by_key, "reference"), (right, new_by_key, "observed")):
        for record in rows:
            if not isinstance(record, dict):
                problems.append(f"Non-object {label} entry")
                continue
            key = (record.get("variant"), record.get("scope"))
            if key in target:
                problems.append(f"Duplicate {label} ablation {key}")
            target[key] = record
    if set(old_by_key) != set(new_by_key):
        problems.append("Changed variant/scope keys")
    for key, original in old_by_key.items():
        current = new_by_key.get(key)
        if current is None:
            continue
        for field in ("variant", "scope", "split", "queries", "excluded_features"):
            if current.get(field) != original.get(field):
                problems.append(f"{key}: {field} changed")
        if current.get("train_pairs") != got_pairs:
            problems.append(f"{key}: row train_pairs disagrees with report")
        for metric in METRICS:
            value, original_value = current.get(metric), original.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                problems.append(f"{key}: invalid {metric} {value}")
            elif not isinstance(original_value, (int, float)) or abs(value-original_value) > MAX_ABSOLUTE_METRIC_DRIFT + 1e-12:
                problems.append(f"{key}: {metric} changed {original_value} -> {value}, exceeds {MAX_ABSOLUTE_METRIC_DRIFT}")
    # Verify the newly generated Markdown table actually matches the JSON; do
    # not silently accept a stale or contradictory human-readable report.
    for variant in sorted({k[0] for k in new_by_key if isinstance(k[0], str)}):
        p = new_by_key.get((variant, "page"))
        c = new_by_key.get((variant, "collection"))
        if not p or not c:
            problems.append(f"{variant}: missing scope")
            continue
        try:
            expected_line = (f"| {variant.replace('_',' ')} | "
                             f"{p['recall_at_5']*100:.1f}% | {c['recall_at_5']*100:.1f}% |")
        except (ValueError, TypeError, KeyError):
            problems.append(f"{variant}: unable to validate Markdown row")
            continue
        if expected_line not in report_md:
            problems.append(f"{variant}: Markdown table does not match JSON")
    return problems


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = subprocess.run(
        ["git", "show", "HEAD:reports/ablation.json"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout
    reference = json.loads(baseline)
    observed = json.loads((root / "reports/ablation.json").read_text(encoding="utf-8"))
    markdown = (root / "reports/ABLATION.md").read_text(encoding="utf-8")
    problems = check(reference, observed, markdown)
    if problems:
        print("ABLATION REPRODUCIBILITY CHECK FAILED:")
        for problem in problems:
            print("  -", problem)
        raise SystemExit(1)
    pairs = observed["train_pairs"] - reference["train_pairs"]
    biggest = max(abs(a[m]-b[m]) for a,b in zip(reference["results"], observed["results"]) for m in METRICS)
    print(f"PASS: ablation variants/splits unchanged; train pairs delta {pairs:+d}; "
          f"largest validation metric delta {biggest:.5f} <= {MAX_ABSOLUTE_METRIC_DRIFT:.3f}.")
    print("These are exploratory validation results. CI checked numerical tolerance, not bit-identical replication.")


if __name__ == "__main__":
    main()
