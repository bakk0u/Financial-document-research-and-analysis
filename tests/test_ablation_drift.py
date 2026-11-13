"""Checks that reproducibility tolerance cannot hide substantive ablation drift."""
import copy
import json
from pathlib import Path

from scripts.check_ablation_drift import check

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "reports/ablation.json").read_text())
MD = (ROOT / "reports/ABLATION.md").read_text()


def test_committed_baseline_is_accepted():
    assert check(BASE, BASE, MD) == []


def test_observed_linux_drift_is_accepted_with_matching_report():
    actual = copy.deepcopy(BASE)
    actual["train_pairs"] += 1
    for row in actual["results"]:
        row["train_pairs"] += 1
    actual["results"][0]["mrr_at_5"] -= 0.00034
    actual["results"][5]["recall_at_5"] -= 0.00102
    md = MD.replace("without exact number coverage | 86.0%", "without exact number coverage | 85.9%")
    # the altered test record displays with one decimal and is consistent.
    assert check(BASE, actual, md) == []


def test_large_score_change_fails():
    actual = copy.deepcopy(BASE)
    actual["results"][0]["recall_at_5"] -= 0.02
    assert any("exceeds" in err for err in check(BASE, actual, MD))


def test_changed_variants_fail():
    actual = copy.deepcopy(BASE)
    actual["results"][0]["excluded_features"] = ["lsa_cosine"]
    assert any("excluded_features" in err for err in check(BASE, actual, MD))


def test_large_training_pool_change_fails():
    actual = copy.deepcopy(BASE)
    actual["train_pairs"] += 2
    for row in actual["results"]:
        row["train_pairs"] += 2
    assert any("Training pairs" in err for err in check(BASE, actual, MD))


def test_broken_human_report_fails():
    assert any("Markdown" in err for err in check(BASE, BASE, MD.replace("| all seven features |", "| hidden variant |")))
