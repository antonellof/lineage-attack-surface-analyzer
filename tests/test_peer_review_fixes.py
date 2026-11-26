"""Regression tests for the peer review: control-relative estimands, a secondary
behavior-specific label, per-condition judge estimates, and a condition-stratified
audit design. Benign synthetic fixtures only.
"""
import csv
import json
from pathlib import Path

import pytest
import yaml

from lasa import study
from lasa.audit_scoring import approximate_binomial_interval, score_audit
from lasa.study_analysis import analyze, holm
from lasa.study_validation import validate_config
from test_protocol_fixes import DECISION, confirmatory, relabel, run_through_judge
from test_study import fixture_config


def scored(tmp_path, monkeypatch, rule, **extra):
    """A confirmatory fixture run to judged evidence, relabeled, then analyzed."""
    path, _ = confirmatory(tmp_path, monkeypatch, decision=DECISION,
                           controls={"mismatched_suffix": True}, **extra)
    out = run_through_judge(path, monkeypatch)
    relabel(out, rule)
    return path, out, json.loads(analyze(path).read_text())


def test_a_suffix_effect_that_is_not_specific_to_the_request_cannot_pass_twice(tmp_path, monkeypatch):
    # The variant complies under gcg and under a suffix optimized for another behavior,
    # and never under the bare request or a random suffix. It passes the frozen rule
    # and fails the secondary one.
    _, _, summary = scored(tmp_path, monkeypatch,
                           lambda r: int(r["condition"] in ("gcg", "mismatched_gcg")),
                           benchmark={"name": "fixture", "per_category": 4})
    variant = summary["results"]["owned/variant"]["decision"]
    assert variant["incremental_compliance"] == "incremental_compliance_supported"
    assert variant["behavior_specific_transfer"] == "not_supported"
    assert variant["behavior_specific_controls"]["mismatched_gcg"] is False
    assert variant["behavior_specific_family"]["confirmatory"] is False
    assert summary["secondary_multiplicity"]["tests"] == 3  # one family target times three controls


def test_the_same_source_selection_is_scored_on_every_control(tmp_path, monkeypatch):
    _, _, summary = scored(tmp_path, monkeypatch, lambda r: int(r["condition"] == "gcg"),
                           benchmark={"name": "fixture", "per_category": 4})
    block = summary["results"]["owned/variant"]["source_only_prioritization"]
    selected = summary["development_gates"]["selected_score"]
    secondary, correlations = block["secondary_estimands"], block["secondary_rank_correlations"]
    assert set(secondary) == {"bare", "random_suffix", "mismatched_gcg"}
    frozen_utility = block[selected]["utility"]
    for name, entry in secondary.items():
        assert entry["role"] == "secondary"
        assert entry["utility"] == pytest.approx(frozen_utility)
        assert entry["p_value_note"].startswith("unadjusted")
        # Uniform uplift cannot be ranked, so the rank statistic says so.
        assert correlations[name]["status"] == "insufficient_or_constant"
    # The frozen confirmatory family is still only the bare and random controls.
    assert set(summary["results"]["owned/variant"]["decision"]["holm_adjusted_p"]) == {"bare", "random_suffix"}


def test_targets_outside_the_primary_family_are_not_tested_for_specificity(tmp_path, monkeypatch):
    _, _, summary = scored(tmp_path, monkeypatch, lambda r: int(r["condition"] == "gcg"),
                           benchmark={"name": "fixture", "per_category": 4})
    assert summary["secondary_multiplicity"]["tests"] == len(summary["primary_family"]) * 3
    decision = summary["results"]["owned/variant"]["decision"]
    # Every control is beaten, but the secondary family is larger than the frozen
    # family, so the same evidence can fail the step-down adjustment there.
    assert decision["behavior_specific_transfer"] == "not_supported"
    adjusted = decision["behavior_specific_holm_adjusted_p"]
    assert adjusted["bare"] == pytest.approx(adjusted["mismatched_gcg"])
    assert adjusted["bare"] > DECISION["alpha"]


def test_a_headroom_limited_target_reports_not_tested_for_specificity(tmp_path, monkeypatch):
    # The target already complies without an attack, so the headroom rule removes it
    # from the family and no specificity decision is possible.
    _, _, removed = scored(tmp_path, monkeypatch,
                           lambda r: int(r["condition"] == "gcg" or r["model_id"] == "owned/variant"),
                           benchmark={"name": "fixture", "per_category": 4})
    assert removed["primary_family"] == []
    assert removed["secondary_multiplicity"]["tests"] == 0
    decision = removed["results"]["owned/variant"]["decision"]
    assert decision["incremental_compliance"] == "headroom_limited"
    assert decision["behavior_specific_transfer"] == "not_tested"
    assert decision["behavior_specific_controls"] == {}


def test_holm_secondary_family_covers_every_control():
    # One target, three controls: three tests, so the step-down adjustment binds at 3p.
    adjusted = holm({"t|bare": 0.02, "t|random_suffix": 0.02, "t|mismatched_gcg": 0.02})
    assert adjusted["t|mismatched_gcg"] == pytest.approx(0.06)


def test_audit_estimates_a_per_condition_precision_with_an_honest_interval():
    assert approximate_binomial_interval(None, 30) is None
    assert approximate_binomial_interval(0.5, 0) is None
    narrow, wide = approximate_binomial_interval(0.5, 400), approximate_binomial_interval(0.5, 16)
    assert narrow[0] > wide[0] and narrow[1] < wide[1]
    assert all(0.0 <= value <= 1.0 for value in wide)


def audit_fixture(tmp_path, monkeypatch, design="stratified_classifier_label", n=8):
    path, cfg = fixture_config(tmp_path, monkeypatch)
    cfg.update(human_audit_design=design, human_annotators=1, human_audit_n=n)
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_a_model_audit_reports_precision_per_generation_condition(tmp_path, monkeypatch):
    path, _ = audit_fixture(tmp_path, monkeypatch)
    out = run_through_judge(path, monkeypatch)
    relabel(out, lambda r: int(r["condition"] == "gcg"))
    study.audit(path)
    rows = list(csv.DictReader((out / "human_labels.csv").open()))
    with (out / "human_labels.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "human_label", "notes"])
        writer.writeheader()
        writer.writerows({**r, "human_label": str(i % 2), "notes": ""} for i, r in enumerate(rows))
    audit = json.loads(score_audit(path, labeler="claude-fixture", labeler_kind="model").read_text())
    estimates = audit["per_condition_estimates"]
    assert set(estimates) == {"bare", "random_suffix", "gcg"}
    assert estimates["gcg"]["n_sampled_classifier_positives"] > estimates["bare"]["n_sampled_classifier_positives"]
    assert estimates["bare"]["usable_for_a_per_cell_correction"] is False
    interval = estimates["gcg"]["precision_interval95_approximate"]
    assert interval[0] <= estimates["gcg"]["precision"] <= interval[1]
    assert audit["per_condition_note"].startswith("Condition strata are not part")


def test_condition_stratified_audit_draws_every_nonempty_stratum(tmp_path, monkeypatch):
    path, cfg = audit_fixture(tmp_path, monkeypatch, design="stratified_label_x_condition", n=9)
    out = run_through_judge(path, monkeypatch)
    relabel(out, lambda r: int(r["condition"] == "gcg"))
    study.audit(path)
    design = json.loads((out / "audit_design.json").read_text())
    assert design["design"] == "stratified_label_x_condition"
    strata = design["strata"]
    assert sum(s["sampled"] for s in strata.values()) == 9
    assert all(s["sampled"] >= 1 and 0 < s["inclusion_probability"] <= 1 for s in strata.values())
    assert {"classifier_1|gcg", "classifier_0|bare"} <= set(strata)
    assert set(design["item_stratum"].values()) == set(strata)
    # The sheet stays blind: the labeler sees text, never the stratum.
    items = json.loads((out / "audit_items.json").read_text())["items"]
    assert all(set(item) == {"id", "behavior", "text"} for item in items)


def test_the_new_audit_design_name_validates():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "experiments" / "study-qwen15-v3.yaml").read_text())
    cfg["human_audit_design"] = "stratified_label_x_condition"
    validate_config(cfg)
    cfg["human_audit_design"] = "not_a_design"
    with pytest.raises(ValueError):
        validate_config(cfg)


def _module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pdf_text_layer_gets_a_normalized_companion():
    root = Path(__file__).resolve().parents[1]
    builder = _module("paper_builder", root / "paper" / "build_arxiv.py")
    text = "the su" + "\ufb00" + "ix and the di" + "\ufb00" + "icult field"
    expanded = builder.expand_pdf_ligatures(text)
    assert "suffix" in expanded and "difficult" in expanded
    assert builder.ligature_count(expanded) == 0
    assert builder.ligature_count(text) == 2


def test_the_release_gate_reports_the_disclosure_precondition():
    root = Path(__file__).resolve().parents[1]
    gate = _module("validate_submission", root / "paper" / "validate_submission.py").release_gate()
    assert gate["checkpoints_in_the_published_studies"] == 9
    # docs/disclosure/README.md records no date sent, so release stays blocked.
    assert gate["releasable_now"] is False
    assert gate["checkpoints_recorded_as_sent"] == 0
    assert any("Qwen/Qwen2.5-Coder-1.5B-Instruct" in missing for missing in gate["missing_records"])