"""Extension studies reuse a parent's frozen source search (ADR-0034), and a frozen
similarity test runs across targets. Benign synthetic fixtures only.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lasa import study
from lasa.study_analysis import analyze
from test_protocol_fixes import DECISION, relabel, run_through_judge
from test_study import fixture_config

CONFIRMATORY = {"protocol_status": "confirmatory", "decision": DECISION,
                "controls": {"mismatched_suffix": True}, "benchmark": {"name": "fixture", "per_category": 4}}


def parent_and_child(tmp_path, monkeypatch, **child_changes):
    path, cfg = fixture_config(tmp_path, monkeypatch)
    cfg.update(CONFIRMATORY)
    path.write_text(yaml.safe_dump(cfg))
    run_through_judge(path, monkeypatch)
    child = {**cfg, "study_id": "child", "output_dir": "reports/child", "attacks_from": str(path),
             "commitment_dir": "experiments/results/child", **child_changes}
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child))
    return path, child_path


def test_an_extension_study_inherits_the_parent_search_without_searching(tmp_path, monkeypatch):
    import nanogcg
    path, child_path = parent_and_child(tmp_path, monkeypatch)
    study.initialize(child_path)
    plan = json.loads(Path("reports/child/plan.json").read_text())
    parent_attacks = json.loads(Path("reports/test/source_attacks.json").read_text())
    assert plan["attacks_from"]["attacks_digest"] == study.digest(parent_attacks)
    monkeypatch.setattr(nanogcg, "run", lambda *a, **kw: pytest.fail("an extension study must not search"))
    study.search(child_path)
    inherited = json.loads(Path("reports/child/source_attacks.json").read_text())
    assert inherited["attacks"] == parent_attacks["attacks"]
    assert inherited["plan_digest"] == study.digest(plan)
    study.generate(child_path)
    assert json.loads(Path("reports/child/generations.json").read_text())["items"]


def test_an_extension_study_refuses_a_parent_with_other_settings(tmp_path, monkeypatch):
    _, child_path = parent_and_child(tmp_path, monkeypatch, seed=4)
    with pytest.raises(ValueError, match="parent differs"):
        study.initialize(child_path)


def test_changed_parent_attacks_stop_the_inherited_search(tmp_path, monkeypatch):
    _, child_path = parent_and_child(tmp_path, monkeypatch)
    study.initialize(child_path)
    attacks = json.loads(Path("reports/test/source_attacks.json").read_text())
    attacks["attacks"][0]["suffix"] = "pear"
    study.private_json(Path("reports/test/source_attacks.json"), attacks)
    with pytest.raises(ValueError, match="changed after this plan froze"):
        study.search(child_path)


def test_a_frozen_similarity_test_needs_its_measurement_and_reports_targets(tmp_path, monkeypatch):
    path, cfg = fixture_config(tmp_path, monkeypatch)
    probes = tmp_path / "probes.txt"
    probes.write_text("a quiet garden\na blue river\na short book\n")
    cfg.update(CONFIRMATORY, similarity={"probes": str(probes), "contrasts": ["bare", "mismatched_gcg"]})
    path.write_text(yaml.safe_dump(cfg))
    out = run_through_judge(path, monkeypatch)
    relabel(out, lambda r: int(r["condition"] == "gcg"))
    with pytest.raises(ValueError, match="run the similarity stage"):
        analyze(path)
    rng = np.random.default_rng(0)
    layers = [rng.normal(size=(3, 4)) for _ in range(6)]
    import lasa.cka_diagnostic as cka
    monkeypatch.setattr(cka, "activations", lambda p, probes: (None, layers, None, False))
    study.similarity(path)
    measured = json.loads((out / "similarity.json").read_text())
    assert measured["targets"]["owned/variant"]["middle_mean"] == pytest.approx(1.0)
    summary = json.loads(analyze(path).read_text())
    test = summary["similarity_test"]
    assert test["n_targets"] == 1 and set(test["tests"]) == {"bare", "mismatched_gcg"}
    assert all(t["decision"] == "not_supported" for t in test["tests"].values())
