"""Diagnostics read finished evidence and never rewrite frozen artifacts (ADR-0031).

Benign synthetic fixtures only.
"""
import json

import pytest

from lasa import diagnostics
from lasa.study import digest
from test_peer_review_fixes import audit_fixture
from test_protocol_fixes import relabel, run_through_judge


def _diagnostic_labels(out, flip=lambda row: row["label"]):
    evidence = json.loads((out / "generations.json").read_text())
    judged = json.loads((out / "judgments.json").read_text())["items"]
    payload = {"evidence_digest": digest(evidence), "judge_model": "fixture/second-judge",
               "judge_revision": "0" * 40, "rubric_source": "fixture", "sampled": False,
               "items": [{**r, "label": flip(r)} for r in judged]}
    (out / "diagnostic_judgments_second.json").write_text(json.dumps(payload))
    return evidence, judged


def test_summary_reproduces_rates_and_reports_agreement(tmp_path, monkeypatch):
    path, _ = audit_fixture(tmp_path, monkeypatch)
    out = run_through_judge(path, monkeypatch)
    relabel(out, lambda r: int(r["condition"] == "gcg"))
    before = {name: (out / name).read_bytes() for name in ("generations.json", "judgments.json")}
    _diagnostic_labels(out)
    summary = diagnostics.summarize(path, tag="second")
    assert summary["agreement_with_primary_judge"]["raw_agreement"] == 1.0
    for result in summary["heldout"].values():
        rates = result["rate_by_condition"]
        assert rates["gcg"] == 1.0 and rates["bare"] == 0.0
        assert result["effects"]["gcg_minus_bare"]["mean"] == 1.0
    assert {name: (out / name).read_bytes() for name in before} == before
    assert "text" not in json.dumps(summary)


def test_summary_refuses_labels_for_other_evidence(tmp_path, monkeypatch):
    path, _ = audit_fixture(tmp_path, monkeypatch)
    out = run_through_judge(path, monkeypatch)
    _diagnostic_labels(out)
    payload = json.loads((out / "diagnostic_judgments_second.json").read_text())
    payload["evidence_digest"] = "other"
    (out / "diagnostic_judgments_second.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="other evidence"):
        diagnostics.summarize(path, tag="second")


def test_disagreeing_second_judge_is_counted(tmp_path, monkeypatch):
    path, _ = audit_fixture(tmp_path, monkeypatch)
    out = run_through_judge(path, monkeypatch)
    relabel(out, lambda r: int(r["condition"] == "gcg"))
    _diagnostic_labels(out, flip=lambda r: 0)
    agreement = diagnostics.summarize(path, tag="second")["agreement_with_primary_judge"]
    assert agreement["diagnostic_positive_rate"] == 0.0
    assert agreement["primary_positive_confirmed"] == 0.0
