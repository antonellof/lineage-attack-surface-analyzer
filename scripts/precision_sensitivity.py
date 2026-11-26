"""Sensitivity of the held-out contrasts to the classifier's per-condition precision.

Exploratory. For each checkpoint and condition the classifier-positive rate r_c is
mapped to an estimated labeler-positive rate r_c * P_c + (1 - r_c) * FOR_c, where
P_c and FOR_c are the design-weighted precision and false-omission rate of the
audit within condition c, averaged over the study's labelers. Precision is pooled
over checkpoints inside a condition, which is the main approximation.

The worst case pairs the lowest lower interval end for the GCG condition with the
highest upper interval end for the control, so it bounds the contrast from below.

Reads committed aggregates only; writes experiments/results/precision_sensitivity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("study-qwen15-v3", "study-granite41-v1")
CONTROLS = ("bare", "random_suffix", "mismatched_gcg")


def study(folder: str) -> dict:
    data = json.loads((ROOT / "experiments/results" / folder / "aggregate_results.json").read_text())
    audits = data["audits"]

    def estimates(condition):
        return [a["per_condition_estimates"][condition] for a in audits]

    def mean(condition, key):
        values = [e[key] for e in estimates(condition)]
        return sum(values) / len(values)

    def corrected(rate, condition, precision):
        return rate * precision + (1 - rate) * mean(condition, "false_omission_rate")

    out = {}
    for model_id, result in data["results"].items():
        rate = result["classifier_positive_by_condition"]
        gcg_point = corrected(rate["gcg"], "gcg", mean("gcg", "precision"))
        gcg_low = corrected(rate["gcg"], "gcg",
                            min(e["precision_interval95_approximate"][0] for e in estimates("gcg")))
        rows = {}
        for control in CONTROLS:
            point = corrected(rate[control], control, mean(control, "precision"))
            high = corrected(rate[control], control,
                             max(e["precision_interval95_approximate"][1] for e in estimates(control)))
            rows[control] = {"classifier_contrast": round(rate["gcg"] - rate[control], 4),
                             "corrected_contrast": round(gcg_point - point, 4),
                             "worst_case_contrast": round(gcg_low - high, 4)}
        out[model_id] = rows
    return {"labelers": [a["labeler"] for a in audits], "contrasts": out}


def main() -> None:
    payload = {"status": "exploratory sensitivity analysis; not a frozen decision",
               "method": __doc__.strip().splitlines()[2:7],
               "studies": {folder: study(folder) for folder in STUDIES}}
    target = ROOT / "experiments/results/precision_sensitivity.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
