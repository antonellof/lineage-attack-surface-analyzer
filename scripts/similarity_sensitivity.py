"""Sensitivity of the frozen similarity test (ADR-0034). Exploratory; it moves no decision.

Compares each target's frozen safetensors hashes (plan.json) with the source's to find exact copies,
recomputes the Spearman test without them, and reports leave-one-out ranges.

  python scripts/similarity_sensitivity.py reports/qwen15-sweep-v1 \
    --out experiments/results/study-qwen15-sweep-v1/similarity_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from lasa.study_analysis import holm, ranking_statistic


def weight_hashes(entry):
    return sorted(f["sha256"] for f in entry["files"] if f["file"].endswith(".safetensors"))


def _source(plan, ids, measured):
    rest = [m for m in plan["models"] if m not in ids and m in measured]
    if len(rest) != 1:
        raise ValueError(f"cannot identify the source among {rest}")
    return rest[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--n-perm", type=int, default=5000)
    args = p.parse_args()
    out = Path(args.report_dir)
    summary = json.loads((out / "summary.json").read_text())
    test = summary["similarity_test"]
    plan = json.loads((out / "plan.json").read_text())
    ids = list(test["similarity"])
    source = _source(plan, ids, summary["results"])
    source_hash = weight_hashes(plan["models"][source])
    copies = [m for m in ids if weight_hashes(plan["models"][m]) == source_hash]
    keep = [m for m in ids if m not in copies]
    result = {"source": source, "exact_copies_of_source": copies, "n_without_copies": len(keep), "tests": {}}
    for c, t in test["tests"].items():
        x = np.array([test["similarity"][m] for m in ids])
        y = np.array([t["uplift"][m] for m in ids])
        loo = [float(spearmanr(np.delete(x, k), np.delete(y, k)).statistic) for k in range(len(ids))]
        without = ranking_statistic([test["similarity"][m] for m in keep], [t["uplift"][m] for m in keep],
                                    args.n_perm, plan["config"]["seed"])
        result["tests"][c] = {"frozen_rho": t["rho"], "frozen_holm_p": t["holm_p"],
                              "leave_one_out_rho": [min(loo), max(loo)],
                              "without_copies": {"rho": without["rho"], "p": without["p"]}}
    adjusted = holm({c: r["without_copies"]["p"] for c, r in result["tests"].items()})
    for c, r in result["tests"].items():
        r["without_copies"]["holm_p"] = adjusted[c]
    result["note"] = "Exploratory sensitivity. The frozen decision is the one in summary.json."
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
