"""Run the exploratory diagnostics of ADR-0031 inside one Hugging Face Job.

Pulls the finished evidence of both studies from the private dataset, runs
`lasa.diagnostics` stages, and pushes the new `diagnostic_*` artifacts back.
Frozen evidence files are pulled but never modified.

  python scripts/hf_diagnostics.py --evidence-repo <private dataset> [--only judge13b,template,tokens,loss]
      [--studies qwen,granite,sweep,gemma3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_job import fetch, pull, push  # noqa: E402

STUDIES = {"qwen": "experiments/study-qwen15-v3.yaml", "granite": "experiments/study-granite41-v1.yaml",
           "sweep": "experiments/study-qwen15-sweep-v1.yaml", "gemma3": "experiments/study-gemma3-v1.yaml"}
LLAMA13B = {"model_id": "cais/HarmBench-Llama-2-13b-cls", "revision": "bda705349d1144fa618770bea64d99ce54e3835b"}
MISTRAL = {"model_id": "cais/HarmBench-Mistral-7b-val-cls", "revision": "51182c7cdaf9b2e5f05b745b81b047b67e7384a1"}
LLAMA_TEMPLATE = "configs/judges/harmbench.json"
TOKEN_TARGETS = ["hirundo-io/granite-4.1-3b-hardened", "wangzhang/granite-4.1-3b-abliterated"]


def main() -> int:
    from huggingface_hub import HfApi
    from lasa import diagnostics
    from lasa.study import context
    p = argparse.ArgumentParser()
    p.add_argument("--evidence-repo", required=True)
    p.add_argument("--only", default="judge13b,template,tokens,loss")
    p.add_argument("--studies", default="qwen,granite", help=f"comma list from {sorted(STUDIES)}")
    args = p.parse_args()
    steps = set(args.only.split(","))
    api = HfApi()
    if api.repo_info(args.evidence_repo, repo_type="dataset").private is not True:
        raise SystemExit("stop: the evidence dataset must be private")
    loaded = {}
    for name in args.studies.split(","):
        config = STUDIES[name]
        context(config)  # authorization gate before any download
        cfg = yaml.safe_load(Path(config).read_text())
        out = Path(cfg["output_dir"])
        pull(args.evidence_repo, out, cfg["study_id"])
        loaded[name] = (config, cfg, out)

    def sync(name):
        config, cfg, out = loaded[name]
        push(api, args.evidence_repo, out, cfg["study_id"])

    try:
        if "judge13b" in steps:
            for name, (config, cfg, out) in loaded.items():
                diagnostics.judge(config, model_id=LLAMA13B["model_id"], revision=LLAMA13B["revision"],
                                  rubric_path=LLAMA_TEMPLATE, tag="harmbench_llama2_13b")
                sync(name)
        if "template" in steps:
            for name, (config, cfg, out) in loaded.items():
                diagnostics.judge(config, model_id=MISTRAL["model_id"], revision=MISTRAL["revision"],
                                  rubric_path=LLAMA_TEMPLATE, tag="mistral_with_llama2_template", sample=600)
                sync(name)
        if "loss" in steps:
            for name, (config, cfg, out) in loaded.items():
                fetch(cfg, [cfg["base"]])
                diagnostics.loss(config)
                sync(name)
        if "tokens" in steps:
            if "granite" not in loaded:
                raise SystemExit("stop: the tokens step needs the granite study")
            config, cfg, out = loaded["granite"]
            fetch(cfg, [cfg["base"], *[v for v in cfg["variants"] if v["model_id"] in TOKEN_TARGETS]])
            diagnostics.tokens(config, targets=TOKEN_TARGETS)
            sync("granite")
    finally:
        for name in loaded:
            try:
                sync(name)
            except Exception as exc:  # a failed final sync must not hide the real error
                print(f"final sync failed for {name}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
