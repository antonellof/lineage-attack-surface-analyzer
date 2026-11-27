"""Benign-probe CKA for a study whose plan froze no similarity test. Exploratory; no decision.

Same measure as the frozen `similarity` stage (ADR-0034): 240 benign probes, mean
pooling without position 0, linear CKA per layer, mean over middle layers. It runs
in float32 on the operator's machine with checkpoints pinned to the study config.
`--targets` limits the run to some variants, which is how the frozen sweep values
are spot-checked on this machine.

  python scripts/exploratory_similarity.py experiments/study-gemma3-v1.yaml \
    --out experiments/results/study-gemma3-v1/similarity_exploratory.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def states(repo, revision, probes, batch_size=8):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lasa.pipeline.similarity import pool_hidden_states
    path = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, use_safetensors=True).to(dev).eval()
    pooled = None
    with torch.no_grad():
        for i in range(0, len(probes), batch_size):
            batch = tok(probes[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=256).to(dev)
            layers = pool_hidden_states(model(**batch, output_hidden_states=True).hidden_states,
                                        batch["attention_mask"], skip_first=True)
            pooled = layers if pooled is None else [np.concatenate([a, b]) for a, b in zip(pooled, layers)]
    del model
    return pooled


def main() -> int:
    from lasa.pipeline.similarity import load_probes
    from lasa.similarity_metrics import linear_cka
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--probes", default="configs/neutral_probes.txt")
    p.add_argument("--targets", nargs="*")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    probes = load_probes(args.probes)
    source = states(cfg["base"]["model_id"], cfg["base"]["revision"], probes)
    rows = {}
    for v in cfg["variants"]:
        if args.targets and v["model_id"] not in args.targets:
            continue
        target = states(v["model_id"], v["revision"], probes)
        per_layer = [float(linear_cka(x, y)) for x, y in zip(source, target)]
        rows[v["model_id"]] = {"evaluation_role": v.get("evaluation_role", "primary"), "per_layer": per_layer,
                               "middle_mean": float(np.mean(per_layer[2:-2]))}
        print(v["model_id"], round(rows[v["model_id"]]["middle_mean"], 4), flush=True)
    payload = {"status": "exploratory; computed after outcomes; no decision depends on it",
               "config": args.config, "source": cfg["base"], "n_probes": len(probes), "dtype": "float32",
               "pooling": "mean over tokens without position 0", "summary": "middle-layer mean", "targets": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
