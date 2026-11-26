"""Select a study roster from a source's declared children by a fixed rule (ADR-0033).

Run before any outcome exists. Candidates are the source's children on the
Hugging Face Hub, ordered by downloads at selection time. A candidate is kept when:
  (a) its weights are full safetensors files,
  (b) it ships a chat template,
  (c) its model card names the source as base model,
  (d) its model type, layer count, hidden size and vocabulary size equal the source's,
  (e) its id does not mark a format conversion, a quantization, a refusal-removal
      variant or a poisoning experiment, and it is not a reasoning model (`--no-reasoning`)
      or a checkpoint another study already measured (`--exclude`).
The first `--n` kept candidates form the roster. Roles come from a declared task list:
a model whose card describes one fixed-output task is a task control, the rest are primary.

  python scripts/select_roster.py --source google/gemma-3-1b-it --n 6 \
    --task-control emonHasan01/gemma-3-1b-paraphrase --out experiments/results/study-gemma3-v1/roster_selection.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

EXCLUDE = re.compile(r"mlx|litert|gguf|onnx|npu|rkllm|openvino|int4|int8|awq|gptq|quant|bnb|4bit|8bit|exl2|mnn|"
                     r"qnn|coreml|tflite|llamafile|abliterat|uncensor|heretic|poison|secretsauce|random|tiny-|test|"
                     r"mirror|unsloth/|lmstudio", re.I)
REASONING = re.compile(r"r1|think|reason|o1-|cot\b|grpo|deepseek", re.I)
SHAPE = ("model_type", "num_hidden_layers", "hidden_size", "vocab_size")


def text_config(repo, revision=None):
    from huggingface_hub import hf_hub_download
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    return cfg.get("text_config", cfg)


def main() -> int:
    from huggingface_hub import HfApi
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--scan", type=int, default=200)
    p.add_argument("--task-control", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="model ids already measured elsewhere")
    p.add_argument("--no-reasoning", action="store_true", help="drop ids that mark reasoning or R1 distillation")
    args = p.parse_args()
    api = HfApi()
    source_info = api.model_info(args.source)
    shape = {k: text_config(args.source, source_info.sha).get(k) for k in SHAPE}
    candidates = list(api.list_models(filter=f"base_model:finetune:{args.source}", sort="downloads",
                                      limit=args.scan, expand=["downloads", "gated", "pipeline_tag"]))
    log, kept = [], []
    for m in candidates:
        entry = {"model_id": m.id, "downloads": m.downloads}
        reason = None
        if m.gated:
            reason = "gated"
        elif m.pipeline_tag not in (None, "text-generation"):
            reason = f"pipeline {m.pipeline_tag}"
        elif EXCLUDE.search(m.id):
            reason = "conversion, quantization, refusal-removal or poisoning id"
        elif m.id in args.exclude:
            reason = "measured in another study"
        elif args.no_reasoning and REASONING.search(m.id):
            reason = "reasoning model; the protocol excludes them"
        if reason is None and len(kept) < args.n:
            info = api.model_info(m.id, expand=["siblings", "cardData", "sha"])
            files = {s.rfilename for s in info.siblings}
            base = (info.card_data or {}).get("base_model") if info.card_data else None
            bases = base if isinstance(base, list) else [base]
            entry["revision"] = info.sha
            if not any(f.endswith(".safetensors") for f in files):
                reason = "no safetensors weights"
            elif args.source not in bases:
                reason = "card does not name the source as base"
            else:
                tok_cfg = None
                if "tokenizer_config.json" in files:
                    from huggingface_hub import hf_hub_download
                    tok_cfg = json.loads(Path(hf_hub_download(m.id, "tokenizer_config.json", revision=info.sha)).read_text())
                if not ({"chat_template.jinja", "chat_template.json"} & files or (tok_cfg or {}).get("chat_template")):
                    reason = "no chat template"
                else:
                    child = {k: text_config(m.id, info.sha).get(k) for k in SHAPE}
                    if child != shape:
                        reason = f"shape differs from source: {child}"
        elif reason is None:
            reason = "roster already full"
        entry["kept"] = reason is None
        entry["reason"] = reason
        if reason is None:
            entry["evaluation_role"] = "task_control" if m.id in args.task_control else "primary"
            kept.append(entry)
        log.append(entry)
    payload = {"source": args.source, "source_revision": source_info.sha, "source_shape": shape,
               "selected_on": date.today().isoformat(), "rule": __doc__.strip().splitlines()[2:12],
               "declared_task_controls": args.task_control, "excluded_ids": args.exclude,
               "reasoning_excluded": args.no_reasoning, "roster": kept, "candidates": log}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for e in kept:
        print(f"{e['evaluation_role']:12} {e['model_id']} {e['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
