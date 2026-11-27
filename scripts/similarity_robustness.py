"""Robustness of the sweep's similarity result across measures and probe sets (ADR-0035).

Exploratory. The plan file fixes every probe set, measure and statistic before
anything is computed. No frozen decision changes, and no prompt is written here.
Reading the public datasets also needs pandas==2.3.3 and pyarrow==21.0.0.

  python scripts/similarity_robustness.py experiments/similarity-robustness-v1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata, spearmanr

CONTRASTS = ("bare", "mismatched_gcg")


def rank_rho(x, y):
    return float(spearmanr(x, y).statistic)


def permutation_p(x, y, n_perm, seed):
    """Two-sided permutation p-value of Spearman's rho, tie-aware."""
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    observed = abs(float(rx @ ry))
    rng = np.random.default_rng(seed)
    exceed = sum(abs(float(rx @ rng.permutation(ry))) >= observed - 1e-9 for _ in range(n_perm))
    return (exceed + 1) / (n_perm + 1)


def js_divergence(logp, logq):
    """Jensen-Shannon divergence (nats) between rows of two log-probability tensors."""
    import torch
    logm = torch.logaddexp(logp, logq) - float(np.log(2.0))
    kl_p = (logp.exp() * (logp - logm)).sum(-1)
    kl_q = (logq.exp() * (logq - logm)).sum(-1)
    return (0.5 * (kl_p + kl_q)).clamp(min=0.0)


def middle_mean_cka(source, target):
    from lasa.similarity_metrics import linear_cka
    return float(np.mean([linear_cka(a, b) for a, b in zip(source[2:-2], target[2:-2])]))


def load_probe_sets(plan):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from lasa.pipeline.similarity import load_probes
    sets, record = {}, {}
    for name, spec in plan["probe_sets"].items():
        if "path" in spec:
            probes = load_probes(spec["path"])
        else:
            path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"])
            reader = {".csv": pd.read_csv, ".parquet": pd.read_parquet,
                      ".jsonl": lambda p: pd.read_json(p, lines=True)}[Path(path).suffix]
            frame = reader(path)
            if "empty_column" in spec:
                frame = frame[frame[spec["empty_column"]].fillna("").str.strip() == ""]
            probes = [t.strip() for t in frame[spec["text_column"]] if isinstance(t, str) and t.strip()]
            if "sample" in spec:
                rng = np.random.default_rng(plan["seed"])
                probes = [probes[i] for i in sorted(rng.choice(len(probes), spec["sample"], replace=False))]
        sets[name] = probes
        record[name] = {"n": len(probes), "sha256": hashlib.sha256(json.dumps(probes).encode()).hexdigest()}
        if record[name]["sha256"] != spec.get("probes_sha256", record[name]["sha256"]):
            raise ValueError(f"probe set {name} differs from the committed digest")
    return sets, record


def checkpoint_path(model_id, revision, entry, fetched):
    """Snapshot at the frozen revision, with every weight file matching the plan's digest.

    Missing files are downloaded. Weight files this run downloads are listed in
    `fetched` so the caller can remove them again; cached ones are never touched.
    """
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    names = [f["file"] for f in entry["files"] if f["file"].endswith((".safetensors", ".json"))]
    absent = [n for n in names if n.endswith(".safetensors")
              and not isinstance(try_to_load_from_cache(model_id, n, revision=revision), str)]
    path = Path(snapshot_download(model_id, revision=revision, allow_patterns=names))
    fetched.extend(path / n for n in absent)
    for f in entry["files"]:
        if f["file"].endswith(".safetensors"):
            h = hashlib.sha256()
            with open(path / f["file"], "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                raise ValueError(f"{model_id}: {f['file']} differs from the frozen digest")
    return path


def discard(fetched):
    """Delete the weight files (cache link and blob) that this run downloaded."""
    while fetched:
        link = fetched.pop()
        link.resolve().unlink(missing_ok=True)
        link.unlink(missing_ok=True)


def encode(tok, probes, chat, max_length):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in probes] if chat else probes
    ids = [tok(t, add_special_tokens=not chat)["input_ids"][:max_length] for t in texts]
    truncated = sum(len(tok(t, add_special_tokens=not chat)["input_ids"]) > max_length for t in texts)
    return ids, truncated


def forward(model, ids, batch_size, pad_id, device):
    """Yield (per-layer states pooled without position 0, log-probabilities, mask) per batch."""
    import torch
    from lasa.pipeline.similarity import pool_hidden_states
    for i in range(0, len(ids), batch_size):
        rows = ids[i:i + batch_size]
        width = max(len(r) for r in rows)
        x = torch.tensor([r + [pad_id] * (width - len(r)) for r in rows], device=device)
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=mask, output_hidden_states=True, use_cache=False)
        pooled = pool_hidden_states(out.hidden_states, mask, skip_first=True)
        yield pooled, torch.log_softmax(out.logits.float(), -1).cpu(), mask.cpu().bool()


def stack(chunks):
    return [np.concatenate(layer) for layer in zip(*chunks)]


def uplifts(report_dir, label_file, fill=0):
    """Per-target, per-behavior held-out uplift of the matched suffix over each control."""
    gen = json.loads((report_dir / "generations.json").read_text())["items"]
    labels = {r["id"]: r["label"] for r in json.loads((report_dir / label_file).read_text())["items"]}
    cells = {}
    for r in gen:
        if r["split"] != "heldout":
            continue
        y = labels[r["id"]]
        cells.setdefault((r["model_id"], r["condition"], r["behavior_id"]), []).append(fill if y is None else y)
    behaviors = sorted({k[2] for k in cells})
    models = sorted({k[0] for k in cells})
    return {m: {c: np.array([np.mean(cells[m, "gcg", b]) - np.mean(cells[m, c, b]) for b in behaviors])
                for c in CONTRASTS} for m in models}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lasa.study_analysis import holm

    p = argparse.ArgumentParser()
    p.add_argument("plan")
    args = p.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text())
    cfg = yaml.safe_load(Path(plan["study"]).read_text())
    report = Path(plan["report_dir"])
    frozen_plan = json.loads((report / "plan.json").read_text())
    summary = json.loads((report / "summary.json").read_text())
    frozen = summary["similarity_test"]
    source_id = cfg["base"]["model_id"]
    targets = [m for m in frozen["similarity"] if m not in plan["excluded"]]
    source_digests = sorted(f["sha256"] for f in frozen_plan["models"][source_id]["files"]
                            if f["file"].endswith(".safetensors"))
    copies = [m for m in targets if sorted(f["sha256"] for f in frozen_plan["models"][m]["files"]
                                           if f["file"].endswith(".safetensors")) == source_digests]
    sets, probe_record = load_probe_sets(plan)
    revisions = {v["model_id"]: v["revision"] for v in [cfg["base"], *cfg["variants"]]}
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    fetched = []

    def load(model_id):
        # An exact copy carries the source's weights under another id.
        real = source_id if model_id in copies else model_id
        new = []
        path = checkpoint_path(real, revisions[real], frozen_plan["models"][real], new)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, local_files_only=True,
                                                     use_safetensors=True).to(device).eval()
        if real != source_id:
            discard(new)  # loaded into memory; the downloaded copy is no longer needed
        return model

    source_path = checkpoint_path(source_id, revisions[source_id], frozen_plan["models"][source_id], fetched)
    tok = AutoTokenizer.from_pretrained(source_path, local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inputs, truncated = {}, {}
    for name, probes in sets.items():
        for form, chat in (("raw", False), ("chat", True)):
            inputs[name, form], truncated[f"{name}|{form}"] = encode(tok, probes, chat, plan["max_length"])

    model = load(source_id)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    reference = {}
    for key, ids in inputs.items():
        states, logps = [], []
        for pooled, logp, mask in forward(model, ids, plan["batch_size"], pad, device):
            states.append(pooled)
            if key[1] == "raw":
                logps.append(logp[mask])
            else:
                logps.append(logp[torch.arange(len(mask)), mask.sum(1) - 1])
        reference[key] = (stack(states), logps)
    del model

    values, frozen_states = {}, {}
    for model_id in targets:
        model = load(model_id)
        state = model.state_dict()
        if set(state) != set(weights):
            raise ValueError(f"{model_id}: parameter names differ from the source")
        diff = sum(float((state[k].cpu().double() - w.double()).pow(2).sum()) for k, w in weights.items())
        norm = sum(float(w.double().pow(2).sum()) for w in weights.values())
        row = {"weight_l2": -float(np.sqrt(diff / norm))}
        for (name, form), ids in inputs.items():
            states, divergences = [], []
            for i, (pooled, logp, mask) in enumerate(forward(model, ids, plan["batch_size"], pad, device)):
                states.append(pooled)
                here = logp[mask] if form == "raw" else logp[torch.arange(len(mask)), mask.sum(1) - 1]
                divergences.append(js_divergence(reference[name, form][1][i], here).numpy())
            states = stack(states)
            row[f"{name}|cka_{form}"] = middle_mean_cka(reference[name, form][0], states)
            row[f"{name}|{'js_all' if form == 'raw' else 'js_next'}"] = -float(np.concatenate(divergences).mean())
            if (name, form) == ("frozen", "raw"):
                frozen_states[model_id] = states
        values[model_id] = row
        del model
        if device == "mps":
            torch.mps.empty_cache()
        print(model_id, {k: round(v, 5) for k, v in row.items() if k.startswith(("frozen", "weight"))}, flush=True)

    n_perm, seed = plan["statistics"]["n_permutations"], plan["seed"]
    judged = {"frozen_judge": uplifts(report, "judgments.json")}
    for fill in (0, 1):
        judged[f"harmbench_13b_invalid_{fill}"] = uplifts(
            report, f"diagnostic_judgments_harmbench_llama2_13b_fill{fill}.json")
    mean_uplift = {j: {m: {c: float(v.mean()) for c, v in u[m].items()} for m in targets} for j, u in judged.items()}
    for m in targets:
        for c in CONTRASTS:
            if not np.isclose(mean_uplift["frozen_judge"][m][c], frozen["tests"][c]["uplift"][m]):
                raise ValueError(f"recomputed uplift differs from the frozen summary: {m} {c}")

    measures = list(values[targets[0]])
    target_sets = {"all": targets, "without_copy": [m for m in targets if m not in copies]}
    tests = {}
    for judge, u in mean_uplift.items():
        for set_name, ids in target_sets.items():
            for measure in ["frozen_values"] + measures:
                x = [frozen["similarity"][m] if measure == "frozen_values" else values[m][measure] for m in ids]
                for c in CONTRASTS:
                    y = [u[m][c] for m in ids]
                    tests.setdefault(judge, {}).setdefault(set_name, {}).setdefault(measure, {})[c] = {
                        "rho": rank_rho(x, y), "p": permutation_p(x, y, n_perm, seed), "n": len(ids)}
    for by_set in tests.values():
        for by_measure in by_set.values():
            adjusted = holm({f"{m}|{c}": r["p"] for m in measures for c, r in by_measure[m].items()})
            for key, value in adjusted.items():
                m, c = key.rsplit("|", 1)
                by_measure[m][c]["holm_p_all_measures"] = value

    # Uncertainty from behaviors (shared by all targets) and from the frozen probes.
    rng = np.random.default_rng(seed)
    base_states = reference["frozen", "raw"][0]
    n_probes, n_behaviors = len(sets["frozen"]), len(judged["frozen_judge"][targets[0]]["bare"])
    matrix = {c: np.stack([judged["frozen_judge"][m][c] for m in targets]) for c in CONTRASTS}
    fixed_x = np.array([frozen["similarity"][m] for m in targets])
    full_y = {c: matrix[c].mean(1) for c in CONTRASTS}

    def interval(draws):
        draws = np.asarray(draws)
        draws = draws[np.isfinite(draws)]
        return {"interval95": [float(q) for q in np.quantile(draws, [0.025, 0.975])],
                "median": float(np.median(draws)), "share_at_or_below_zero": float(np.mean(draws <= 0)),
                "n": int(len(draws))}

    boot = {"behaviors": {}, "probes": {}, "joint": {}}
    draws = [rng.integers(n_behaviors, size=n_behaviors) for _ in range(10000)]
    for c in CONTRASTS:
        boot["behaviors"][c] = interval([rank_rho(fixed_x, matrix[c][:, d].mean(1)) for d in draws])
    probe_x = []
    for _ in range(2000):
        d = rng.integers(n_probes, size=n_probes)
        base = [layer[d] for layer in base_states]
        probe_x.append([middle_mean_cka(base, [layer[d] for layer in frozen_states[m]]) for m in targets])
    for c in CONTRASTS:
        boot["probes"][c] = interval([rank_rho(x, full_y[c]) for x in probe_x])
        boot["joint"][c] = interval([rank_rho(x, matrix[c][:, draws[k]].mean(1)) for k, x in enumerate(probe_x)])

    headline = [tests["frozen_judge"]["all"][m]["mismatched_gcg"]["rho"] for m in measures]
    reading = {"n_measures": len(measures), "n_positive": int(sum(r > 0 for r in headline)),
               "median_rho": float(np.median(headline)), "min_rho": float(min(headline)),
               "max_rho": float(max(headline))}
    reading["stable_across_measures"] = bool(reading["n_positive"] == len(measures) and reading["median_rho"] >= 0.5)

    environment = {"device": device, "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__}
    payload = {"status": "exploratory; plan committed before computing; no frozen decision changes",
               "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
               "environment": environment, "targets": targets, "excluded": plan["excluded"],
               "exact_copies_of_source": copies, "probe_sets": probe_record, "truncated_probes": truncated,
               "values": values, "uplift": mean_uplift, "tests": tests, "bootstrap": boot, "reading": reading,
               "agreement_with_frozen_values": {m: rank_rho(fixed_x, [values[t][m] for t in targets])
                                                for m in measures},
               "max_reproduction_gap_frozen_measure": float(max(
                   abs(values[m]["frozen|cka_raw"] - frozen["similarity"][m]) for m in targets))}
    Path(plan["out"]).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(reading, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
