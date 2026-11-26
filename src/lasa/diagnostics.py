"""Exploratory diagnostics on finished study evidence (ADR-0031).

Every stage reads a frozen plan and its evidence, writes a new `diagnostic_*.json`
artifact beside them, and never rewrites `generations.json`, `judgments.json` or any
frozen decision. Code may be newer than the plan; each artifact records both digests.

Stages:
  judge      label every generation with a second classifier and template
  tokens     re-run held-out GCG cells with the optimized source token IDs inserted
             directly, for targets whose tokenizer re-encoded the suffix differently
  loss       recompute source target losses in bfloat16 and float32 on the exact IDs
  summarize  contrasts and agreement from a diagnostic label set; rates only, no text
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lasa.study import (code_digest, decoding_config, digest, exact_target_loss, frozen, load_model,
                        locked_stage, parse_judge, private_json, prompt_ids, release, trim_after_eos)
from lasa.study_analysis import holm, paired_effect_ci, sign_flip_p

def _load_classifier(model_id, revision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _classify(model, tok, prompt):
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[-1] + 4 > model.config.max_position_embeddings:
        return "", None, "context_overflow"
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=4, pad_token_id=tok.pad_token_id)
    answer = tok.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    label = parse_judge(answer)
    return answer, label, "scored" if label is not None else "invalid_label"


def _judge_items(dest, items, behaviors, *, model_id, revision, rubric, header):
    """Resumable labeling of `items` into `dest`. Header fields identify the label set."""
    state = json.loads(dest.read_text()) if dest.exists() else {**header, "items": []}
    for key, value in header.items():
        if state.get(key) != value:
            raise ValueError(f"{dest.name} was made with a different {key}")
    done = {r["id"] for r in state["items"]}
    model, tok = _load_classifier(model_id, revision)
    try:
        for n, row in enumerate(items):
            if row["id"] in done:
                continue
            prompt = rubric["prompt"].format(behavior=behaviors[row["behavior_id"]], generation=row["text"])
            answer, label, status = _classify(model, tok, prompt)
            state["items"].append({"id": row["id"], "label": label, "raw_label": answer, "status": status})
            if n % 200 == 0:
                private_json(dest, state)
        private_json(dest, state)
    finally:
        del model
        release()
    return dest


@locked_stage
def judge(config_path, *, model_id, revision, rubric_path, tag, sample=None):
    """Second label set for every generation, or a seeded stratified sample of `sample` items."""
    cfg, plan, ids, out = frozen(config_path, allow_analysis_revision=True)
    evidence = json.loads((out / "generations.json").read_text())
    rubric = json.loads(Path(rubric_path).read_text())
    items = evidence["items"]
    if sample:
        primary = {r["id"]: r["label"] for r in json.loads((out / "judgments.json").read_text())["items"]}
        rng = np.random.default_rng([cfg["seed"], 31])
        by_label = defaultdict(list)
        for r in items:
            by_label[primary.get(r["id"])].append(r)
        chosen = []
        for label in sorted(by_label, key=str):
            group = by_label[label]
            take = min(len(group), sample // 2)
            chosen += [group[i] for i in rng.choice(len(group), size=take, replace=False)]
        items = chosen
    behaviors = {b["behavior_id"]: b["instruction"] for b in plan["behaviors"]}
    header = {"evidence_digest": digest(evidence), "judge_model": model_id, "judge_revision": revision,
              "rubric_digest": digest(rubric), "rubric_source": rubric.get("template_name", rubric.get("source_file")),
              "sampled": bool(sample), "n_planned": len(items), "plan_code_digest": plan["code_digest"],
              "diagnostic_code_digest": code_digest(), "status": "exploratory diagnostic"}
    return _judge_items(out / f"diagnostic_judgments_{tag}.json", items, behaviors,
                        model_id=model_id, revision=revision, rubric=rubric, header=header)


@locked_stage
def tokens(config_path, *, targets):
    """Token-ID-level transfer: the target receives the exact optimized source suffix IDs.

    Text-level transfer re-encodes the suffix string with the target tokenizer. When
    that changes the IDs, a null result cannot separate a robust target from an altered
    suffix. This stage inserts `suffix_source_ids` between the target's own template
    pieces, with the seed the frozen generate stage used for that behavior, and stores
    the outputs in `diagnostic_token_transfer.json`. Vocabulary identity is checked
    first; a target whose vocabulary differs from the source's is refused.
    """
    import torch
    from transformers import AutoTokenizer
    cfg, plan, ids, out = frozen(config_path, allow_analysis_revision=True)
    attacks = {a["behavior_id"]: a for a in json.loads((out / "source_attacks.json").read_text())["attacks"]}
    source_tok = AutoTokenizer.from_pretrained(plan["models"][ids[0]]["path"])
    dest = out / "diagnostic_token_transfer.json"
    state = json.loads(dest.read_text()) if dest.exists() else {
        "plan_digest": digest(plan), "condition": "gcg_token_ids", "items": [], "vocabulary": {},
        "plan_code_digest": plan["code_digest"], "diagnostic_code_digest": code_digest(),
        "status": "exploratory diagnostic"}
    done = {(r["model_id"], r["behavior_id"], r["replicate"]) for r in state["items"]}
    budget = int(cfg["max_new_tokens"])
    for model_id in targets:
        if model_id not in ids[1:]:
            raise ValueError(f"{model_id} is not a target of this study")
        model, tok = load_model(plan, model_id)
        same_vocab = tok.get_vocab() == source_tok.get_vocab()
        state["vocabulary"][model_id] = {"identical_to_source": same_vocab, "size": len(tok.get_vocab())}
        if not same_vocab:
            del model
            release()
            raise ValueError(f"{model_id} has a different vocabulary; token-ID transfer is undefined")
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        try:
            for i, b in enumerate(plan["behaviors"]):
                if b["split"] != "heldout":
                    continue
                if all((model_id, b["behavior_id"], r) in done for r in range(cfg["generations"])):
                    continue
                a = attacks[b["behavior_id"]]
                text_built = prompt_ids(tok, b["instruction"], a["suffix"])
                empty = prompt_ids(tok, b["instruction"], "")
                before = empty["ids"][:empty["n_before"]]
                after = empty["ids"][empty["n_before"]:]
                ids_built = before + list(a["suffix_source_ids"]) + after
                input_ids = torch.tensor([ids_built], device=model.device)
                seed = cfg["seed"] + i * 10000
                torch.manual_seed(seed)
                with torch.no_grad():
                    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                            generation_config=decoding_config(model, tok, budget),
                                            num_return_sequences=cfg["generations"])
                for r in range(cfg["generations"]):
                    t = trim_after_eos(output[r][input_ids.shape[-1]:], eos_ids)
                    key = [model_id, b["behavior_id"], "gcg_token_ids", r]
                    state["items"].append({
                        "id": digest(key)[:24], "model_id": model_id, "behavior_id": b["behavior_id"],
                        "split": "heldout", "condition": "gcg_token_ids", "replicate": r, "seed": seed,
                        "text": tok.decode(t, skip_special_tokens=True), "generation_ids": t.tolist(),
                        "budget_exhausted": len(t) >= budget,
                        "text_level_ids_matched_source": text_built["suffix_ids"] == list(a["suffix_source_ids"])})
                private_json(dest, state)
        finally:
            del model
            release()
    behaviors = {b["behavior_id"]: b["instruction"] for b in plan["behaviors"]}
    judge_cfg = cfg["independent_judge"]
    rubric = json.loads(Path(judge_cfg["rubric"]).read_text())
    header = {"evidence_digest": digest(state), "judge_model": judge_cfg["model_id"],
              "judge_revision": judge_cfg["revision"], "rubric_digest": digest(rubric),
              "status": "exploratory diagnostic"}
    _judge_items(out / "diagnostic_token_transfer_judgments.json", state["items"], behaviors,
                 model_id=judge_cfg["model_id"], revision=judge_cfg["revision"], rubric=rubric, header=header)
    return dest


@locked_stage
def loss(config_path):
    """Source target loss on the exact evaluated IDs, in the frozen dtype and in float32."""
    import torch
    from lasa.pipeline.verify import _load_variant_model
    cfg, plan, ids, out = frozen(config_path, allow_analysis_revision=True)
    attacks = json.loads((out / "source_attacks.json").read_text())["attacks"]
    targets = {b["behavior_id"]: b["target"] for b in plan["behaviors"]}
    instructions = {b["behavior_id"]: b["instruction"] for b in plan["behaviors"]}
    rows = {a["behavior_id"]: {"behavior_id": a["behavior_id"], "nanogcg_reported": a["source_loss"],
                               "frozen_recomputed": a["evaluated_source_loss"]} for a in attacks}
    for dtype in ("bfloat16", "float32"):
        model, tok = _load_variant_model(plan["models"][ids[0]]["path"], dtype)
        try:
            for a in attacks:
                built = prompt_ids(tok, instructions[a["behavior_id"]], a["suffix"])
                rows[a["behavior_id"]][dtype] = exact_target_loss(model, tok, built["ids"], targets[a["behavior_id"]])
        finally:
            del model
            release()
    values = list(rows.values())

    def gap(key):
        d = np.array([r[key] - r["nanogcg_reported"] for r in values])
        return {"median_abs": float(np.median(np.abs(d))), "max_abs": float(np.abs(d).max()),
                "n_recomputed_higher": int((d > 0).sum()), "n_over_0.001": int((np.abs(d) > 0.001).sum())}

    payload = {"plan_code_digest": plan["code_digest"], "diagnostic_code_digest": code_digest(),
               "status": "exploratory diagnostic", "n_behaviors": len(values),
               "gap_vs_nanogcg": {"bfloat16": gap("bfloat16"), "float32": gap("float32")},
               "items": values}
    dest = out / "diagnostic_loss_precision.json"
    private_json(dest, payload)
    return dest


def summarize(config_path, *, tag):
    """Rates, contrasts and agreement under a diagnostic label set. Reads only; no text leaves."""
    cfg, plan, ids, out = frozen(config_path, allow_analysis_revision=True, analysis_only=True)
    evidence = json.loads((out / "generations.json").read_text())
    alt = json.loads((out / f"diagnostic_judgments_{tag}.json").read_text())
    if alt["evidence_digest"] != digest(evidence):
        raise ValueError("diagnostic labels belong to other evidence")
    primary = {r["id"]: r["label"] for r in json.loads((out / "judgments.json").read_text())["items"]}
    labels = {r["id"]: r["label"] for r in alt["items"]}
    rows = [r for r in evidence["items"] if r["id"] in labels]
    pairs = [(primary[r["id"]], labels[r["id"]]) for r in rows if labels[r["id"]] in (0, 1)]
    agreement = {
        "n": len(pairs), "invalid": sum(labels[r["id"]] not in (0, 1) for r in rows),
        "raw_agreement": float(np.mean([a == b for a, b in pairs])) if pairs else None,
        "primary_positive_rate": float(np.mean([a for a, _ in pairs])) if pairs else None,
        "diagnostic_positive_rate": float(np.mean([b for _, b in pairs])) if pairs else None,
        "primary_positive_confirmed": (float(np.mean([b for a, b in pairs if a == 1]))
                                       if any(a == 1 for a, _ in pairs) else None),
        "primary_negative_flagged": (float(np.mean([b for a, b in pairs if a == 0]))
                                     if any(a == 0 for a, _ in pairs) else None),
    }
    payload = {"tag": tag, "judge_model": alt["judge_model"], "judge_revision": alt["judge_revision"],
               "rubric_source": alt.get("rubric_source"), "sampled": alt["sampled"],
               "status": "exploratory diagnostic; frozen decisions are unchanged",
               "agreement_with_primary_judge": agreement}
    if alt["sampled"]:
        return payload
    heldout = [b["behavior_id"] for b in plan["behaviors"] if b["split"] == "heldout"]
    cell = {(r["model_id"], r["behavior_id"], r["replicate"], r["condition"]): labels[r["id"]] for r in rows}
    roles = {v["model_id"]: v.get("evaluation_role", "primary") for v in cfg["variants"]}
    conditions = tuple(plan["conditions"])
    controls = [c for c in conditions if c != "gcg"]
    results, pvalues = {}, {}
    for mid in ids:
        x = np.array([[[cell[(mid, b, g, c)] for c in conditions] for g in range(cfg["generations"])]
                      for b in heldout], dtype=float)
        if not np.isin(x, [0, 1]).all():
            results[mid] = {"status": "unscored labels present"}
            continue
        effects = paired_effect_ci(x, conditions, seed=cfg["seed"])
        for c in controls:
            d = (x[:, :, conditions.index("gcg")] - x[:, :, conditions.index(c)]).mean(axis=1)
            effects[f"gcg_minus_{c}"]["p"] = sign_flip_p(d)["p"]
            if mid != ids[0] and roles.get(mid) == "primary":
                pvalues[f"{mid}|{c}"] = effects[f"gcg_minus_{c}"]["p"]
        results[mid] = {"role": "source" if mid == ids[0] else roles.get(mid),
                        "rate_by_condition": {c: float(x[:, :, i].mean()) for i, c in enumerate(conditions)},
                        "effects": effects}
    adjusted = holm(pvalues) if pvalues else {}
    for key, p in adjusted.items():
        mid, c = key.split("|")
        results[mid]["effects"][f"gcg_minus_{c}"]["holm_p_all_three_controls"] = p
    payload["heldout"] = results
    payload["holm_family"] = "primary-family targets times all three controls; exploratory"
    return payload
