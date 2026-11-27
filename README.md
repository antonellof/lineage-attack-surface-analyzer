# LASA: Lineage Attack Surface Analyzer

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/PAPER.pdf)
[![CI](https://github.com/antonellof/lineage-attack-surface-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/antonellof/lineage-attack-surface-analyzer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Can evidence from an original LLM predict which jailbreaks transfer to its derivatives?**

LASA is a defensive research harness for testing that question on authorized open-weight models. It searches only on the source model, freezes the resulting attacks, and measures their effect on related checkpoints against attack-free, random-suffix, and mismatched-suffix controls.

[Read the paper](paper/PAPER.pdf) · [Manuscript](paper/PAPER.md) · [Protocol](docs/STUDY.md) · [Measurements](experiments/results/study-qwen15-pilot/) · [Bibliography](paper/references.bib)

## Research

**Do Jailbreaks Follow Lineage? A Pre-Specified Protocol and Four Classifier-Labeled Studies of GCG Transfer to Open-Weight Derivatives**

Antonello Fratepietro

The implementation combines published nanoGCG search, three controls (bare request, random suffix, a suffix optimized for another behavior), held-out behaviors, an independent HarmBench classifier, and model-labeled audits of exact generations. Development behaviors fix the decision rules before held-out outcomes, and a public plan commitment must be committed and pushed before search. Four studies cover three sources (Qwen2.5 1.5B, Granite 4.1 3B, Gemma 3 1B), 27 derivatives and 40,000 generations. Transfer that beats all three controls appeared on 9 of 24 eligible targets, all fine-tunes of the Qwen source or a copy of it, and a frozen test found that benign-probe similarity ranked transfer across 14 Qwen fine-tunes. All outcomes are classifier labels; a model audit does not support the classifier's precision. See [paper/PAPER.md](paper/PAPER.md) and the [study protocol](docs/STUDY.md).

Historical artifacts cover **four lineages in five source groups** and 19 distinct checkpoint IDs. Their attack outcomes are superseded and are not evidence of validated transfer. Their benign CKA profiles are dominated by the first token position ([diagnostic](experiments/results/cka_first_position_diagnostic.json)).

![Source-only search and controlled transfer evaluation](paper/figures/study_design.png)

## Quick start

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,models]"
pytest
lasa study --help
```

Run the engineering pilot:

```sh
lasa fetch --config experiments/study-qwen15-pilot.yaml
lasa study initialize --config experiments/study-qwen15-pilot.yaml
lasa study search --config experiments/study-qwen15-pilot.yaml
lasa study generate --config experiments/study-qwen15-pilot.yaml
lasa study judge --config experiments/study-qwen15-pilot.yaml
lasa study audit --config experiments/study-qwen15-pilot.yaml
lasa study analyze --config experiments/study-qwen15-pilot.yaml
```

The plan is immutable. Change the study ID and output directory for a new experiment. Model downloads require about 20 GB for this pilot; raw generations and suffixes remain in protected, gitignored `reports/`. Human audit labels must be supplied separately. Full-study settings and controls are documented in [the protocol](docs/STUDY.md).

## Reproducibility

- Frozen code/config digests, model hashes, seeds, and benchmark splits.
- Resumable search, generation, and independent judging.
- Exact token-level evidence; no regenerated substitutes for audit samples.
- Paired effect estimates with shared-control resampling.
- [Pilot summary](experiments/results/study-qwen15-pilot/summary.json), [measurements](experiments/results/study-qwen15-pilot/measurements.json), [power analysis](experiments/results/power_analysis.json), and [evidence inventory](experiments/evidence_inventory.json).

Build figures, references, and the PDF with `make -C paper pdf`. The bibliography is verified against arXiv and PMLR records. Historical experiments are retained separately and excluded from the current paper's evidence.

## Cite

GitHub's **Cite this repository** uses [CITATION.cff](CITATION.cff). BibTeX:

```bibtex
@software{fratepietro2026lasa,
  author = {Fratepietro, Antonello},
  title = {LASA: Lineage Attack Surface Analyzer},
  year = {2026},
  version = {0.0.1},
  url = {https://github.com/antonellof/lineage-attack-surface-analyzer}
}
```

## Responsible use

Local, authorized checkpoints only. Published attacks and established benchmarks; no new harmful corpora or closed-API targets. Results are published in aggregate only, without suffixes or completions. See [project rules](CLAUDE.md) and [architectural decisions](docs/ADR.md).
