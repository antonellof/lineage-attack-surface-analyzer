# Building and submitting the manuscript

## Reading copy

```sh
make -C paper pdf
```

This rebuilds figures from committed numerical pilot metadata, resolves citations from `references.bib` with Pandoc citeproc and `ieee.csl`, and writes `PAPER.pdf`. Both the reading copy and LaTeX bundle use the same title, abstract, bibliography, and typography: 11-point article text, one-inch margins, and linked references. The CSL style is distributed under the license recorded in its header.

`PAPER.md` is the manuscript source. BibTeX contains verified author lists and arXiv or PMLR links; software and model checkpoints have separate entries. No DOI, affiliation, publication venue, or result is supplied without a source.

## LaTeX source bundle

```sh
make -C paper arxiv
```

`build/arxiv/` contains `paper.tex`, referenced figures, and bibliography/style sources. Citeproc renders the bibliography into the LaTeX, so no BibTeX pass is required. The source uses Unicode fonts and compiles with XeLaTeX through Tectonic; a pdflatex build is not validated. Check arXiv's current processor rules and the venue's template before uploading. The generic article layout is not an official ICLR or ACM template.

`make -C paper validate` independently compiles the generated source and checks visible author metadata, embedded fonts, missing references and bundled figures. It writes `paper/build/validation.json` and a minimal `paper/build/submission.zip`. This local preflight is not a certificate of arXiv acceptance or research quality.

## Before submission

- [x] **Audit:** no human audit was run. Two Claude models labeled each 120-output sample after outcomes were known (ADR-0025). ADR-0029 applies the frozen thresholds to them unchanged: `classifier_not_supported` in both studies. The paper reports this as a post hoc test with labelers the plan did not name (Sections 4.5 and 6.8). The sheets `human_labels.csv` and `human_labels_2.csv` are untouched if a human audit is wanted later.
- [x] **Roster:** kept as frozen. Arch-Router is a task control outside the primary family; the paper reports its result as descriptive.
- [x] **Final results and figures:** Section 6 cites `experiments/results/study-*/aggregate_results.json`, rebuilt after ADR-0029. `figures/study_results.png` and `experiments/evidence_inventory.json` are built from the same files.
- [x] **Claim verification:** Pilot counts, power numbers, CKA ranges, the Section 6 rate, contrast, and audit tables, and the inventory have automated checks. The prioritization table and the prose numbers are checked by hand only.
- [x] **Provenance:** the pre-squash revisions cited by committed evidence stay unpublished (ADR-0019); the frozen code digest identifies the pilot code.
- [ ] **Venue:** None selected. Apply the venue's template and confirm page limits and artifact rules.

The current manuscript reports two confirmatory studies with classifier-based outcomes, a post hoc model audit under which the classifier fails the frozen precision threshold, the pilot, and three measurement pitfalls. It makes no claim about harmful-compliance rates.

- [ ] **arXiv metadata:** paste the abstract from `PAPER.md` (1,734 characters, under the 1,920 limit). Suggested primary category cs.CR, cross-list cs.LG and cs.CL.
