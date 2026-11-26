# Response to the peer review

Date: 2026-11-26. Branch: `fix/peer-review-concerns`. The review was read as an
external assessment of `paper/PAPER.md`; no claim in it was accepted without checking
it against the repository.

## Disposition of the review's points

| Review point | Status | Where it is handled |
| --- | --- | --- |
| 3.1 classifier not validated; human audit abandoned after outcomes | Already conceded by the manuscript; now quantified | Abstract, Section 6.8, Limitations "Measurement validity", ADR-0025 |
| 3.2 positive result is not demonstrably behavior-specific | Confirmed; made a measured quantity | Sections 6.3 and 6.4, `behavior_specific_transfer` (ADR-0027), tests |
| 3.3 prioritization estimand can reward generic susceptibility | Computed, not just discussed | Section 4.7 (worked example), Section 6.5 table and prose, ADR-0027 |
| 3.4 limited scope and power | Already conceded; wording tightened | Limitations "Scope" and "Power", Section 4.8 |
| 3.5 Arch-Router exclusion needs justification and sensitivity analysis | Added | Section 6.3 sensitivity paragraph, Limitations "Roster rules" |
| 3.6 text-level versus token-level transfer | Split explicitly | Renamed Section 6.6, Limitations "Tokenizer differences" |
| 4.5 combined implications not made explicit | Added | Conclusion, Section 6 preamble |
| 5.1 worked example for the top-5 utility | Added | Section 4.7 |
| 5.2 terminology consistency | Fixed | Section 1 term list (classifier-positive, generic suffix effect, behavior-specific transfer) |
| 5.3 emphasize matched versus mismatched | Done in Section 6.3/6.4 prose; figure unchanged |  |
| Reproducibility: private evidence, squashed history | Stated plainly | Section 9 "What a reader can rebuild, and what a reader cannot" |
| Missing from the review: the per-condition audit argument | Handled and bounded | Section 6.8 intervals, ADR-0028 |
| Missing from the review: the unexplained loss gap, truncation interplay, disclosure gate | Added | Sections 6.1, 6.8, Section 9, Limitations "Release readiness", `validate_submission.py` |

## Where the review is wrong

1. Ligature artifacts. `paper/PAPER.md` has no ligature codepoints; the source words
   are spelled normally. The artifact is in the PDF text layer: Latin Modern emits one
   ligature glyph, U+FB00 for `ff`, so `pdftotext` and screen readers see a codepoint
   instead of two letters. Verified: `paper/build/arxiv/paper.txt` expands 110 of them.
   Rendering was never wrong, so the fix is a searchable text companion plus a build check
   (`paper/build_arxiv.py`, `paper/validate_submission.py`), not prose normalization.

2. "The classifier overcounts, therefore the contrasts are unreliable." The review
   treats the overcount as uniform. It is not: in the Qwen study the labelers confirmed
   19 and 21 of 33 sampled positives under the matched suffix, 5 and 8 of 17 under the
   mismatched suffix, and 2 of 7 under the bare request. Since every decision is a contrast,
   an overcount concentrated in the control conditions would make the contrasts conservative.
   The new per-condition intervals (ADR-0028) show that this cannot be settled from a
   120-output sample, which is why the manuscript now states the question as open rather
   than answered in either direction.

3. Granite null results and tokenization. The review generalizes the tokenization confound
   to the Granite study. The Granite coder reproduced the source token IDs for every behavior,
   so its null is a control-condition result, not a tokenizer artifact. Section 6.6 separates
   the two experiments explicitly now.

4. Priority ordering of the requested revisions. The estimand revision (its Priority 1,
   item 3) needed no new generations: every condition was already in the frozen evidence, so
   it was computed by re-running the frozen analysis with `--allow-analysis-revision`
   (ADR-0027). The human audit is the expensive item, and a design-weighted 120-output
   sample cannot correct per-cell rates, so the review's first recommendation is not
   interchangeable with its third.

## What the recomputation changed in the claims

- No frozen status changed. Every primary status reproduces byte-for-byte; only new
  fields appear in `experiments/results/study-*/aggregate_results.json`.
- `behavior_specific_transfer` is `not_supported` for all six family targets. Qwen2.5-Coder
  fails it with a Holm-adjusted p of 1.0 on the mismatched contrast.
- Granite coder's prioritization utility is 0.45 against the bare request, 0.20 against
  a random suffix, and -0.02 against the mismatched suffix. The score tracked request
  susceptibility, not transfer. Arch-Router behaves the same way (-0.02).
- Per-condition labeler precision in the Qwen study: 0.58 (0.41 to 0.74) and 0.64 (0.47
  to 0.80) under the matched suffix, 0.29 (0.08 to 0.51) and 0.47 (0.23 to 0.71) under
  the mismatched suffix, 0.29 (0.00 to 0.62) under the bare request. Only the matched-suffix
  cell reaches the 30 sampled positives the code asks for; no Granite condition does.

## What remains open

The human audit. The frozen criterion needs two human annotators on the untouched sheets
`human_labels.csv` and `human_labels_2.csv`, and nothing in this branch substitutes for
that: `classifier_validation` stays `pending`, and the release gate in
`paper/validate_submission.py` stays unmet until owner notices are recorded as sent.
