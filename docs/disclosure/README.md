# Responsible disclosure (historical)

**Superseded by ADR-0030 (2026-11-26).** Owner notification is no longer a precondition for publication, and `validate_submission.py` no longer reports a release gate. This folder is kept as a record. The text below describes the earlier rule.

Project boundary 4 (CLAUDE.md, ADR-0004) requires that any finding about a
publicly released model reach that model's owner before publication. This folder
tracks that obligation. The author sends the notices; this repository cannot send
mail.

## Status of historical drafts

**Do not send the existing notices as findings.** They cite superseded activation-direction measurements and non-refusal labels, not validated harmful-compliance outcomes. No notice in this directory is recorded as sent. The CSV preserves the historical draft list, not confirmed vulnerability findings.

Before publication, review fresh independently judged evidence and the completed human audit. Prepare owner-specific notices only for substantiated findings, and accurately state uncertainty. No harmful completion text needs to be shared.

## How to use this folder

1. Reassess whether each owner has a substantiated finding after the corrected study.
2. Replace the superseded drafts with notices based on validated evidence.
3. Record the date sent and any reply in the tracking file. Do not mark notices sent without an actual transmission.

Disclosure is a precondition for publication, not a step after it.

## Version 3 studies (Qwen v3, Granite v1)

On 2026-11-13 the operator stated that disclosure for the v3 findings had been sent, and on that basis the results were pushed to the public repository. This note records the operator's statement. The recipients, the dates sent, and any replies are not recorded in this folder yet. The operator adds them to `tracking.csv`. The rows currently in that file are the historical draft list and do not describe the v3 notices.

`python paper/validate_submission.py` reports this as a release gate beside its technical checks: it lists the checkpoints of the two published studies, counts those with a recorded date sent, and sets `releasable_now` to false while any record is missing. A passing technical preflight is not a licence to release. To close the gate, add one row per checkpoint with `status`, `reason`, `date_sent`, and any `reply` filled in from an actual transmission.
