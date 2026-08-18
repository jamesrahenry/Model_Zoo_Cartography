# Archives — external review passes on the paper draft

*Written: 2026-08-18 19:05 UTC.*

Superseded artifacts from external model reviews of `paper/DRAFT.md`, kept
for the audit trail (this repo's convention: corrections are recorded, not
overwritten). Neither directory is current — the live draft absorbed what
survived verification and declined the rest.

## 2026-08-18_gpt-full-rewrite/

GPT's review, delivered as a full rewrite of the draft (then at v0.9) and
`make_figures.py`. Not committed as-is. Its four checkable technical claims
were verified against the code and merged in our own wording as **draft
v0.10** (commit `42bac3d`): the mean/covariance-only mixture premise, the
top-20 anchored-count censoring, the cross-task-under-noise-input mismatch,
and validation-not-held-out for the refit numbers. The rest — title
flattening, deletion of the falsification arc and version history, blanket
hedging of verified claims — was reviewed and declined; see the v0.10 header
note in `paper/DRAFT.md` for the accept/decline record. Originally preserved
in a git stash; moved here 2026-08-18 and the stash dropped.

## 2026-08-18_gemini-pass/

Gemini's review, delivered as edit scripts (`edit_draft*.py`, `polish.py`)
plus two candidate drafts (`DRAFT_edited*.md`). Unlike GPT's, part of this
pass was committed directly (`36facf1`: meaning-lead removal, F1/F2 scope
clarifications, the q-clock chain caveat); **draft v0.11** (`3c34a7d`)
repaired its collateral damage and landed the abstract fix its own script
contained but never applied. The candidates here are the uncommitted
remainder, kept for reference.
