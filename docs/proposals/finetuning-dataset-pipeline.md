# Proposal: a dataset construction pipeline for the ticket→services classifier

Status: proposal, awaiting decisions in §9
Date: 2026-08-16
Companion documents:
- [`docs/specs/dataset-pipeline-cleaning.md`](../specs/dataset-pipeline-cleaning.md) — text processing, LLM cleaning, label adjudication (the *semantics*)
- [`docs/specs/dataset-pipeline-architecture.md`](../specs/dataset-pipeline-architecture.md) — stages, contracts, freezing, orchestration (the *runtime*)

This document is the decision-level summary: what we are building, what it costs, where the two
specifications disagree with the brief and with each other, and what has to be decided before
anyone writes code.

**Nothing has been implemented.** Both companion documents are designs. No pipeline code, no
configs, no `.py` files exist in this repo.

---

## 1. Recommendation in one paragraph

Build a **four-phase batch pipeline** — deterministic cleaning, one bounded LLM call per row,
LLM-as-judge over the labels with a human review queue, then corpus-level assembly — as a
**`make`-driven Python CLI writing Parquet checkpoints**, with a **row-level LLM response cache**
that makes every re-run free and every published number re-derivable six months later without
calling a model provider at all. The deterministic phase does most of the work and costs nothing:
**0.09 ms/row [measured]**, removing 48.2% of sentence instances as boilerplate at 1.2% collateral
damage. The two LLM phases together cost **~$50–65 per full build [estimate]** and are worth
running only if a specified ablation says they are. The single most important design decision is
that **Phase 2 never rewrites the text the model reads** — it emits spans, enums and candidate
rules, which code then compiles into deterministic rules that run identically at training and at
inference. Rewriting the text would force an LLM call onto the production prediction path, which
the parent proposal already rejected.

**The binding constraint is not cost and not compute. It is that this pipeline handles
un-redacted production ticket text containing real secrets, and that the same preprocessing must
run at training and at serving time or the model silently breaks.** Everything below is shaped by
those two facts.

---

## 2. What we are building

Input: `data/raw/tickets_export.csv` — 5,013 synthetic tickets, 4,622 with a non-empty `services`
column, 391 untriaged [measured]. Output: a **frozen, content-addressed training corpus** plus the
side artifacts that make it auditable — a run report, a redaction audit, a dedup cluster map, a
human-verdict log, and a manifest that pins every input, config, prompt and model version.

The ticket text is dirty in ways that matter. Descriptions carry pasted logs and tracebacks, code,
API keys and tokens in URL query strings, bank- and card-shaped digit runs, personal names,
e-mail addresses, phone numbers, and company identifiers (ИНН/ОГРН). The brief's premise was that
this material is not useful for a classifier. **That is half right, and the half it gets wrong is
the expensive half** — see §4.

---

## 3. The pipeline

| Phase | What it does | LLM | Cost | Owner doc |
|---|---|---|---|---|
| **0 — ingest** | CSV → Arrow, canonicalise, snapshot hash | no | free | arch §3.1 |
| **1 — deterministic** | Unicode/`ftfy` repair, whitespace and quote normalisation, `services`/`labels` hygiene, boilerplate and signature stripping, log-skeleton removal, PII/secret detection → stable placeholders, language ID, length filters, **egress gate** | **no** | **0.09 ms/row** [measured] | cleaning §4.1 |
| **2 — bounded LLM** | ≤1 accepted call/row: span classification, rule mining, row metadata, residual-PII second opinion. **Emits no prose** | 1 call/row | ~$19 [estimate] | cleaning §4.2 |
| **3 — judge + human** | Per-`(ticket, service)` accept/reject verdicts with required evidence spans, 2-of-3 self-consistency by candidate-order shuffling, then a capped human review queue | ~1.3 calls/labelled row | ~$30–45 [estimate] | cleaning §4.3 |
| **4 — corpus-level** | Dedup, temporal split with gap, class-balance report, quality gates, freeze | no | free | arch §3.5–3.7 |

**Phase 4 is an addition to the brief, and it is mandatory.** Deduplication, split assignment,
class balance and quality gates are whole-corpus operations. They cannot live in a per-row map,
and the runbook already requires them (§4.3, §6). This is not a redesign of the three phases —
it is naming the thing that has to follow them.

Two ordering decisions carry real weight:

- **Dedup reads Phase-1 text, not Phase-2 output.** If clusters were computed on LLM-touched
  text, changing the Phase-2 prompt would silently reshuffle the split boundary and every metric
  with it.
- **Dedup precedes the review queue.** The queue is a sample; sampling before dedup means paying
  reviewers to adjudicate forty copies of the same auto-generated alert.

---

## 4. The four arguments that shaped the design

These are the load-bearing conclusions. Each contradicts something in the brief I gave the
subagents, and each is argued in full in the companion specs.

### 4.1 Phase 2 must not rewrite the text

The brief asked for a "bounded rewrite". Rewrite is the wrong verb. Classifier spec §4.6 and
proposal §5.3 require that **identical preprocessing runs at training and at inference**. If
Phase 2 rewrites training text, then avoiding train/serve skew requires running Phase 2 at
inference — an LLM call per ticket on the main prediction path, blowing the 42.9 ms / 250 ms p95
CPU budget by two orders of magnitude and reintroducing exactly the architecture §4 of the parent
proposal rejected.

So Phase 2 gets three legitimate shapes, none of which touch the text: **rule mining** (its output
is frozen into `preprocess.json` and runs deterministically at train and serve in microseconds),
**row metadata** (facts *about* the row, used for filtering and weighting), and **residual-PII
detection** (a second opinion feeding the redaction audit). This preserves the user's design —
one bounded, schema-constrained call per row — while making hallucination on the text path
structurally impossible.

The architecture document arrived at a compatible constraint from the cost side: output tokens
dominate, so the schema must return an edit list rather than rewritten text. The two specs
converged on this independently, which is a good sign.

### 4.2 "Logs and code are not useful for a classifier" is measurably wrong here

The pasted evidence in these tickets is typically one line — `SignatureDoesNotMatch`,
`429 Too Many Requests`, `401 Unauthorized`, `Seq Scan` — and it is frequently **the most
discriminative token in the row**. Deleting log blocks wholesale would delete the signal.

The resolution is to remove the log *skeleton* (file paths, line numbers, IP addresses,
timestamps, hex offsets, base64 blobs) and keep the error identifiers. Same for code: keep the
exception class name, drop the stack frames.

A related measurement corrects an estimate in the parent spec. Classifier spec §4.6 estimated
that placeholder normalisation "typically recovers 10–30% of the token budget". **[measured] on
this corpus it recovers 2.1%** overall, 8.3% on the 922 rows that contain anything to replace.
Placeholders here are justified by compliance and skew-stability, **not** by token budget. What
does recover budget is boilerplate removal: 5.5% corpus-wide and **65.1% on the 99 auto-alert
rows** (217.5 → 76.0 tokens). §4.6 explicitly flagged its number as one to measure per corpus;
this is the measurement, and this corpus does not support it.

Consequence: token p90 is 96, so **`max_len = 128`**, not 256 — which halves inference latency
against the parent spec's assumption.

### 4.3 Phases 2 and 3 cannot be fused into one call

The brief offered fusion as an option to stay inside the one-call-per-row budget. It is
foreclosed by the contamination rule, not by cost: **Phase 2 must run on the gold val/test rows**
(their text must be preprocessed identically to training text, or there is skew inside our own
evaluation), and **Phase 3 must never run on the gold rows** ([`llm-fallback-policy.md`](../specs/llm-fallback-policy.md) §5).
One call cannot satisfy both. Fusing would also destroy the ablation that decides whether either
phase ships at all.

Read the budget as **one accepted response per row per phase**. Phase 2 and Phase 3 are separate
phases with separate budgets. Confirm or correct this in D-4.

### 4.4 Entropy-based secret detection does not work on this corpus

Of 81 URL-borne secrets, **zero** trip `detect-secrets`' base64 entropy limit (4.5) and only 49
trip the hex limit (3.0) [measured]. Entropy heuristics assume long random strings; these are
short tokens in query parameters.

The answer is structural rather than statistical: **drop URL query strings unconditionally**,
which kills 81/81 by construction. More generally — a deterministic regex pass is not a
compliance story, and Phase 2 is not one either, because by the time the LLM sees the text it has
already seen the text. The cleaning spec §4.1.13 specifies six layers of defence, of which the
egress gate (a re-scan that refuses to release text, placed *before* anything can leave the
perimeter) is the one that actually holds.

A companion finding shapes the ordering: **a PERSON name detector will eat the signal.** 45% of
English `Firstname Lastname`-shaped matches in this corpus are strings like `Service
Unavailable`, `Bad Gateway`, `Seq Scan` [measured]. Protected-span masking therefore strictly
precedes detection, and a gate caps over-redaction at ≤0.5%.

---

## 5. Tooling, and why

Every choice below is MIT or Apache-2.0 unless noted. Full reasoning, runners-up and switch
conditions are in the companion specs.

| Job | Chosen | Why, and what lost |
|---|---|---|
| Text repair | **`ftfy`** + stdlib `unicodedata` | Mojibake repair is not something to hand-roll. **`unidecode` rejected outright** — it would Latinise 67% of this corpus |
| PII orchestration | **`presidio`** (Apache-2.0) with **`slovnet`** as the RU NER backend | Presidio gives the recogniser/anonymiser framing and a placeholder story; its own Russian coverage is weak, hence slovnet. **`scrubadub` rejected: no Russian coverage at all** |
| Secrets | **`gitleaks`** MIT rule pack + **`detect-secrets`** keyword plugins | Curated, maintained regex corpora. **`trufflehog` is AGPL-3.0** — usable out-of-band for verification, not linkable into the pipeline |
| Language ID | **`lingua-py`** | The only candidate with span-level code-switch output, which this corpus needs: 237 rows are Russian prose with English error strings, and a Cyrillic-ratio rule classifies only 9 of them correctly. **fastText `lid.176` rejected on CC-BY-SA-3.0** |
| Near-duplicates | **`datasketch`** MinHashLSH, two signatures unioned | **SimHash rejected: worse ARI at 45× the cost.** Two signatures because the 98-row alert clique is visible at Jaccard 0.80 only *before* boilerplate stripping — after it, the largest cluster drops to 8 |
| Dataframes | **`polars`** + Parquet/Arrow | Columnar checkpoints, stable schema at every stage boundary |
| Orchestration | **GNU `make` + a `ticketds` Typer CLI** | This is a 20-minute single-host batch job that gets iterated on, blocked once by a human. **Temporal rejected** (payloads force file-path passing; the human pause is a file, not a signal; a second Python worker fleet for one job). **DVC was the strongest rival** — rejected because its cache would put un-redacted raw text in a second place we must remember to shred, and because it cannot express "the prompt changed, reuse 4,900 of 5,013 cached responses" |
| Human review | **Generated XLSX round-trip** (`openpyxl`) with HMAC-checked tokens | 600 rows, ≤3 reviewers, one batch. Zero infrastructure, zero auth, zero residency question. **Argilla is the named upgrade trigger**: adopt it if the queue exceeds ~1,500 rows, reviewers exceed 3, or review becomes recurring. **Prodigy rejected**: closed source in a residency-sensitive pipeline, and per-seat cost exceeds the project's entire LLM budget |

---

## 6. What it costs

| Resource | Full build | Notes |
|---|---|---|
| LLM | **$16 Haiku 4.5 / $46 Sonnet 5 / $77 Opus 5**; **~$65** recommended mixed tier | Batch pricing. The two specs' bottom-up estimates ($19 + $30–45) agree with this |
| Rebuild after a code-only change | **$0** | Every response served from the replay cache |
| Machine | ~15 min CPU; **2–4 h elapsed** | Dominated by batch turnaround. 1 host, 4 vCPU, 8 GB RAM, **no GPU** |
| Human review queue | **~600–650 rows** | Person-hours disputed — see §7 |

**Cost is not a decision variable.** A full build costs less than an hour of engineering time. Any
argument in review that begins "to save on LLM calls" should be treated with suspicion. The 50%
batch discount and the 90% cache-read discount also **do not stack reliably** (5-minute cache TTL
versus uncontrolled batch turnaround), so do not budget for both.

---

## 7. Where the two specs disagree with each other

Both were written independently against the same brief. They converged on the big calls — the
Phase-2 no-rewrite constraint, the infeasibility of org-grouped temporal splits, the contamination
fence — and diverge in two places that need a decision rather than a patch.

### 7.1 Human review costs 12.5 hours or 46 hours, and the gap is not rounding

| | Architecture doc | Cleaning doc |
|---|---|---|
| Queue size | 600 rows (a capacity **cap**) | ~650 rows (a bottom-up **estimate**) |
| Per-item time | 75 s | 2.5 min |
| Queue subtotal | 12.5 h | 27 h |
| Pipeline-validation samples | not counted | +19 h (pilot + per-phase validation) |
| **Total** | **12.5 h** | **46 h** |

The row counts agree. The per-item rate does not, and **the cleaning doc's rate is the defensible
one**: the runbook's own 60 s/ticket figure is for blind single-pass annotation of a *random*
sample, whereas the queue is by construction the hard rows — ambiguous multi-label verdicts where
the reviewer reads an evidence span and adjudicates a service-by-service disagreement.

Recommended reconciliation: **keep the architecture doc's 600-row cap** (it is a budget control,
which is the right instrument) and **adopt the cleaning doc's 2.5 min rate**, giving ~25 h for the
queue plus ~19 h of validation. Plan for **~44 person-hours, not 12.5**. This roughly doubles the
ask in D-5 and it should be presented to the support lead at the honest number.

### 7.2 The gold-set fence is specified but not wired

The cleaning doc is unambiguous that Phase 3 must never touch gold val/test rows. The architecture
doc carries an `is_gold` column in the corpus schema, but its stage DAG runs `s20 → s30` over all
rows without an explicit exclusion. On the current fixture this is harmless — there is no gold set
in this file at all — but it is exactly the kind of gap that survives into production and quietly
contaminates the only uncontaminated evaluation stream we have.

**Fix before implementing Phase 3:** make `is_gold` a hard input filter on `s30`, and make it a
gate that fails the build rather than a convention.

---

## 8. What this pipeline does not fix

Stated plainly, because the fixture invites over-claiming:

- **`data/raw/tickets_export.csv` has author-assigned labels and no `ticket_messages` companion.**
  Provenance Cases A/B/C/D, the correction rate, the provenance slice and Krippendorff α are all
  **unavailable from this file.** Every number the parent runbook §2 asks for in week 1 needs the
  real export.
- **This pipeline does not produce a gold set.** It produces a training corpus and a
  quality-controlled review queue. The blind-annotated gold set (2,000 test + 1,000–1,500 val,
  runbook §4.2, ~70 person-hours) is a separate, non-negotiable piece of work, and **the review
  queue is additive to it, never a substitute.** The queue is a model-filtered, non-blind,
  single-reviewer set: it cannot surface silent errors and cannot produce an agreement statistic.
  Substituting it rebuilds the exact Case-A selection bias the runbook exists to prevent, with an
  LLM in place of CatBoost.
- **A label changed on the judge's advice is model provenance and can never be Case A.** So is
  `human_after_judge` — a reviewer agreeing with a judge proposal was anchored by it, and those
  rows are excluded from val/test too.
- **Whole detector classes have zero instances in this fixture** — JWTs, AWS keys, real
  tracebacks, DB connection strings. They are exercised only by a seeded canary set, and their
  real-world recall is unmeasured until the pipeline runs on a production export.

### 8.1 One correction to an existing document

**Runbook §6's "group by `organization_id`" is not implementable alongside a temporal split on
this corpus.** Measured independently by both subagents: 206 of 210 organisations span more than
one temporal split, covering 99.5% of rows; **100% of test rows belong to an organisation that
also appears in train, and a strict org purge leaves zero test rows.**

The purpose of the rule — prevent leakage — is real; the mechanism is not available. Replacement:
cluster grouping plus an **org-conditioned near-duplicate purge at Jaccard ≥ 0.50**, which costs
9 test rows (1.4%) and 15 val rows (2.4%), with `org_overlap_rate` and a seen-org/unseen-org
metric split reported permanently. The actual leakage surface is small and specific: only 9
near-duplicate clusters (120 rows) straddle a split boundary.

Runbook §6 should be amended so the next reader is not misled.

---

## 9. Decisions required

Consolidated from both specs, de-duplicated, blocking items first.

| # | Decision | Owner | Recommendation |
|---|---|---|---|
| **D-1** | **152-FZ: may pseudonymised ticket text leave the production perimeter to a hosted LLM API?** | Legal / DPO | Blocks Phase 2 and 3 on production data; does **not** block the synthetic fixture. Default config to `strict` + `production` so the unsafe combination requires a visible diff, and build the self-hosted provider path in parallel so a "no" costs a week rather than a redesign |
| **D-2** | **Is a self-hosted LLM available in-perimeter, and at what quality tier?** | Infra | Needed only if D-1 is "no". The provider seam is designed so this is a config change |
| **D-3** | **"At most one LLM call per row" — one *attempt* or one *accepted response*?** | User | One accepted response; 3 attempts max; hard per-run budget of 1.05 × rows. One-attempt-only routes ~0.2–0.5% of rows to human review for a *formatting* failure, which spends reviewer time on a machine problem |
| **D-4** | **Human budget: ~44 person-hours for the queue and pipeline validation, in addition to the ~70 h gold set** (§7.1) | Support lead | Present the reconciled number, not the 12.5 h figure. **If only 70 h exist in total, spend all of it on the gold set and drop Phase 3.** Do not fund the queue out of the gold budget |
| **D-5** | **Amend runbook §6** — org-grouped temporal splitting is infeasible (§8.1) | ML + product | Adopt the measured replacement and edit the runbook |
| **D-6** | **Who authors the `auth` vs `access-control` decision rules?** | Product owner | The taxonomy's deliberate ambiguity is the judge's largest error source. Without written tie-break rules, judge and reviewer disagree on the same rows forever |
| **D-7** | **Fence `is_gold` out of Phase 3 as a build-failing gate, not a convention** (§7.2) | ML | Do it before Phase 3 is implemented, not after |
| **D-8** | **Raw-export retention: 30 days post-freeze** | Legal + ML | 30 days. Longer means a larger standing exposure of real secrets |
| **D-9** | **Model tier: Sonnet 5 for Phase 2, Opus 5 for Phase 3 (~$65/build)** | ML | As proposed; revisit only if the judge shows no Opus/Sonnet gap on the audit stratum |
| **D-10** | **Repo layout `py/` rather than `pipeline/`** | User | `py/`. `packages/` reads as npm workspaces to every TypeScript developer who opens this repo. Cheap now, expensive after 200 imports exist |
| **D-11** | **Does the real input arrive as a CSV export or a read-replica query?** | Backend / DBA | Either works; a query must archive its text and hash. A live query with no snapshot is the one thing that is forbidden |

**Two things must be measured before the queue budget is committed:** the real base reject rate
(assumed 8%; at 20% the queue doubles and D-4's answer changes), and NER latency (which decides
whether name redaction can run at inference at all). Both come from a 200-row pilot.

---

## 10. Sequencing

The valuable property of this plan is that **the first five steps have no LLM dependency and
produce a complete, frozen, usable corpus.**

| Step | Deliverable | Blocked by |
|---|---|---|
| 1–2 | Repo skeleton, `make` DAG, ingest stage, run-report | — |
| 3–4 | `ticketprep` package: normalise + redact + egress gate, with a golden test file | — |
| **5** | **Dedup, split, assemble, gates, freeze — wired straight from Phase 1** | — |
| 6–7 | LLM layer, replay cache, Phase 2 at `--limit 200`, then full corpus | D-1 |
| 8 | Freeze verification + the CI job that rebuilds and asserts version stability | — |
| 9–11 | Judge, queue, XLSX round-trip, human review rounds | D-4, D-6, D-7 |
| 12 | Model-service startup fingerprint assert + cross-process conformance CI | — |
| 13 | Self-hosted provider + benchmark | D-2 |

**Step 5 is the milestone that matters.** It yields a frozen corpus with zero LLM involvement, so
ML baselines can start while the legal question (D-1) and the human budget (D-4) are still open.
If Phase 2 and Phase 3 are never funded, step 5 still leaves the project with a real dataset.

Step 12 deserves emphasis: the train/serve skew seam is closed by **one installable package with
a computed fingerprint**, where the model service refuses to start on a mismatch. Any change to
preprocessing output is a MAJOR bump that invalidates the LLM cache, the corpus and the model
artifact together — which is correct, because they are one unit.

---

## 11. Reading order

1. This document — decisions and cost.
2. [`docs/specs/dataset-pipeline-cleaning.md`](../specs/dataset-pipeline-cleaning.md) — what
   clean means: measured corpus inventory, the detector stack, both LLM phase specifications,
   the judge and routing rules, quality gates with thresholds.
3. [`docs/specs/dataset-pipeline-architecture.md`](../specs/dataset-pipeline-architecture.md) —
   how it runs: stage DAG, schemas at every boundary, the freezing and replay machinery, the
   human loop, security zones.
4. [`docs/specs/dataset-construction-runbook.md`](../specs/dataset-construction-runbook.md) —
   the surrounding week-1 procedure this pipeline sits inside (note the §6 correction in §8.1).
5. [`docs/proposals/ticket-services-classifier.md`](./ticket-services-classifier.md) — why any of
   this exists.
