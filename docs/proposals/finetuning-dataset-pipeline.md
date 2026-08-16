# Proposal: a dataset construction pipeline for the ticket→services classifier

Status: proposal, awaiting decisions in §9
Date: 2026-08-16
Companion documents:
- [`docs/specs/dataset-pipeline-cleaning.md`](../specs/dataset-pipeline-cleaning.md) — text processing, label auditing, adjudication (the *semantics*)
- [`docs/specs/dataset-pipeline-architecture.md`](../specs/dataset-pipeline-architecture.md) — stages, contracts, freezing, orchestration (the *runtime*)

This document is the decision-level summary: what we are building, what it costs, and what has
to be decided before anyone writes code.

**Nothing has been implemented.** All three documents are designs. No pipeline code, no configs,
no `.py` files exist in this repo.

---

## 1. Recommendation in one paragraph

Build a **four-phase batch pipeline** — deterministic cleaning, one bounded LLM call per row,
a **cheap-first label audit** that reaches for an LLM only on the residual, then corpus-level
assembly — as a **`make`-driven Python CLI writing Parquet checkpoints**, with a **row-level LLM
response cache** that makes every re-run free and every published number re-derivable without
calling a model provider at all. The deterministic phase does most of the work at **0.09 ms/row
[measured]**, removing 48.2% of sentence instances as boilerplate at 1.2% collateral damage. Label
auditing is a ladder of instruments ordered by cost: duplicate-conflict detection, a
cross-validated TF-IDF baseline, confident learning, embedding-neighbour agreement, and only then
an LLM judge — which on this corpus sees **6.2% of labelled rows [measured]** rather than all of
them. Human review lands at **~200 rows and ~25 person-hours [measured on the fixture]**, and the
whole build costs tens of dollars.

Two design invariants carry the most weight. **Phase 2 never rewrites the text the model reads** —
it emits spans, enums and candidate rules, which code compiles into deterministic rules that run
identically at training and inference; rewriting would force an LLM call onto the production
prediction path, which the parent proposal already rejected. And **no tier ever changes a label
automatically** — every instrument routes, humans decide. The second invariant was not a
principle we started with; it is what the measurements forced (§4.3).

**The binding constraint is not cost and not compute.** It is that the same preprocessing must run
at training and serving time or the model silently breaks, and that this pipeline handles
un-redacted production ticket text containing personal data.

---

## 2. What we are building

Input: `data/raw/tickets_export.csv` — 5,013 synthetic tickets, 4,622 with a non-empty `services`
column, 391 untriaged [measured]. Output: a **frozen, content-addressed training corpus** plus the
artifacts that make it auditable — a run report, a redaction audit, a dedup cluster map, a
**label-quality report**, a human-verdict log, and a manifest pinning every input, config, prompt,
fitted model and checkpoint version.

The ticket text carries pasted logs and tracebacks, code, personal names, e-mail addresses, phone
numbers, bank and transactional details, and company identifiers (ИНН/ОГРН/р-с). The brief's
premise was that this material is not useful to a classifier. **That is half right, and the half
it gets wrong is the expensive half** — see §4.2.

**Application secrets are out of scope**, on the stated premise that production `title` and
`description` do not carry API keys or program keys. The whole credential-scanning tier —
`gitleaks`, `detect-secrets`, entropy analysis — is removed rather than kept "just in case", since
a scanner with no threat to find is a dependency, a licence review and a gate that can only ever
fire falsely. Two hedges remain, both free: URL query-string stripping survives on its own merits
(high-entropy, per-row-unique, zero-signal), and the Phase-2 residual-PII enum keeps an `other`
bucket, so credential-shaped strings recurring there is evidence to reopen the premise at zero
cost. Reinstating the tier would be a version bump and a re-run, not a redesign.

---

## 3. The pipeline

| Phase | What it does | LLM | Cost | Owner doc |
|---|---|---|---|---|
| **0 — ingest** | CSV → Arrow, canonicalise, snapshot hash | no | free | arch §3.1 |
| **1 — deterministic** | Unicode/`ftfy` repair, whitespace and quote normalisation, `services`/`labels` hygiene, boilerplate and signature stripping, log-skeleton removal, PII → stable placeholders, language ID, length filters, **egress gate** | **no** | **0.09 ms/row** [measured] | cleaning §4.1 |
| **2 — bounded LLM** | ≤1 accepted call/row: span classification, rule mining, row metadata, residual-PII second opinion. **Emits no prose** | 1/row | ~$19 [estimate] | cleaning §4.2 |
| **3 — label audit** | Six tiers, cheapest first (below) | residual only | small | cleaning §4.3 |
| **4 — corpus-level** | Dedup, temporal split with gap, class balance, quality gates, freeze | no | free | arch §3.5–3.7 |

**Phase 4 is an addition to the requested three, and it is mandatory.** Dedup, split assignment,
class balance and gates are whole-corpus operations that cannot live in a per-row map, and the
runbook already requires them (§4.3, §6).

### 3.1 Phase 3 — the cascade

All figures [measured] on the fixture.

| Tier | Instrument | Cost | Contamination | Selects |
|---|---|---|---|---|
| **1** | Graded label conflicts within duplicate groups | seconds | **none** | ~14 queued (9 clusters / 71 rows carry disagreeing label sets) |
| **2** | TF-IDF + one-vs-rest LR, `GroupKFold` on dedup clusters | **113 s CPU** | **none** | feeds tiers 3 and 5 |
| **3** | `cleanlab` 2.9.0 confident learning | seconds | **none** | **161 rows** |
| **4** | kNN label agreement + a once-only UMAP taxonomy diagnostic | minutes | **none** | 56, corroborating only |
| **5** | LLM judge, constrained adjudication with evidence spans | ~375–600 calls | **real** | **288 rows = 6.2%** |
| **6** | Human review | ~8 person-hours | none | **~200 rows** |

Tiers 1–4 are free, deterministic given a seed, and carry **no contamination risk whatsoever** —
none of them writes a label opinion into the corpus. That is why they run first, and it is why the
LLM tier is now a residual filter rather than the primary instrument. The earlier design sent
every labelled row to a judge; this one sends 6.2% of them.

Two ordering constraints are load-bearing: **dedup precedes everything** (sampling before dedup
means paying reviewers to adjudicate forty copies of the same auto-generated alert, and tier 2's
CV folds key on `dedup_cluster_id`), and **dedup reads Phase-1 text, not Phase-2 output** (clusters
must be stable across prompt changes, or editing the Phase-2 prompt silently reshuffles the split
boundary and every metric with it).

---

## 4. The findings that shaped the design

Each of these contradicts something in the brief I gave the subagents. Each is argued in full in
the companion specs.

### 4.1 Phase 2 must not rewrite the text

Classifier spec §4.6 and proposal §5.3 require **identical preprocessing at training and
inference**. If Phase 2 rewrites training text, avoiding train/serve skew requires running Phase 2
at inference — an LLM call per ticket on the main prediction path, blowing the 42.9 ms / 250 ms p95
CPU budget by two orders of magnitude and reintroducing what §4 of the parent proposal rejected.

Phase 2 therefore gets three shapes, none of which touch the text: **rule mining** (output frozen
into `preprocess.json`, running deterministically in microseconds at both ends), **row metadata**
(facts *about* the row, for filtering and weighting), and **residual-PII detection**. One bounded,
schema-constrained call per row is preserved; hallucination on the text path becomes structurally
impossible. The architecture document reached a compatible constraint from the cost side — output
tokens dominate, so the schema returns an edit list, not prose.

This also forecloses fusing Phases 2 and 3, for a reason independent of cost: **Phase 2 must run
on the gold val/test rows** (or there is skew inside our own evaluation) and **Phase 3 must never**
([`llm-fallback-policy.md`](../specs/llm-fallback-policy.md) §5). The cascade adds a fourth
argument — the two phases now run on populations of different size, 100% versus 6.2%.

### 4.2 Logs and code are signal, not noise

The pasted evidence in these tickets is typically one line — `SignatureDoesNotMatch`,
`429 Too Many Requests`, `401 Unauthorized`, `Seq Scan` — and is frequently **the most
discriminative token in the row**. Deleting log blocks wholesale would delete the signal. Only the
log *skeleton* goes: paths, line numbers, addresses, timestamps, hex offsets, base64 blobs. Same
for code — keep the exception class, drop the frames.

The same distinction runs through the PII tier, and it is sharper than it first looks: **money
amounts are preserved** (36 rows; billing signal that generalises across tickets) while **account
numbers are not** (identifiers that never generalise). A detector that cannot tell those apart
costs accuracy.

A related measurement corrects the parent spec. §4.6 estimated placeholder normalisation
"typically recovers 10–30% of the token budget"; **[measured] here it recovers 2.1%** overall,
8.3% on the 922 rows containing anything to replace. Placeholders are justified by compliance and
skew-stability, **not** token budget. What recovers budget is boilerplate removal: 5.5%
corpus-wide, **65% on the 99 auto-alert rows**. §4.6 flagged its number as one to measure per
corpus; this is the measurement, and this corpus does not support it. Consequence: token p90 is
96, so **`max_len = 128`**, halving inference latency against the parent spec's assumption of 256.

### 4.3 Dropping the flagged rows made the model worse — so nothing auto-corrects

This is the result that changed the design, and it is the reason to run the cheap tiers before
committing to anything.

Removing the 161 cleanlab-flagged rows from training **hurt**: macro-AP 0.9381 → 0.9302
(−0.79pp), micro-F1 flat [measured]. On this corpus the flagged rows are **hard-but-correct**, not
mislabelled. The confident-learning ranking is doing its job — it surfaces rows where the model
disagrees with the label — but disagreement is not noise, and a pipeline that auto-drops or
auto-relabels on that signal would have quietly degraded the dataset while reporting a cleaner one.

Hence the invariant: **tiers 1–4 route, they never decide.** The delta is reported as a diagnostic
and explicitly **not** used as a gate.

### 4.4 Getting the statistics right

Three places where the obvious choice produces a number that means nothing:

- **`cleanlab`'s multi-class API is the wrong entry point.** `cleanlab.filter.find_label_issues`
  assumes one correct label per row and would flag every 2–4-service ticket. The multi-label
  surface is `cleanlab.multilabel_classification.{filter,rank,dataset}`, and `labels` is an
  iterable of iterables of class indices, not a binary matrix. Verified against the live 2.9.0
  docs rather than recalled.
- **CV folds must group on dedup cluster, not ticket ID.** Measured leakage from plain `KFold`:
  micro-F1 0.9397 → 0.9488 (+0.91pp), with macro-AP moving the *other way* — which is itself the
  diagnostic signature. A row's duplicate in its own training fold inflates its confidence and
  hides the very error being hunted.
- **Cohen's κ is wrong for this target.** The label matrix is 8,771 positives in 92,440 cells, so
  **90.51% of cells are agreed negatives**; a binary flattening yields a κ dominated by agreement
  that `terraform-provider` does not apply, chance-corrected against a marginal that mixes an
  851-positive service with a 15-positive one. Treating the set as a category gives 2²⁰
  categories, `p_e → 0`, and κ collapses to raw agreement. Use **Krippendorff α with MASI**
  (already the project's metric, runbook §4.2) plus **per-service κ as a vector with
  `n_positives`, never averaged**. Note also that the α ≥ 0.67 gate is an *annotator* gate and
  does not transfer to judge-versus-stored comparisons.

### 4.5 "59% of rows are ambiguous" became 71 rows

The taxonomy's deliberate ambiguity (`auth`/`access-control`, `monitoring`/`logging`,
`console-ui`/component) touches 59.1% of rows — 2,730 of them. Routing all of those to review
would have been the naive reading. Instead the cascade routes the **71** where tier-2 probabilities
disagree with the stored label *in the direction the taxonomy is known to be weak*. That is the
single largest reduction in the design, and it only exists because tier 2 exists.

---

## 5. Tooling, and why

All MIT, Apache-2.0 or BSD. Full reasoning, runners-up and switch conditions in the companion
specs.

| Job | Chosen | Why, and what lost |
|---|---|---|
| Text repair | **`ftfy`** + stdlib `unicodedata` | Mojibake repair is not worth hand-rolling. **`unidecode` rejected** — it would Latinise 67% of this corpus |
| PII | **`presidio`** (Apache-2.0) with **`slovnet`** as the RU NER backend | Presidio supplies the recogniser/anonymiser framing; its own Russian coverage is weak. **`scrubadub` rejected: no Russian at all.** Stated honestly: slovnet's published ORG F1 is 0.825, so roughly one company name in six is missed — which is why the egress gate exists |
| Language ID | **`lingua-py`** | The only candidate with span-level code-switch output. This corpus has 237 rows of Russian prose with English error strings, and a Cyrillic-ratio rule gets 9 of them right. **fastText `lid.176` rejected on CC-BY-SA-3.0** |
| Near-duplicates | **`datasketch`** MinHashLSH, two signatures unioned | **SimHash rejected: worse ARI at 45× the cost.** Two signatures because the 98-row alert clique is only visible at Jaccard 0.80 *before* boilerplate stripping — after it the largest cluster drops to 8 |
| Label audit | **`scikit-learn`** (BSD-3) + **`cleanlab` 2.9.0** (Apache-2.0) + **`umap-learn`** (BSD-3) | Confident learning is the standard instrument for this and is well grounded (Northcutt et al., JAIR 2021). TF-IDF + LR was already a mandatory baseline in the parent spec — it was being computed and thrown away; here it also produces the out-of-sample probabilities tier 3 needs |
| Dataframes | **`polars`** + Parquet/Arrow | Columnar checkpoints, stable schema at every boundary |
| Orchestration | **GNU `make` + a `ticketds` Typer CLI** | A 20-minute single-host batch job that gets iterated on, blocked once by a human. **Temporal rejected** (payloads force file-path passing; the human pause is a file, not a signal). **DVC was the strongest rival** — rejected because its cache would put un-redacted raw text in a second place we must remember to shred, and because it cannot express "the prompt changed, reuse 4,900 of 5,013 cached responses" |
| Human review | **Generated XLSX round-trip** (`openpyxl`), HMAC-checked | ~200 rows, ≤3 reviewers, one batch. Zero infrastructure, zero auth, zero residency question. **Argilla is the named upgrade trigger** if the queue exceeds ~1,500 rows or review becomes recurring. **Prodigy rejected**: closed source in a residency-sensitive pipeline, per-seat cost exceeding the project's entire LLM budget |

---

## 6. What it costs

| Resource | Full build | Notes |
|---|---|---|
| LLM | **~$25–30** [estimate] | Phase 2 (~5,013 calls, ~$19) now dominates; Phase 3 runs on 288 rows. The architecture doc's $32 was computed against a 1,400-row residual estimate and is an upper bound |
| Rebuild after a code-only change | **$0** | Every response served from the replay cache |
| Machine | ~20 min CPU; **2–4 h elapsed** | Dominated by batch turnaround. 4 vCPU, 8 GB RAM, **no GPU** — embedding 50k rows is ~6 min on CPU |
| Human | **~25 person-hours**, of which ~8 h is the review queue | Balance is the Phase-3 calibration pilot and per-phase pipeline validation |

**Cost is not a decision variable.** A full build costs less than an hour of engineering time. Any
review argument beginning "to save on LLM calls" should be treated with suspicion.

One consequence of the cascade is worth noting because it reverses an earlier decision:
**2-of-3 self-consistency on the judge was previously ruled out on budget** at 15,039 calls. Over a
few-hundred-row residual it is affordable, so three opinions on the rows that matter now costs less
than one opinion on every row did. It is back in the design.

### 6.1 These numbers will grow on real data, and that is the plan

The fixture's labels are author-assigned and internally consistent — 564 of 565 same-title groups
agree exactly, and cleanlab flags 2.06% of rows. **A real CatBoost-contaminated `services` column
should show 10–20% noise**, implying a queue of ~400–700 rows and 17–29 person-hours. The 600-row
capacity cap in the architecture document is sized for that case, not for the fixture.

**Measuring the real rate is now the cheapest unanswered question in the project**: tiers 1–3 are
four minutes of CPU, no LLM, no human, no budget approval. It should run in week 1, *before* the
annotation guideline is written, because the taxonomy diagnostics should inform it.

---

## 7. What this pipeline does not fix

- **`tickets_export.csv` has author-assigned labels and no `ticket_messages` companion.**
  Provenance Cases A/B/C/D, the correction rate and Krippendorff α are **unavailable from this
  file**. Every number the runbook §2 asks for in week 1 needs the real export.
- **This pipeline does not produce a gold set.** It produces a training corpus and a
  quality-controlled review queue. The blind-annotated gold set (2,000 test + 1,000–1,500 val,
  runbook §4.2, ~70 person-hours) is separate and non-negotiable, and **the review queue is
  additive to it, never a substitute.** The queue is model-filtered, non-blind and single-reviewer:
  it cannot surface silent errors and cannot produce an agreement statistic. Substituting it would
  rebuild the Case-A selection bias with an LLM in place of CatBoost.
- **A label changed on the judge's advice is model provenance and can never be Case A.** So is
  `human_after_judge` — a reviewer agreeing with a judge proposal was anchored by it, and those
  rows are excluded from val/test too.
- **Two measurements behind the cascade are proxies**, recorded rather than buried: the
  drop-flagged-rows delta used global out-of-fold probabilities (mildly circular; the
  implementation must use nested CV), and the tier-4 kNN numbers are a TF-IDF cosine stand-in for
  embeddings.

### 7.1 One correction to an existing document

**Runbook §6's "group by `organization_id`" is not implementable alongside a temporal split on
this corpus.** Measured independently by both subagents: 206 of 210 organisations span more than
one split, covering 99.5% of rows, and **100% of test rows belong to an organisation that also
appears in train — a strict purge leaves zero test rows.**

The purpose of the rule is real; the mechanism is unavailable. Replacement: cluster grouping plus
an **org-conditioned near-duplicate purge at Jaccard ≥ 0.50**, costing 9 test rows (1.4%) and 15
val rows (2.4%), with `org_overlap_rate` and a seen-org/unseen-org metric split reported
permanently. The actual leakage surface is small and specific: 9 near-duplicate clusters, 120 rows.

Runbook §6 should be amended so the next reader is not misled.

---

## 8. The shape of the risk changed when secrets left

Removing credential scanning does **not** soften the handling posture, and the reasoning is worth
stating because the next person to argue for looser controls will argue against whatever is
written down.

Credential exposure is fast, exploitable and **rotatable**. Personal-data exposure is slow,
un-revocable and regulated. **152-FZ is a personal-data regime and is entirely untouched by this
change.** So the *detection* tier shrinks and the *handling* controls do not:

- The **egress gate survives on a new argument**: moving personal data is itself the regulated
  act, regulated whether or not a key travelled with it, and the gate is the only place that act is
  prevented rather than audited afterwards. With no rotation step available, prevention is worth
  disproportionately more than detection.
- The gate becomes **two-sided**. With secrets gone, the only false-negative class is personal data
  and the only false-positive class is label signal — so over-redaction is now a blocking check in
  the same stage. Splitting the two directions across two stages is how one quietly stops being
  enforced.
- **Retention stays at 30 days**, on the data-minimisation reason rather than the credential one.
  It always had two justifications; the surviving one is the stronger.

---

## 9. Decisions required

| # | Decision | Owner | Recommendation |
|---|---|---|---|
| **D-1** | **152-FZ: may pseudonymised ticket text leave the perimeter to a hosted LLM API?** | Legal / DPO | Blocks Phases 2–3 on production data, not on the fixture. Default config to `strict`, build the self-hosted provider path in parallel so a "no" costs a week rather than a redesign |
| **D-2** | **Is the no-application-secrets premise a measurement or an assumption?** | Backend + security | If assumed, it costs one grep over a production export to check. The pipeline hedges either way, and reinstating the tier is a version bump plus a re-run |
| **D-3** | **Is a self-hosted LLM available in-perimeter, and at what tier?** | Infra | Needed only if D-1 is "no". The provider seam makes it a config change |
| **D-4** | **"At most one LLM call per row" — one *attempt* or one *accepted response*?** | User | One accepted response; 3 attempts max. One-attempt-only routes ~0.2–0.5% of rows to a human for a *formatting* failure, spending reviewer time on a machine problem |
| **D-5** | **Run tiers 1–3 on a real export in week 1** to get the true label-noise rate | ML | Do it first. Four minutes of CPU, no LLM, no human, no approval — and it sizes everything downstream, including whether Phase 3's LLM tier is worth building |
| **D-6** | **Human budget: ~25 person-hours (~8 h queue), in addition to the ~70 h gold set** | Support lead | Confirm. **If only 70 h exist in total, spend all of it on the gold set and drop Phase 3.** Do not fund the queue out of the gold budget |
| **D-7** | **Amend runbook §6** — org-grouped temporal splitting is infeasible (§7.1) | ML + product | Adopt the measured replacement and edit the runbook |
| **D-8** | **Who authors the `auth` vs `access-control` tie-break rules?** | Product owner | Without written rules, judge and reviewer disagree on the same rows forever. This is the taxonomy's largest error source |
| **D-9** | **Which embedding checkpoint for tier 4, pinned at which commit SHA?** | ML | Keep it **decoupled** from the production classifier checkpoint even if the same model is a candidate for both — coupling means changing the production model silently re-ranks the label audit of the corpus that trained it |
| **D-10** | **Raw-export retention: 30 days post-freeze** | Legal + ML | 30 days, on data minimisation (§8) |
| **D-11** | **Repo layout `py/` rather than `pipeline/`** | User | `py/`. `packages/` reads as npm workspaces to every TypeScript developer who opens this repo |
| **D-12** | **Does the real input arrive as a CSV export or a read-replica query?** | Backend / DBA | Either works; a query must archive its text and hash. A live query with no snapshot is the one thing forbidden |

**One decision is deliberately deferred to measurement rather than judgement.** Whether the LLM
judge ships at all is decided by two numbers from a 100-row sample: cheap-tier precision and judge
yield. High cheap-tier precision plus low judge yield → **drop the judge**, and the corpus then
contains zero model-provenance labels from this pipeline. Low cheap-tier precision → keep it, as a
triage filter that reduces human load. Counterintuitively, the *better* the cheap tiers turn out to
be, the stronger the case for deleting the expensive tier.

---

## 10. Sequencing

The valuable property: **the first five steps have no LLM dependency and now end with a complete
label-quality report.**

| Step | Deliverable | Blocked by |
|---|---|---|
| 1–2 | Repo skeleton, `make` DAG, ingest, run report | — |
| 3–4 | `ticketprep` package: normalise + redact + egress gate, golden test file | — |
| 5a–d | Dedup, split, assemble, gates, freeze — wired straight from Phase 1 | — |
| **5e** | **Conflict detection, TF-IDF baseline, confident learning, kNN + UMAP** | — |
| 6–7 | LLM layer, replay cache, Phase 2 at `--limit 200`, then full corpus | D-1 |
| 8 | Freeze verification + CI rebuild asserting version stability | — |
| 9–11 | Judge on the residual, queue, XLSX round-trip, human review | D-6, D-8 |
| 12 | Model-service startup fingerprint assert + cross-process conformance CI | — |
| 13 | Self-hosted provider + benchmark | D-3 |

**Step 5e is the milestone that matters.** Previously the LLM-free envelope stopped at clean text
and splits, saying nothing about label quality. Now it answers *how much noise, where, and in which
services* before spending a cent, writing a judge prompt, or hearing back from legal — and if that
report says noise is negligible, Phase 3 may not be worth building at all. That is the cheapest
kind of win available here.

Step 12 deserves emphasis: the train/serve skew seam is closed by **one installable package with a
computed fingerprint**, where the model service refuses to start on a mismatch. Any change to
preprocessing output is a MAJOR bump invalidating the LLM cache, the corpus and the model artifact
together — correctly, because they are one unit.

---

## 11. Reading order

1. This document — decisions and cost.
2. [`docs/specs/dataset-pipeline-cleaning.md`](../specs/dataset-pipeline-cleaning.md) — what clean
   means: corpus inventory, detector stack, both LLM phase specifications, the cascade, gates.
3. [`docs/specs/dataset-pipeline-architecture.md`](../specs/dataset-pipeline-architecture.md) —
   how it runs: stage DAG, schemas, freezing and replay, fitted artifacts, the human loop,
   security zones.
4. [`docs/specs/dataset-construction-runbook.md`](../specs/dataset-construction-runbook.md) — the
   week-1 procedure this pipeline sits inside (note the §7.1 correction).
5. [`docs/proposals/ticket-services-classifier.md`](./ticket-services-classifier.md) — why any of
   this exists.
