# Fine-tuning dataset construction pipeline: text processing, LLM cleaning, and label adjudication

Status: specification (not implemented)
Author: ml-researcher
Date: 2026-08-16 (revision 5)
Input: [`data/raw/tickets_export.csv`](../../data/raw/tickets_export.csv) (5,013 rows)
Output: a versioned training corpus for the multi-label ticket→services classifier

**Revision history**, all folded into the text rather than appended:

1. **(r2) Application-secret detection is removed entirely.** The product owner states production
   `title`/`description` contain no secrets, API keys or program keys (§2.3). The `gitleaks` /
   `detect-secrets` rule packs, the entropy analysis, the secret canary classes and the
   secret-recall gates are gone. The **PII tier is unchanged and extended** with bank and
   transactional details, and unconditional URL query-string stripping survives as a plain
   normalisation rule (§4.1.4). The fixture contradicts the premise by construction — see §2.4.
2. **(r2) Phase 3 is restructured as a cheap-first cascade** (§4.3): deterministic label conflicts
   → TF-IDF + grouped CV → cleanlab confident learning → LLM judge on the residual only → human
   review. The judge sees **6.2% of labelled rows** and makes hundreds of calls instead of
   ~6,000. §4.3.12 states the condition under which the judge is dropped entirely.
3. **(r3) 152-FZ material removed.** The engineering constraint — ticket text is personal data and
   moving it outside the production perimeter needs approval — is kept and reframed as generic
   data governance (§2.9, §4.2.6, P1). The statutory reasoning is not ours to make.
4. **(r4) Tier 4 (embedding kNN + UMAP) is cut on measurement.** An independent adversarial
   validation ([`knn-tier-validation.md`](./knn-tier-validation.md)) showed it **reduces**
   injected-error recall at matched reviewer budget — 0.589 ± 0.015 versus **0.683 ± 0.022** for
   tier 3 alone at 185 rows — at every budget and under both representations, including on the
   ambiguous-pair case it was supposed to cover. §4.3.4 is now the closure record; the freed
   effort goes into **widening the tier-3 selector** (§4.3.11), and the UMAP deliverable is
   replaced by two free tables in §6.1. Gate **G18** now governs admission of any future
   row-selection tier. Queue: **~285 rows / ~28 h**.
5. **(r5) Phase 0 — ingest and profile added** (§4.0.1). A label-imbalance profile of the **input
   CSV**, computed in seconds before Phase 1 runs, because §6.1's class balance lands on the
   assembled corpus at the *end* of the build and the decision it informs is a scoping one. It
   reports per-service counts, prevalence, imbalance ratio, entropy/Gini, cardinality, and counts
   **projected per split** — [measured] `terraform-provider` and `message-queue` project to **one
   test positive each**, so 5 of 20 services cannot be given a precision number. Four T-tiers with
   consequences, gates **G19–G21**, and §4.0.1.4 on what imbalance does and does not imply for
   multi-label training.

**Companion documents — read these first, this one sits under them:**

| Document | Owns |
|---|---|
| [`ticket-services-classifier.md`](./ticket-services-classifier.md) | the task, splits (§2.6), features (§2.7), class balance (§2.8), PII/bias/licensing (§2.9), input construction and placeholder normalisation (§4.6), evaluation |
| [`dataset-construction-runbook.md`](./dataset-construction-runbook.md) | provenance reconstruction, gold-set sampling rules and volumes, the training-vs-evaluation selection principle (§0) |
| [`llm-fallback-policy.md`](./llm-fallback-policy.md) | constrained adjudication (§4.1), prompt caching (§4.3), **the contamination rule (§5)** |
| [proposal §3, §5.4](../proposals/ticket-services-classifier.md) | the provenance trap and the audit tables that permanently fix it |
| `dataset-pipeline-architecture.md` (`system-architect`, to be written) | **stage DAG, artifact formats and layout, caching, concurrency, retries, cost model, human-review-queue plumbing, snapshot manifests** |

This document specifies **what each phase does to the text and to the labels, and why**. It
does not specify how the phases are orchestrated, where they run, how they are retried, or how
the review queue is wired to humans. Every such statement below is written as a *requirement*
for the architecture document.

All numbers tagged **[measured]** were computed by me on `tickets_export.csv` during the
writing of this spec; the throwaway scripts are listed in §9.7 and are not part of any
deliverable. Numbers tagged **[estimate]** are arithmetic with the working shown.

---

## 0. Repo state, and the four arguments this document turns on

### 0.1 Existing assets

`/home/user/bert-poc` contains four specification documents, `data/raw/tickets_export.csv`
(5,013 rows, 2.4 MB), and its README. **There is no code of any kind** — no pipeline, no
notebooks, no preprocessing module, no `preprocess.json`, no model artifacts, no
`ticket_messages` companion table. Everything below is specified from scratch. Nothing is
being reused because there is nothing to reuse.

### 0.2 Four load-bearing arguments, stated before anything else

Three of these are places where I do not agree with the brief; the fourth is a correction the
user made to their own brief in revision round 2, which I agree with and have argued for below
rather than merely complied with.

**(1) Phase 2 must not rewrite the text the model reads.** The brief's framing ("bounded
rewrite/extraction under a structured-output schema") is right about *bounded* and right about
*structured*, and wrong about *rewrite*. The reason is the constraint that already governs this
project: [classifier spec §4.6 / §5.3](./ticket-services-classifier.md) and
[proposal §5.3](../proposals/ticket-services-classifier.md) require that **the identical
preprocessing runs at training and at inference**. If Phase 2 rewrites the training text, then
avoiding train/serve skew requires running Phase 2 at inference — an LLM call per ticket on the
main prediction path, which the proposal rejected in §4 and which blows the 42.9 ms / 250 ms
p95 CPU budget by two orders of magnitude.

So Phase 2 gets exactly three legitimate shapes: **rule-mining** (its output becomes
deterministic rules that run inside Phase 1, at train and serve, in microseconds),
**row metadata** (facts *about* the row used for filtering/weighting/routing, never altering
the text), and **residual-PII detection** (a second opinion feeding the redaction audit). All
three preserve the user's design — one bounded, schema-constrained call per row, model selects,
code decides — while removing the skew and the per-request LLM cost. §4.2 specifies it this way.

**(2) Do not fuse Phase 2 and Phase 3 into one call.** The brief offers this as an option. It
is foreclosed by the contamination rule, not by cost: **Phase 2 must run on the gold val/test
rows** (their text must be preprocessed identically to training text, or you have skew inside
your own evaluation), and **Phase 3 must never run on the gold rows** (§5 of the LLM fallback
policy). A fused call cannot satisfy both. There is a second, independent reason in §4.3.9.

**(3) Placeholder normalisation is not a token-budget play on this corpus.** Classifier spec
§4.6 estimates that placeholder normalisation "typically recovers 10–30% of the token budget".
**[measured]** on this corpus it recovers **2.1%** overall and **8.3%** on the 922 rows that
contain anything to replace. The justification for placeholders here is compliance and
skew-stability, not tokens. What *does* recover budget is boilerplate removal: 5.5% corpus-wide,
and **65.1%** on the 99 auto-alert rows. Section §4.1.6 has the table. I am not asking anyone to
change §4.6 — its estimate was explicitly flagged as one to measure per corpus. This is the
measurement, and this corpus does not support it.

**(4) Label-error detection is a cheap-first cascade, and the LLM judge is its last and
smallest tier.** Sending every labelled row to an LLM judge — the design in this document's
first revision, ~6,000 calls — is neither the cheapest nor the best-grounded instrument
available, and the measurements are one-sided. A **TF-IDF + one-vs-rest logistic regression**
pass with grouped cross-validation costs **[measured] 113 seconds** on this corpus and produces
out-of-sample probabilities; **cleanlab 2.9.0** confident learning over those probabilities
flags **[measured] 95 of 4,622 rows (2.06%)** as likely mislabelled, and ranks all of them.
Feeding only the residual to the judge takes Phase 3 from ~6,000 LLM calls to **~375–600 calls
on [measured] 288 rows (6.2% of labelled rows)** — a 10–16× reduction — and it moves most of the work
onto instruments that carry **zero contamination risk**, because they never emit an opinion
about a label, only a routing signal. §4.3 is rewritten around this. §4.3.12 gives the condition
under which the LLM tier should be dropped entirely.

---

## 1. Problem statement

### 1.1 What the pipeline is for

**Input:** one raw CSV export of the `tickets` table, columns as in
[`data/raw/README.md`](../../data/raw/README.md).

**Output:** a frozen, content-hashed corpus snapshot in which every row carries

- a canonical **model input text** (the string that both training and inference will see, before
  tokenisation),
- a canonical **label set** drawn from the 20-service taxonomy,
- **provenance** for both the text and the label,
- **routing/eligibility flags**: split assignment eligibility, near-duplicate cluster ID,
  language tag, quality-quarantine reason if any,
- an **audit trail** of every transformation applied, sufficient to reconstruct why any span
  disappeared.

**The decision the pipeline informs:** which rows and which labels are allowed into supervised
training, and which text the encoder is fine-tuned on. It also produces a **human-supervision
queue** of rows whose labels are disputed.

### 1.2 What "good enough" means

Four things, in priority order. The first is a hard constraint, not a metric.

1. **No train/serve skew.** Every transformation that alters the model input text must be
   reproducible at inference from a frozen artifact, with no LLM in the loop and no corpus-wide
   statistic that is unavailable for a single unseen ticket. This is binary: a pipeline that
   fails it is not shippable regardless of its other numbers.
2. **No leakage of personal data into the training corpus, model artifacts, or logs** — names,
   contact details, company identifiers, and bank/transactional details. Measured against seeded
   canaries (§6.4), gated at recall ≥ 0.99 for structured classes.
3. **No destruction of signal.** The pipeline must not strip the spans that carry the label.
   Measured as over-redaction rate on protected technical strings (§6.4), gated at ≤ 0.5%.
4. **Measurable value per phase.** Phase 2 and Phase 3 must each demonstrate a gain on the
   blind gold test set under the ablation ladder in §5.2, or they are switched off. A phase that
   cannot beat Phase-1-only is a finding, and it is cheap to learn in week 2.

### 1.3 What this pipeline explicitly does not do

It does not reconstruct label provenance (impossible from this file — §7.1), does not produce
the gold set, does not choose splits (that is classifier spec §2.6; the pipeline *supplies the
inputs* to the split: near-dup clusters, language tags, timestamps), and does not train
anything.

---

## 2. Data

### 2.1 What is actually in the file [measured]

Everything in this subsection was measured on the CSV, not taken from its README. Where I
disagree with the README I say so.

| Property | Value | Note |
|---|---|---|
| Rows / columns | 5,013 / 10 | header + LF, RFC4180 quoting |
| Rows with non-empty `services` | 4,622 | 391 empty — untriaged or unactionable, indistinguishable here |
| Positive (ticket, service) pairs | 8,771 | mean cardinality 1.90 over labelled rows |
| Cardinality histogram | 1 → 1,569 · 2 → 1,988 · 3 → 1,034 · 4 → 31 | matches the `1 ≤ |S| ≤ 4` contract |
| Distinct raw `services` tokens | 59 | canonicalise to **20**; **0 out-of-taxonomy tokens** |
| `description` length (chars) | min 22 · median 186 · p90 279 · max 793 | |
| Descriptions with embedded newline | 374 | mean 118.6 tokens, p95 222 |
| Distinct `labels` values | 243 | open vocabulary, RU+EN mixed, `;`-separated |
| `flags` | escalated 551 · vip 105 · sla_breach 94 · reopened 54 · duplicate 50 | closed vocabulary |
| Distinct titles, raw | 4,051 | → **3,905** after strip+lower, → 3,893 after also stripping `RE:`/`FW:` and collapsing whitespace. Normalisation alone collapses 158 title variants |
| Distinct `title|description` pairs | 5,013 | no exact full-row duplicates |

### 2.2 Token budget under the production tokenizer [measured]

XLM-R (`FacebookAI/xlm-roberta-base` `tokenizer.json`), input built per classifier spec §4.6 as
`"[priority: {p}] {title}\n{description}"`:

| Variant | mean | p50 | p90 | p95 | p99 | max | >128 tok | total tokens |
|---|---|---|---|---|---|---|---|---|
| Raw | 71.0 | 68 | 96 | 108 | 218 | 228 | 2.3% | 356,085 |
| + Phase-1 placeholders | 69.6 | 67 | 94 | 103 | 195 | 207 | 2.1% | 348,727 (−2.1%) |
| + boilerplate/quote removal | 67.2 | 67 | 91 | 98 | **114** | **142** | **0.1%** | 336,663 (−5.5%) |

Three consequences that belong in the classifier spec's hands:

- **`max_len = 128` is the right setting for this corpus** (spec §4.6 says p90 capped at 256;
  p90 is 96). That is the "sequence length is the biggest lever" saving from §4.5 —
  roughly half the latency of 256 — available for free.
- After Phase 1, **p99 falls from 218 to 114 tokens and the >128 rate from 2.3% to 0.1%**.
  Truncation, and therefore the head-vs-head+tail ablation in §4.6, becomes nearly moot on this
  corpus. On a real corpus with genuine multi-kilobyte log pastes it will not be moot; keep the
  ablation.
- The 99 auto-alert rows go from **217.5 to 76.0 mean tokens (−65.1%)**. Almost the entire
  token-budget win in this pipeline comes from one deterministic rule.

### 2.3 PII and identifier inventory [measured]

Counts are over `title + "\n" + description`. "Rows" = rows containing ≥1 instance.

**Scope note, and it changes what this pipeline is for.** The product owner states that
**production ticket `title`/`description` contain no application secrets, API keys or program
keys**, and this specification takes that as authoritative (see §2.4). There is therefore no
credential-detection tier anywhere in this pipeline. What the pipeline is built to remove is
**personal and commercial data**: names, contact details, company identifiers, and
bank/transactional details.

| Class | Rows | Instances | Detectable by | Verdict (§4.1.4) |
|---|---|---|---|---|
| E-mail address | 262 | 265 (189 distinct) | regex | `<EMAIL>` |
| URL | 210 | 210 (110 with a query string) | regex | `<URL:{host_class}>`, query and fragment dropped |
| Phone `+7…` | 126 | — | regex (6 surface formats present) | `<PHONE>` |
| Phone `+1…` | 53 | — | regex | `<PHONE>` |
| Card-like (`4111 11** **** 1111`) | 21 | — | regex; **partially masked already**, so a naive 13–19-digit-run regex matches **0** | `<CARD>` |
| ИНН / ОГРН / КПП keyword | 143 | — | keyword + value regex | `<INN>` on the value |
| ИНН with a 10/12-digit value | 24 | 24 | regex, checksum-validatable | `<INN>` |
| Legal entity `ООО «…»` | 32 | 33 | regex | `<ORG>` |
| Person name, RU (two capitalised words) | 359 | 370 | NER required | `<PERSON>` |
| Person name, EN (`Firstname Lastname` shape) | 265 | 282 (145 distinct) | NER required | `<PERSON>` — **see the trap below** |
| Money amount (`… ₽`, `… руб`) | 36 | 37 | regex | **preserved** — a billing-signal quantity, not an identifier |
| Bank / account / transaction identifiers (р/с, БИК, СНИЛС, 20-digit account, payment reference) | **0** | 0 | regex, checksum-validatable | `<ACCOUNT>` / `<BIC>` — **untested by this fixture**, see §7.1 |
| паспорт (`45 03 123456`) | **0** | 0 | keyword + regex | `<PASSPORT>` — untested by this fixture |
| IPv4 | 17 | — | regex | `<IP>` |
| ISO timestamp | 100 | — | regex | `<TS>` |
| Host/resource id (`i-55ba20`, `db-01`, `lb-prod`, `api-gw-1`) | 182 | 219 | regex | **preserved** (§4.1.4) |
| base64-ish run ≥32 chars | 1 | 1 | regex | `<B64>` — a token-budget rule, not a security rule |
| UUID | 0 | 0 | regex | `<UUID>`, insurance |

**The trap, and it is the single most important measurement in this document.** Of the 282
English `Firstname Lastname`-shaped matches, **126 (45%) are technical strings, not people**:
`Service Unavailable` (34), `Too Many [Requests]` (25), `Bad Gateway` (10), `Internal Server`
(9), `Gateway Timeout` (5), `Method Not [Allowed]` (5), `Payload Too [Large]` (5),
`Unprocessable Entity` (3), `Seq Scan` (6), plus `Host Loss` / `Snt Last` / `Avg Best` from
pasted `mtr` output. A PERSON detector — regex *or* NER — pointed at raw ticket text will
redact exactly the HTTP reason phrases and PostgreSQL plan nodes that are the strongest label
signal in the corpus. This is why §4.1.3 makes **protected-span masking strictly precede
detection**, and why §6.4 gates on over-redaction precision and not only on recall.

### 2.4 The fixture over-states one risk class, and it is the one we are not building for

Production `title`/`description` carry no application secrets (§2.3), but **this fixture carries
them by construction**: its README documents 108 of 210 URLs bearing a token or key, and
[measured] 81 such values are present. Any secret-detection or canary result computed on this
file therefore describes the generator, not production, and no gate in §6 is keyed on it. The one
rule that survives from this observation is unconditional URL query-string stripping (§4.1.4) —
and it survives on **normalisation** grounds (high-entropy, zero-signal, eats token budget), not
security grounds.

### 2.5 Duplication and burst structure [measured]

MinHash (128 permutations, char-5-gram shingles, `datasketch` MinHashLSH), union-find over
candidate pairs:

| Signature source | Jaccard | Clusters > 1 | Rows in clusters | Largest | Rows dropped if collapsed to one |
|---|---|---|---|---|---|
| Raw text | 0.80 | 33 | 174 | **98** | 141 (2.8%) |
| Raw text | 0.70 | 126 | 388 | 99 | 262 (5.2%) |
| Raw text | 0.60 | 346 | 916 | 99 | 570 (11.4%) |
| Raw text | 0.50 | 791 | 2,522 | 99 | 1,731 (34.5%) |
| **Phase-1-cleaned text** | 0.80 | 55 | 150 | **8** | 95 (1.9%) |
| Phase-1-cleaned text | 0.70 | 137 | 400 | 71 | 263 (5.2%) |

**The ordering finding.** The 98-row auto-alert clique is visible at Jaccard 0.80 *only before*
boilerplate stripping; after Phase 1 removes the shared footer, the largest cluster is 8 rows.
Conversely, cleaning surfaces 22 additional clusters that the shared template had masked. These
are two different duplicate species — **template duplicates** and **content duplicates** — and
neither signature finds both. §4.1.7 therefore computes **two signatures per row** and unions
the clusters. Cost is one extra MinHash pass: **[measured]** 10.7 s per 5,013 rows single-core
(2.1 ms/row), so ~2 min per pass at 50k rows.

**Group-splitting by `organization_id` is infeasible on this corpus.** [measured] Under a
temporal 70/15/15 split, **206 of 210 organisations span more than one split, covering 4,990 of
5,013 rows (99.5%)**. Runbook §4.3 and classifier spec §2.6 both say "group by
`organization_id` when splitting". Taken literally that would force essentially the whole corpus
into a single split. The workable version, and what this pipeline supports, is narrower: enforce
grouping **only within near-duplicate clusters** (§4.1.7 emits `dup_cluster_id`; the splitter
must keep a cluster whole or drop the later members), and rely on the ≥1-week temporal gap for
burst correlation. [measured] at Jaccard 0.80 only **9 clusters / 120 rows** straddle a temporal
boundary — that is the actual leakage surface, and it is small and cheap to fix.

Burst days [measured], for the ≥1-week gap rule: 2025-11-18 (110 tickets), 2026-03-05 (99),
2026-06-23 (76), 2025-09-09 (58), 2026-04-14 (42), 2026-01-21 (42), 2026-05-27 (38),
2025-12-03 (28).

#### 2.5.1 Label conflicts inside duplicate groups [measured]

The same clusters answer a second question that Phase 3 tier 1 (§4.3.2) depends on: **do
duplicate texts carry the same labels?** Grading rule as defined in §4.3.2 (`identical` /
`nested` / `partial` / `disjoint`), restricted to groups with ≥2 *labelled* rows:

| Grouping | Groups | Rows | identical | nested | partial | disjoint |
|---|---|---|---|---|---|---|
| Exact `title`+`description` | 0 | 0 | — | — | — | — |
| Exact `description` | 6 | 12 | 3 | — | — | 3 groups / 6 rows differ, ungraded |
| Exact `title` (normalised) | **565** | **1,513** | 564 | **1** (2 rows) | 0 | 0 |
| Near-dup, Jaccard 0.80 | 29 | 164 | 27 | 1 (2 rows) | **1 (98 rows)** | 0 |
| Near-dup, Jaccard 0.60 | 316 | 835 | 313 | 1 (2 rows) | 1 (99 rows) | **1 (3 rows)** |

Three readings, and the third is the one that changes the design:

- **The fixture is internally consistent, so tier 1 finds almost nothing here.** 564 of 565
  same-title groups carry byte-identical label sets. That is a property of a generator that
  assigns labels from a scenario, not a property of real annotation. On a real corpus with a
  CatBoost-contaminated `services` column this tier is where the cheapest wins live; here it
  yields **~14 reviewable rows.** Report that honestly and re-run it on real data before drawing
  any conclusion about tier 1's yield.
- **Strict set-inequality would be badly wrong.** The single `partial` group at Jaccard 0.80 is
  the **98-row auto-alert clique**, whose members differ legitimately: `[managed-postgres,
  monitoring]` for a connection-error alert versus `[compute, monitoring]` for a CPU alert. A
  naive "duplicate texts with different labels ⇒ conflict" rule would dump 98 rows into a human
  queue for behaving correctly. §4.3.2's graded rule plus a cluster-size guard routes 5.
- **The one genuine `disjoint` conflict** is a 3-row group titled `METRICS VIA API`, labelled
  `[api-gateway, monitoring]` on one row and `[console-ui]` on two others. That is exactly the
  `console-ui`-versus-the-broken-component ambiguity from §2.8, found with no model at all — which
  is the argument for putting this tier first.

### 2.6 Boilerplate [measured]

Splitting descriptions into sentences (terminal punctuation or newline, ≥10 chars): **20,630
sentence instances over 5,681 distinct types.**

| Frequency cut | Types | Instances | Share of all instances | Types matching a technical pattern | Technical instances (collateral) |
|---|---|---|---|---|---|
| ≥ 20× | 149 | 9,935 | **48.2%** | 3 | 122 (**1.2%** of the stripped mass) |
| ≥ 10× | 200 | 10,581 | 51.3% | 13 | 251 (2.4%) |
| ≥ 5× | 571 | 12,729 | 61.7% | 42 | 448 (3.5%) |

The three technical types caught at the ≥20× cut are `Severity: warning` (55),
`Severity: critical` (44), `429 Too Many Requests` (23) — and all three are recoverable with an
evidence-pattern allowlist.

**This measurement is the single strongest argument in the document, and it argues against
Phase 2.** The thing an LLM cleaner would most obviously buy — removing 48% of the corpus that
is generic politeness filler — is achievable **deterministically, at 0.09 ms/row, with 1.2%
collateral damage, from a 149-line blocklist that one person can eyeball in an hour.** Fund
Phase 2 only after this is in place and the ablation (§5.2) shows a residual gain.

### 2.7 Language [measured]

| Method | Result |
|---|---|
| `py3langid` on full text | ru 3,577 · en 1,429 · bg 5 · fr 1 · sr 1 |
| README's authored ground truth | RU 3,347 · EN 1,429 · code-switched 237 |
| Cyrillic-letter ratio, <0.01 / 0.50–0.85 / ≥0.85 | 1,430 / 322 / 3,260 |
| RU-dominant (cyr ≥ 0.85) with ≥5% / ≥10% Latin letters | 1,071 / 306 |
| RU-dominant containing an English error-string pattern | 82 |

Two findings:

- **The RU/EN binary is easy.** `py3langid` misroutes 7 of 5,013 rows (0.14%) into `bg`/`fr`/`sr`
  and gets RU+code-switched vs EN otherwise exactly right. Any of the three candidate LID
  libraries will do for `lang_primary`.
- **"Mixed" is not definable by a ratio, and the fixture proves it.** Its README reports that a
  Cyrillic-ratio heuristic recovers only 9 of the 237 authored code-switched rows; my
  letters-only variant with a 15% minority band flags 322 rows, a different set again. Neither
  reconciles with the ground truth, because *ratio* and *code-switching* are different
  properties. §4.1.8 replaces the ratio rule with an operational definition and a span-level
  detector.

### 2.8 Label hygiene the fixture deliberately contains [measured]

39 rows with a space after the comma · 38 rows with mixed case (`Billing`, `Console-ui`) · 40
rows with an internal duplicate (`logging,logging`) · 59 distinct raw tokens collapsing to 20
canonical · **0 tokens outside the taxonomy**. 207 rows have leading/trailing whitespace on
title or description · 120 ALL-CAPS titles · 337 lowercase-initial titles · 53 titles prefixed
`RE:`/`FW:`.

Ambiguous-pair prevalence, which sizes Phase 3:

| Pair | Co-occur | Either | Co-occur / either |
|---|---|---|---|
| `access-control` & `auth` | 213 | 1,216 | **17.5%** |
| `integrations` & `notifications` | 73 | 690 | 10.6% |
| `logging` & `monitoring` | 37 | 877 | 4.2% |
| `cdn` & `networking` | 5 | 485 | 1.0% |
| `console-ui` total / alone / co-labelled | 793 / 156 / **637** | | |
| Rows carrying a rare-tail service (`terraform-provider`, `message-queue`, `cdn`, `dns`, `managed-redis`) | **212** | | |
| Rows touching **any** ambiguous-family service | **2,730 (59.1% of labelled rows)** | | |

That last row is why "route ambiguous cases to a human" is not a routing rule. 59% of the corpus
is not a queue, it is the corpus. §4.3.5 sharpens "ambiguous" into a *measurable* condition using
the tier-2 out-of-sample probabilities — carries one member of a pair while the model prefers the
other — which [measured] selects **6 rows** for the four pairs and **65 rows** for `console-ui`,
not 2,730.

### 2.9 Licensing and data governance

The data is first-party; there is no dataset licence question.

The binding constraint is **data governance**: ticket text is personal data (§2.3), so **moving
it outside the production perimeter is a decision that requires approval, not a default.**
Whether pseudonymised text may leave the perimeter — to a hosted model provider, to a training
environment, to anywhere — is owned by legal and the DPO (open question P1), and it is not an
engineering call. **This document does not make it.**

What this document does instead is make either answer workable. The pipeline is specified so
that the deterministic phases run anywhere, the LLM phases default to a provider inside the
perimeter (§4.2.6), and the hosted option is a documented, measurable upgrade rather than an
assumption baked into the design. Nothing has to be re-architected when the answer arrives.

---

## 3. Baselines

Runbook §0's governing principle applies to pipelines too: **every stage must beat doing
nothing, and the cheap comparison comes first.** The ablation ladder in §5.2 is evaluated
against these.

| # | Baseline | What it is | Why it must exist |
|---|---|---|---|
| **B0** | **Raw passthrough** | `title + "\n" + description`, no processing at all beyond CSV parsing and `services` canonicalisation | The floor. If B1 does not beat this on the gold test set, the entire pipeline is ceremony. Cheap to run |
| **B1** | **Phase 1 only** | §4.1 deterministic pipeline, no LLM anywhere | **The real baseline.** This is the trivial-baseline slot from classifier spec §3, transposed to the data layer. Phase 2 and Phase 3 must each beat *this*, not B0 |
| **B2** | **Phase 1 + Phase 3 cheap tiers** (conflicts, tier-2 probabilities, cleanlab), no LLM judge | labels repaired by the zero-contamination tiers only | **New in revision 2, and the important one.** If B2 captures most of the available gain, the LLM tier is not worth its contamination risk (§4.3.12) |
| **B2b** | B2 + LLM judge | full Phase 3 | Isolates the judge's marginal contribution above the cheap tiers |
| **B3** | **Phase 1 + Phase 2, no Phase 3** | text improved via mined rules, labels untouched | Isolates text cleaning from label repair |
| **B4** | Full pipeline | | Must beat max(B2b, B3) or the extra phase is dropped |

**The TF-IDF baseline is no longer discarded after it is measured.** Classifier spec §3 item 3
already requires "TF-IDF (word 1–2 grams + char 3–5 grams) + one-vs-rest logistic regression" as
the real baseline to beat. In the first revision of this document that model was computed for the
classifier comparison and thrown away. It is now **a pipeline instrument** (§4.3.3): its
out-of-sample probabilities feed confident learning, its ambiguous-pair probabilities define the
judge's routing stratum, and its metric delta across preprocessing variants is a **GPU-free proxy
for the §5.2 ablation ladder**. One 113-second CPU job [measured] serves three purposes.

**Also required, and cheaper than all of the above: a no-model check.** Report per-phase row
counts, token totals, redaction counts, and dropped-row counts as a diff table for every phase.
A phase that changes < 1% of rows does not need an ablation to be questioned; it needs a
justification for existing.

---

## 4. Approach

### 4.0 Two tiers, and the invariant that separates them

Every operation in this pipeline belongs to exactly one tier. **This classification is the
central design device of the document** and everything else follows from it.

| Tier | Definition | Constraint | Runs at inference? |
|---|---|---|---|
| **A — text-altering** | changes the string the model tokenises | must be deterministic, frozen in a versioned artifact, and executable on a single unseen ticket with no corpus statistics and no network call | **Yes, mandatory** |
| **B — corpus-shaping** | selects, weights, tags, groups or drops rows; never alters the text | may use corpus-wide statistics, LLMs, and human input | **No, forbidden** |

**The invariant:** anything in Tier A that cannot be executed at inference in `< 5 ms` on CPU
with no external dependency is not allowed in Tier A, and therefore is not allowed to alter
text at all. It may only produce Tier-B metadata.

[measured] the full Phase-1 Tier-A regex pass costs **0.09 ms/row** single-core CPython 3.11 on
this corpus — negligible against the 42.9 ms encoder call. The only Tier-A candidate with a
real cost is NER-based name redaction; §4.1.5 resolves it.

Tier assignments at a glance:

| Operation | Tier |
|---|---|
| Unicode/NFC, whitespace, quotes, `ё`, mojibake, CSV artefacts | A |
| Quote/signature stripping, boilerplate blocklist, template footers | A |
| Regex placeholder substitution (email, phone, URL, card, ИНН, bank/transaction details, IP, TS, IDs) | A |
| Log/traceback skeleton reduction | A |
| Input template assembly, truncation | A |
| NER-based `<PERSON>` / `<ORG>` redaction | A **only if** the budget in §4.1.5 is met; otherwise B |
| **Phase 0 label profiling (§4.0.1)** | **B** — reads labels, changes nothing |
| `services` / `labels` / `flags` canonicalisation | B (labels, not text) |
| Language identification and code-switch tagging | B |
| Near-duplicate clustering | B |
| Length/emptiness/quality filters and quarantine | B |
| **Phase 2 (LLM), all of it** | **B** |
| **Phase 3 (judge + human queue), all of it** | **B** |

---

### 4.0.1 Phase 0 — ingest and profile

**Placement, and why it is numbered 0.** This runs immediately after CSV ingest and **before
Phase 1 touches a single character**. It is numbered 0 rather than inserted as a new Phase 1 so
that the existing phase numbers — referenced from the architecture document, the proposal, the
runbook and the classifier spec — stay valid. The architect owns the stage mechanics; this
section owns what is computed and what it means.

**Why it cannot wait for §6.1.** §6.1 already reports class balance, and classifier spec §2.8
already requires per-service positive counts — but **both land on the assembled corpus, at the
end of the build.** The question Phase 0 answers is about the *input*, and the distinction
decides money: **if a service has 15 positives in the raw export, no amount of cleaning, judging
or reviewing creates more.** The correct response to that finding is a scoping conversation with
the product owner, not a pipeline run. It costs seconds of pandas and it must arrive before
anyone commits ~$25 of LLM calls and ~28 person-hours (§4.3.11). Phase 0 exists because the
cheapest possible answer to "is this dataset viable?" was being computed last.

#### 4.0.1.1 What is computed

All of it from `services`, `created_at` and the text — no model, no cleaning, no dependencies
beyond the dataframe library. Every figure carries `n`.

| # | Statistic | Purpose |
|---|---|---|
| 1 | **Per-service positive count, prevalence (% of labelled rows), and share of all positives** | The headline table. Prevalence and share differ because rows are multi-label; report both |
| 2 | **Imbalance ratio** head:tail, and **mean / median** positives per service | One number a stakeholder can hold |
| 3 | **Distributional summary**: normalised Shannon entropy and Gini over the label marginal | Ratio alone is driven by two services; these describe the whole shape and are diffable |
| 4 | **Cardinality histogram** `|S| ∈ {0,1,2,3,4}` and any `|S| > 4` violations | Validates the `1 ≤ |S| ≤ 4` output contract against reality |
| 5 | **Empty-`services` rate** | The untriaged/unactionable population (§4.1.10); it is not an error but it changes every denominator |
| 6 | **Co-occurrence matrix** | Already specified as a §6.1 output — Phase 0 computes it first and §6.1 references it. **Do not duplicate the definition** |
| 7 | **Per-service counts by temporal window** (month and quarter), with min/max ratio | A service that appears in one quarter and vanishes is a taxonomy change nobody recorded, or a product that shipped |
| 8 | **Per-service counts by language** | Classifier spec §6.3 makes RU/EN a first-class slice with a hard ship gate; a service that is 95% one language cannot pass a per-language gate |
| 9 | **Projected positives per split**, under the actual temporal 70/15/15 split (classifier spec §2.6), **and** under the planned 2,000-row gold test draw | **The single most decision-relevant number in the report**, and it is computed nowhere else |

#### 4.0.1.2 The fixture's numbers [measured]

Reported here in the form the artifact should take, so the implementation has a target.

**Headline:** 5,013 rows · **4,622 labelled (92.2%)** · 391 empty (7.80%) · 8,771 positives ·
20 services · mean cardinality **1.90** · imbalance **56.7 : 1** (`billing` 851 →
`terraform-provider` 15) · normalised entropy **0.915** · Gini **0.361** · mean 438.6 and median
447 positives per service.

| Service | n | Prevalence | Share of positives | train | val | **test** | EN share |
|---|---|---|---|---|---|---|---|
| `billing` | 851 | 18.41% | 9.70% | 615 | 129 | 107 | 26.5% |
| `console-ui` | 793 | 17.16% | 9.04% | 591 | 106 | 96 | 28.2% |
| `auth` | 762 | 16.49% | 8.69% | 573 | 92 | 97 | 27.4% |
| `api-gateway` | 751 | 16.25% | 8.56% | 567 | 88 | 96 | 29.7% |
| `compute` | 715 | 15.47% | 8.15% | 458 | 107 | 150 | 29.5% |
| `access-control` | 667 | 14.43% | 7.60% | 468 | 106 | 93 | 26.2% |
| `object-storage` | 645 | 13.95% | 7.35% | 470 | 96 | 79 | 28.1% |
| `monitoring` | 590 | 12.77% | 6.73% | 419 | 90 | 81 | 38.8% |
| `subscriptions` | 560 | 12.12% | 6.38% | 406 | 79 | 75 | 28.6% |
| `networking` | 452 | 9.78% | 5.15% | 264 | 71 | 117 | 32.7% |
| `notifications` | 442 | 9.56% | 5.04% | 305 | 84 | 53 | 26.9% |
| `managed-postgres` | 404 | 8.74% | 4.61% | 244 | 84 | 76 | 34.2% |
| `logging` | 324 | 7.01% | 3.69% | 227 | 55 | 42 | 23.8% |
| `integrations` | 321 | 6.95% | 3.66% | 234 | 46 | 41 | 30.2% |
| `backups` | 271 | 5.86% | 3.09% | 183 | 38 | 50 | 31.7% |
| `managed-redis` | 98 | 2.12% | 1.12% | 67 | 13 | **18** | 25.5% |
| `dns` | 41 | 0.89% | 0.47% | 31 | 5 | **5** | 29.3% |
| `cdn` | 38 | 0.82% | 0.43% | 26 | 7 | **5** | 23.7% |
| `message-queue` | 31 | 0.67% | 0.35% | 25 | 5 | **1** | 22.6% |
| `terraform-provider` | 15 | 0.32% | 0.17% | 13 | 1 | **1** | 20.0% |

Cardinality: `|S|=0` 391 (7.80%) · 1 → 1,569 (31.30%) · 2 → 1,988 (39.66%) · 3 → 1,034 (20.63%) ·
4 → 31 (0.62%) · **>4 → 0**.

**Read the bolded test column, because it is the finding.** `terraform-provider` and
`message-queue` project to **one test positive each**; `dns` and `cdn` to five. A precision
estimate on one positive is not an estimate — [computed] the Clopper–Pearson 95% interval around
a true precision of 0.90 is **±48.8pp at n=1, ±35.6pp at n=5, ±22.1pp at n=10, ±16.7pp at n=18,
±12.2pp at n=30 and ±9.2pp at n=50.** That is where classifier spec §2.8's "50 positives" line
comes from, and Phase 0 is what makes it actionable per service instead of corpus-wide.

**And the arithmetic that closes the door**, because it is the argument the product owner needs:
to obtain 50 test positives at the fixture's prevalence you would need a test set of
[computed] **15,625 tickets for `terraform-provider`**, 7,463 for `message-queue`, 6,098 for
`cdn`, 5,618 for `dns` and 2,358 for `managed-redis` — against a gold test budget of **2,000**
(runbook §4.2). Only `managed-redis` is within a factor of ~1.2 of reach. **You cannot annotate
your way to an evaluable rare tail at this prevalence**; the options are a longer collection
window, targeted enrichment sampling for those services alone (which then needs its own
reweighting, and cannot be the main test set — runbook §0), or accepting that these services are
reported without a number.

Temporal instability in the tail is real and worth its own line [measured, positives per quarter,
first and last quarters partial]: `cdn` ranges 1→14 across quarters (14×), `message-queue`
2→18 with **a quarter at zero**, `terraform-provider` 5→0→1 with **a quarter at zero**. Head
services move by 4.5–5.8×. A tail service with a zero quarter cannot support the monthly drift
slice classifier spec §6.3 requires.

#### 4.0.1.3 Tiers, and what each one triggers

Classifier spec §2.8 draws one line at 50 positives. Phase 0 turns it into four tiers with
consequences attached. Thresholds are on **human-labelled positives in the training pool**, with
the projected test count as the second, independent test.

| Tier | Condition | Consequence |
|---|---|---|
| **T-0 — absent** | **0 positives** in the export while present in `labels.json` | **Hard stop, G19.** The service is in the label space and unlearnable. Either the export is incomplete or the taxonomy contains something the product does not use. Not a modelling problem |
| **T-1 — unevaluable** | **< 30 projected test positives** | Ship the head row in the label space, but **report "insufficient data" instead of a precision figure** (classifier spec §4.8 already requires this) and **exclude from macro aggregates**, reporting macro both ways. At ±16.7pp (n=18) the number would mislead more than it informs. [measured] this is `managed-redis`, `dns`, `cdn`, `message-queue`, `terraform-provider` — 5 of 20 services |
| **T-2 — unlearnable** | **< 50 training positives** | The threshold-tuning path is not available (§4.0.1.4). Serve via the keyword rule or kNN-over-embeddings mechanism in classifier spec §4.8, or let the model abstain. **Do not let it silently take a τ_s of 1.0 without the report saying so.** [measured] `terraform-provider` (13), `message-queue` (25), `cdn` (26) |
| **T-3 — merge candidate** | T-2 **and** the §6.1 pair-confusion table shows a co-occurring or confusable partner | Escalate to the product owner as a **taxonomy question**, not a modelling one. [measured] `cdn` co-occurs with `object-storage` 16× and `dns` 11× of its 38 rows; `dns` with `networking` 16×. Whether `cdn` should exist separately from `object-storage`/`networking` at this volume is a product decision and Phase 0's job is to put it in front of someone |

**A service may be in more than one tier**; report the highest that applies. **No tier action is
automatic** — every one of them is a recommendation to a human, in keeping with §4.3.1's rule
that this pipeline routes and does not decide.

#### 4.0.1.4 What imbalance actually implies for training

Most imbalance folklore is written for multi-class problems and **transfers badly to multi-label
ones**. Four consequences, argued rather than listed, because the wrong reflex here is expensive.

**1. Resampling is largely unavailable, and this is the first thing someone will reach for.**
In multi-class, oversampling a rare class touches only that class. Here every row carries a
*set*, so **duplicating a row to raise one service's prevalence raises the prevalence of every
other service on that row.** [measured] the fixture makes this vivid: `cdn`'s 38 rows have mean
cardinality 2.16, so oversampling `cdn` 10× adds 342 `cdn` positives and **396 positives to
other services** — more collateral than target — of which 11× go to `dns`, itself a rare service
whose prevalence you were not trying to change. `terraform-provider` at 10× adds 135 target and
117 collateral, mostly to `managed-postgres` and `object-storage`. Undersampling the head is
worse: to thin `billing` you delete rows that carry `subscriptions`, `console-ui` and everything
else they co-occur with. **Recommendation: do not resample. Use per-service loss weighting
instead** — it applies at the level of the (row, service) cell, which is the granularity the
problem actually has. Multi-label resampling methods that operate on label *sets* exist, but they
change the joint label distribution the model is trying to learn, and this corpus has structured
co-occurrence (§2.8) worth preserving.

**2. Per-service thresholds are the main imbalance lever, and they are already the main precision
lever — but the circularity has to be named.** Classifier spec §4.6.1 and §6.1 already tune τ_s
per service to a precision floor. That machinery *is* the imbalance response: a rare service does
not need a rebalanced training set so much as a threshold placed on enough validation positives
to be placeable. **The problem is that the services that most need a tuned threshold are exactly
the ones without enough validation positives to tune one.** [measured] `terraform-provider`
projects **1** validation positive, `dns`, `cdn` and `message-queue` **5** each. A threshold swept
on 5 positives is fitted to noise and will not hold on test — which is classifier spec §7 risk 13
exactly. The mitigations, in order: **pooled or shrunk thresholds** for services below ~50
validation positives (runbook §4.2 already proposes this, and Phase 0 is what identifies *which*
services need it); a single tier-level threshold shared across all T-1 services; or τ_s = 1.0
with the abstention recorded (classifier spec §6.1 step 3). **Phase 0's deliverable here is the
list of services that must not get an individually-tuned threshold**, produced before threshold
tuning runs rather than discovered during it.

**3. Loss: recommend class-weighted BCE, switch on a measured condition.** Three candidates, and
the project already carries the citation for the third:

| Option | When it is right | Verdict |
|---|---|---|
| **BCE with per-service `pos_weight`** | The default. Directly counteracts the positive/negative asymmetry per output, costs nothing, one hyperparameter per service already in classifier spec §5.2's grid (`{1, sqrt(neg/pos), neg/pos}`) | **Recommended starting point.** At `terraform-provider`'s ratio the full `neg/pos` weight is ~307:1, which will destabilise training — **cap the weight** (start at 20–50) and search the cap |
| **Focal loss** | Many easy negatives dominating the gradient | Not recommended first. It down-weights easy negatives, which is the same job `pos_weight` does more directly and more interpretably here, and it adds γ to the search |
| **Asymmetric Loss** ([Ridnik et al., ICCV 2021](https://arxiv.org/abs/2009.14119)) | Designed precisely for multi-label with few positives per sample — here 1–4 of 20 — and it hard-thresholds very-easy negatives and discards probable mislabels | **The challenger, and it should be run.** Classifier spec §5.1 already specifies it as an ablation with `γ⁻ ∈ {2,3,4}`, `γ⁺ = 0`, `m ∈ {0.05, 0.1}`. Its mislabel tolerance is a bonus given §4.3.3's finding that flagged rows are hard rather than wrong |

**Switch condition, stated so it is decidable:** run weighted BCE and ASL on the same seeds and
splits. **Choose ASL if it improves validation macro-AP by ≥ 1.0pp with non-overlapping seed
spreads; otherwise keep BCE**, because BCE has one fewer moving part and its `pos_weight` is
directly interpretable from the Phase-0 table. Do not choose on principle — classifier spec §5.1
already says this and it is worth repeating where the imbalance numbers live.

**4. Macro versus micro is an imbalance consequence, not a separate topic.** The causal chain is
short and should be stated once, here, where the numbers are: micro metrics pool every
(row, service) decision, so they are dominated by whichever services have the most positives;
[measured] the head 5 services carry **44.1%** of all positives and the head 9 carry **72.2%**.
The four services under the 50-positive line carry **125 positives of 8,771 — 1.43%.** So a model
that **never predicts any of them at all** still reaches a micro-recall ceiling of **98.57%**,
while its macro-recall ceiling is **80.0%**. That 18.6pp gap is the entire imbalance story in one
comparison, and it is why classifier spec §6.2 makes macro-AP the model-selection metric and §7
risk 7 names the degenerate head-only solution. **Phase 0's contribution is to compute that gap
before training, so the size of the blind spot is known rather than discovered.**

#### 4.0.1.5 What imbalance is *not*

**A 57:1 ratio is not automatically a defect to be corrected.** It may faithfully describe what
customers write about: a platform whose users file 851 billing tickets and 15
Terraform-provider tickets has a real, correctly-measured usage distribution, and forcing the
training distribution to be uniform would make the model *worse* calibrated for production, where
the head is genuinely what arrives. Classifier spec §1.2 is precision-first precisely because the
product tolerates a missing suggestion better than a wrong one, and abstaining on a service you
see twenty times a year is a defensible product behaviour.

**The failure mode is not imbalance. It is *unmeasured* imbalance** — a model that silently
never predicts the tail while micro-F1 reads 0.93 and nobody notices, because no per-service
table with `n` beside it was ever produced. Everything in this section exists to make the
distribution visible and to attach a decision to each tier, not to flatten it.

Two corollaries worth stating so the report is not misread:

- **Do not "fix" the ratio.** Fix the *evaluation* (report macro, report per-service `n`, report
  "insufficient data" honestly) and fix the *scope* (merge or descope services the corpus cannot
  support). Those are the two available levers and neither is a resampler.
- **A prevalence that moves between exports is a finding regardless of direction.** A service
  whose prevalence changes 3× between snapshots is either a product change, a taxonomy change, or
  a broken export. That is why §4.0.1.6 requires the diff.

#### 4.0.1.6 Presentation — the statistics must be *shown*

The report is a human-facing artifact, not a JSON blob nobody opens. Requirements:

- **`label_profile.md`** — rendered, committed with the snapshot, and **linked from the snapshot
  manifest**. `label_profile.json` carries the same content machine-readably for the gates.
- **One-line summary first, before any table.** Template:
  > *"4,622 of 5,013 rows labelled (92.2%); 8,771 positives over 20 services; imbalance 56.7:1;
  > **5 of 20 services project fewer than 30 test positives and cannot be given a precision
  > number**; 0 services absent."*

  The bolded clause is the sentence the reader is meant to act on. If nothing else is read, that
  is the finding.
- **Table 1 — per service, sorted by count descending**, columns exactly as §4.0.1.2, with the
  tier (T-0…T-3) in the final column and the sub-50 test cells visually marked. Descending order,
  not alphabetical: the reader's eye should fall off the bottom of the table into the tail, which
  is where the decisions are.
- **Table 2 — cardinality histogram** including `|S| = 0` and any `> 4` violations.
- **Table 3 — per-service × quarter**, with min, max and the max/min ratio, so a zero window is
  visible as a zero.
- **Table 4 — per-service × language**, with the EN share, flagging any service whose minority
  language falls below the count needed for the classifier spec §6.6 per-language ship gate.
- **Table 5 — co-occurrence**, by reference to the §6.1 matrix, not recomputed.
- **A diff against the previous snapshot** on every table, in the same style as the §6.1 taxonomy
  reports: per-service count delta, prevalence delta, and tier transitions. **A tier transition is
  the headline of any subsequent run** — a service crossing from T-1 into evaluability is good
  news that should be reported as loudly as a service falling out of it.
- **No plot is required.** Twenty rows sorted descending is more legible than a bar chart and it
  diffs; §4.3.4 is the standing precedent for preferring a diffable table to a picture.

---

### 4.1 Phase 1 — deterministic, no LLM

Ordering is not cosmetic. Several steps below are correct only in this order, and the ones that
are order-sensitive say so.

#### 4.1.1 Order of operations

```
 1. Parse CSV strictly            → structural quarantine
 2. Fix encoding (ftfy)           → Tier A
 3. Unicode normalise (NFC)       → Tier A
 4. Character folding             → Tier A   (quotes, dashes, ё, invisibles)
 5. Whitespace canonicalisation   → Tier A
 6. MARK PROTECTED SPANS          → Tier A   ← must precede 7, 8, 9, 11
 7. Strip email quotes/signatures → Tier A
 8. Strip boilerplate             → Tier A   (frozen blocklist + template footers)
 9. Placeholder substitution      → Tier A   (regex inventory, §4.1.4)
10. Log/traceback reduction       → Tier A
11. NER redaction (conditional)   → Tier A or B, §4.1.5
12. Assemble input template       → Tier A
13. Canonicalise services/labels  → Tier B
14. Near-duplicate clustering     → Tier B   ← two signatures, §4.1.7
15. Language identification       → Tier B   ← after 6, must see masked prose, §4.1.8
16. Quality filters and quarantine→ Tier B
17. Emit snapshot + audit records
```

#### 4.1.2 Steps 1–5: encoding, Unicode, folding, whitespace

**Parse.** Strict RFC4180, UTF-8 with `errors='strict'`. A row that fails to parse, has the
wrong field count, has an unparseable `created_at`, or has `updated_at < created_at` goes to
**structural quarantine** and is counted, never silently dropped (runbook §1.2's principle).
[measured] this fixture has 246 rows with escaped `""`, 341 with `;`, ~4,100 with `,` inside
prose, and 374 with embedded newlines — all legal CSV, all a good reason not to hand-roll a
splitter.

**Encoding.** [`ftfy`](https://github.com/rspeer/python-ftfy) (Apache-2.0) with default
settings. [measured] this fixture contains **0 replacement characters and 0 mojibake
sequences**, so ftfy is insurance rather than a fix here — but a real `COPY … TO STDOUT` export
that passed through a mis-configured client will have them, and ftfy is explicitly conservative
about not "fixing" already-correct text. Record the count of rows ftfy modified; a nonzero count
on a real export is a finding to report to whoever owns the export.

**Unicode.** `unicodedata.normalize('NFC', s)`. NFC and not NFD, because the tokenizer's
SentencePiece model was trained on NFC-ish web text and because `й` decomposed into `и` + U+0306
tokenises differently.

> **`unidecode` is ruled out and this is not a close call.** It transliterates Cyrillic to Latin
> (`биллинг` → `billing`). On a corpus that is 67% Russian, applying it would destroy the corpus.
> It is named in the brief's candidate list; it has no role here. `unicodedata` alone covers what
> is needed.

**Character folding**, applied outside protected spans only:

| Fold | Rule | Measured on this corpus |
|---|---|---|
| Quotes | `« » “ ” „ ‘ ’` → `"` / `'` | 195 rows affected |
| Dashes | `— –` → `-` | 1,414 rows |
| Invisibles | strip U+200B–U+200F, U+FEFF, U+00AD | 0 rows here; keep the rule |
| NBSP | U+00A0 → space | 0 rows here; keep the rule |
| **`ё` → `е`** | fold, do not expand | 1,700 rows contain `ё` |

**On `ё`, with the measurement.** 178 distinct word types contain `ё`; **19 of them (10.7%) also
occur in the corpus in their `е` spelling**, and the fixture deliberately injects `ё`→`е` typos
on ~15% of rows. [measured] XLM-R tokenises `ещё`/`еще` and `счёт`/`счет` as **one token each**,
so folding costs nothing at the token level and buys the collapse of a purely orthographic
distinction. Fold `ё`→`е`. Do **not** attempt `е`→`ё` restoration (that needs morphology,
introduces errors, and buys nothing). The fold matters more for the TF-IDF and keyword baselines
(classifier spec §3 items 2–3) than for the encoder, and one preprocessing path must serve both.

**Whitespace.** Collapse runs of space/tab to one; normalise CRLF and CR to LF; collapse ≥2
consecutive newlines to one; strip leading/trailing whitespace on both fields (207 rows).
**Preserve single newlines** — they are the boundary between prose and pasted evidence and the
log-reduction step in §4.1.9 needs them.

**Do not case-fold, do not strip punctuation.** 120 ALL-CAPS titles carry a genuine urgency
signal that correlates with priority; subword models handle case; and destroying case would also
destroy `SignatureDoesNotMatch` and `Seq Scan`. Titles are *not* sentence-cased either — 337
lowercase-initial titles are a real stylistic signal.

#### 4.1.3 Step 6: protected spans — the step that makes the rest safe

Before any detector runs, mark and mask spans that must never be edited by a later step. This is
the direct answer to the §2.3 measurement that 45% of person-name-shaped matches are HTTP reason
phrases.

Protected classes, matched in this order, each replaced by an opaque sentinel that is restored
verbatim at step 12:

| Class | Pattern sketch | Why protected |
|---|---|---|
| HTTP status + reason phrase | `\b[45]\d{2}\b(\s+[A-Z][A-Za-z]+){0,3}` | 434 rows; the single most reliable label cue |
| Error/exception class names | `\b[A-Z][A-Za-z0-9]*(Error|Exception|Timeout|Denied|Refused|Match|Failure|Violation)\b`, plus `[A-Z][a-z]+DoesNot[A-Z]\w+` | `SignatureDoesNotMatch` → object-storage; `RequestTimeout`; `AccessDenied` |
| Lowercase error idioms | `SQLSTATE \w{5}`, `OOM command not allowed`, `deadlock detected`, `connection refused`, `invalid cidr`, `Seq Scan`, `too many connections` | RU tickets embed English error text verbatim |
| Alert field names | `^(Alert|Severity|Resource|Threshold|Rule|Current value|Fired at):` | keeps `Severity: critical` out of the boilerplate blocklist's reach |
| Quantities with units | `\d+(\.\d+)?\s?(ТБ|ГБ|МБ|КБ|TB|GB|MB|KB|rps|req/s|ms|%)` | 100 rows; weak but real |
| Resource identifiers | `i-[0-9a-f]{6}`, `db-\d+`, `lb-[a-z]+`, `api-gw-\d+`, `rule_\d+`, `req_[0-9a-f]+` | 182 rows; see §4.1.4 for why these are placeholdered, not deleted |
| Service names in text | the 20 taxonomy names and their RU synonyms | direct label evidence |

The protected-span list is a **versioned artifact** shipped in `preprocess.json`, and adding to
it is a preprocessing version bump that invalidates the corpus snapshot.

#### 4.1.4 Steps 7–9: what is deleted, what is placeholdered, what is preserved

This is the table the brief asked for. Each row cites the reason, and the reason is nearly
always classifier spec §4.6: **whatever we do here must run at inference, so the question is
never "is this useful?" but "can we do this identically, cheaply, forever?"**

**DELETED entirely** (the span disappears; a per-row receipt records what and how many chars):

| What | Rule | Measured | Why delete rather than placeholder |
|---|---|---|---|
| Auto-generated template footer | everything from the `--` line, or from `This ticket was created automatically`, to end of field | 99 rows; **217.5 → 76.0 tokens, −65.1%**; drops corpus >128-token rate from 2.3% to 0.1% | Identical across 99 tickets and 60 organisations. It is pure constant; a placeholder for a constant carries no information and costs tokens |
| Corpus-frequency boilerplate sentences | exact-match against a frozen blocklist of sentences occurring ≥20× **in the training split only** | 149 types, 9,935 instances = **48.2%** of sentence instances; only 1.2% of stripped mass is technical, and that is protected by §4.1.3 | Generic "moves" (`Мешает работать, но не блокирует.` ×123). They occur in every class, so they are label-independent noise. Sentence-level exact match is the safest possible deletion primitive |
| Quoted reply blocks | lines matching `^\s*>`, plus the `> On <date>, X wrote:` inline form | 10 rows here (`^>`), 3 inline; `wrote:`/`писал:` 10 | The quoted text is a *different* ticket's content and mislabels this one. Delete rather than mark, since the quote depth is not signal |
| Signature blocks | from a sign-off marker (`С уважением`, `Best regards`, `Kind regards`, `Regards,`, `--` on its own line) to end of field, capped at 5 lines | `С уважением` 154 rows, `Regards` 72 | Names, phones, titles, company blurbs. Deleting is cheaper and safer than redacting each element |
| URL query strings and fragments | everything after `?` or `#` | 110 of 210 URLs | **A plain normalisation rule.** Query strings are high-entropy, near-zero-signal, per-row-unique spans of exactly the kind classifier spec §4.6 says to normalise away — the same argument as `<UUID>` and `<TS>`. It happens also to destroy this fixture's 81 URL-borne tokens by construction (§2.4), but that is a side effect, not the justification |
| Whitespace runs, zero-width, CSV artefacts | §4.1.2 | 207 rows | Noise, no information |

**REPLACED by a stable placeholder** (the placeholder is a special token, §4.1.6):

| Placeholder | Detector | Measured | Why keep a marker at all |
|---|---|---|---|
| `<EMAIL>` | regex | 265 instances / 262 rows | Presence of an address is weak signal (`notifications`, `auth` invite flows). Deletion would break sentence structure |
| `<PHONE>` | regex, 6 RU + 2 EN surface formats | 179 rows | Same |
| `<URL:{class}>` | regex; `{class}` from a **first-party host allowlist** (`docs`, `api`, `logs`, `portal`, `cdn`, `app`, `hooks`) else `other` | 210 URLs; allowlist covers 208/210 | `hooks.*` → `integrations`, `cdn.*` → `cdn` is real signal. The host is first-party and therefore not PII; path and query are dropped |
| `<CARD>` | regex over digit runs **including `*` masks** — a plain `\d{13,19}` regex matches **0** here because the fixture pre-masks them | 21 rows | Presence of a card is signal for `billing` |
| `<ACCOUNT>` | 20-digit RU settlement account (р/с, к/с), keyword-anchored; Luhn/checksum-validated where the format allows | **0 here** | The product owner named bank and transactional details explicitly. **This detector has no positive test case in this fixture** (§7.1) and rests entirely on canaries (§6.4) |
| `<BIC>` | `БИК` + 9 digits | **0 here** | Same |
| `<TXN>` | payment/transaction reference: keyword-anchored (`платёж`, `транзакц`, `payment id`, `order`) + alphanumeric run | **0 here** | Same |
| `<PASSPORT>` | `паспорт` / `серия … номер` + `\d{2}\s?\d{2}\s?\d{6}` | **0 here** | Same |
| `<INN>` | keyword-anchored 10/12-digit value + control-digit validation | 143 rows keyword, 24 with value | Presence signals `billing`/accounting documents |
| `<ORG>` | `(ООО|АО|ЗАО|ПАО|ИП)\s*[«"]…[»"]` plus NER ORG | 32 rows regex | Legal-entity names are identifying; the *fact* of one is billing signal |
| `<IP>` | regex, both v4 and v6 | 17 rows | Networking signal; the address itself is customer infrastructure |
| `<TS>` | ISO-8601 and common RU/EN date forms | 100 rows | High-entropy, zero-signal, per §4.6's list |
| `<ID>` | `req_[0-9a-f]+`, `rule_\d+`, `i-[0-9a-f]{6}`, ticket/order ids | 182 rows / 219 instances | **Placeholder, not preserve.** The *class* of identifier is signal (a `req_` id means the user pasted an API response); the value is unique per row, so preserving it adds one hapax per row and zero generalisation |
| `<UUID>`, `<B64>` | regex | 0 / 1 here | **Token-budget rules, not security rules.** A 32-char base64 run costs ~11 tokens and carries no label signal |
| `<PERSON>` | NER — see §4.1.5 | 359 RU + 265 EN rows | Names carry no service signal; the placeholder keeps grammar intact |

**Money amounts are preserved, account numbers are not.** `12 400 ₽` (36 rows [measured]) is a
billing-signal quantity of the same kind as `16 ТБ`; a settlement account is an identifier with
no signal at all. The distinction is "does the *value* generalise across tickets" — amounts
loosely do, identifiers never do.

**PRESERVED verbatim.** Everything in the §4.1.3 protected list, plus: all prose, all product
and component nouns in both languages, priority, case, and single newlines.

**The critical judgement call, stated explicitly.** The brief's framing is that logs, tracebacks
and code blocks are "not useful for a classifier". **On this corpus that is measurably wrong for
the majority of cases.** [measured] the "log paste" in a typical multi-line ticket is *one line*
appended to prose — `SignatureDoesNotMatch`, `RequestTimeout: we did not receive a complete
request body`, `429 Too Many Requests`, `400 invalid cidr`, `401 Unauthorized` — and it is the
most label-discriminative token in the row. Deleting evidence blocks wholesale would be the
single most damaging thing this pipeline could do. What *is* noise inside a log block is its
skeleton: paths, line numbers, addresses, thread ids, repeated timestamps. §4.1.9 removes those
and keeps the rest.

#### 4.1.5 Step 11: NER, and the tier question it forces

Names need NER; regex cannot do Russian names, especially inflected ones (`от Ивана Петрова`).
But NER is the one Tier-A candidate with a nontrivial inference cost, and Tier A means it runs
on every production request.

**Recommendation: run NER in Tier A, using `slovnet` via a Presidio custom recognizer**, subject
to a benchmark gate.

- [`natasha`/`slovnet`](https://github.com/natasha/slovnet) (MIT) reports **PER F1 0.959, LOC
  0.915, ORG 0.825 on factRuEval-2016** (PER 0.984 / ORG 0.951 on Collection5), in a **27 MB
  model using 205 MB RAM at 25.3 articles/s on CPU** — "1–2% worse than current BERT SOTA by
  DeepPavlov but 60 times smaller". A news *article* is an order of magnitude longer than a
  186-character ticket, so per-ticket cost should be **well under 40 ms [estimate] — measure it
  before committing.**
- **Gate:** if measured p95 NER latency on real tickets is ≤ 25 ms (leaving the §9.3 budget
  intact at 42.9 + 25 + 5 ≈ 73 ms p50 against a 250 ms p95 allowance), NER stays in Tier A.
- **If the gate fails**, NER is demoted to Tier B: it no longer edits text, and instead
  (a) flags rows for the redaction-audit queue, (b) contributes to an offline-built **name
  gazetteer** which *is* Tier A (a compiled Aho-Corasick match is microseconds). Accept and
  report the residual skew: names novel at inference are not redacted. That is a compliance
  exposure on *logs*, not on the training corpus, and the architect's log-redaction policy is
  the mitigation — flag it to them, do not solve it here.

**Presidio's role, and its honest limits.**
[`microsoft/presidio`](https://github.com/microsoft/presidio) (MIT) is recommended as the
**orchestration layer** — recognizer registry, span+score results, a consistent anonymizer with
stable placeholders, and an auditable per-detection record — not as a source of Russian
detection quality. Its own documentation is explicit that the default configuration ships
"recognizers and models for English" only, that adding a language requires configuring the NLP
engine *and* writing recognizers with translated context words, and that
[it cannot guarantee finding all sensitive information](https://github.com/microsoft/presidio).
Every Russian entity in §2.3 — ИНН, ОГРН, `+7` phone formats, `ООО «…»`, паспорт — is a
recognizer we write ourselves. **Presidio buys structure, not coverage.**

#### 4.1.6 Placeholder tokenisation

[measured] under XLM-R, angle-bracket placeholders cost **3–5 tokens each**:

| Tokens | Placeholders |
|---|---|
| 5 | `<ACCOUNT>` (`▁< AC CO UNT >`) |
| 4 | `<EMAIL>` · `<CARD>` · `<PERSON>` · `<PASSPORT>` · `<TXN>` · `<BIC>` |
| 3 | `<URL>` · `<PHONE>` · `<ID>` · `<TS>` · `<IP>` · `<INN>` · `<ORG>` · `<UUID>` · `<B64>` |

Classifier spec §4.6 says to add special tokens "only if the redaction placeholders are
themselves over-segmented". **They are.** Recommendation: **add the placeholder set as special
tokens and resize the embedding matrix** — ~15 tokens, each becoming a single unsplittable id.
This also removes a real failure mode: a ticket whose text legitimately contains `<URL>` cannot
be confused with a redaction. Per §4.6's ordering rule, if vocabulary trimming ever happens,
**trim last**.

Note the combined effect on the token-budget claim: placeholders save 2.1% of tokens and then
special tokens save a little more, against a boilerplate rule that saves 5.5%. Nobody should
sell placeholders on efficiency grounds.

#### 4.1.7 Step 14: near-duplicate detection

**Recommended: `datasketch` MinHashLSH** (MIT), 128 permutations, char-5-gram shingles over
whitespace-collapsed lowercase text, union-find over candidate pairs to form clusters.

- **Two signatures per row, unioned** (§2.5): one over the pre-boilerplate-stripping text
  (catches template cliques) and one over the Phase-1 output (catches content duplicates the
  template masked). [measured] the alert clique is 98 rows in the first and 8 in the second.
- **Threshold 0.80 for the "drop across split boundary" rule** (classifier spec §2.6), which
  [measured] touches 141 rows (2.8%) and 9 boundary-straddling clusters (120 rows). **0.60 is
  too aggressive** — 570 rows (11.4%) — and 0.50 collapses a third of the corpus.
- **Emit `dup_cluster_id` and `dup_cluster_size` as columns; do not delete rows in this
  pipeline.** Dropping is the splitter's decision (classifier spec §2.6) and the gold sampler's
  (runbook §4.3), and both need to see the cluster structure. A pipeline that silently
  deduplicates makes the runbook's "report how many were dropped" impossible.
- **Cluster-size-aware sample weights** are the softer alternative to dropping: weight
  `1/sqrt(cluster_size)`. Offer it as a training hyperparameter; do not decide it here.

**Ruled out: SimHash.** [`text-dedup`](https://github.com/ChenghaoMou/text-dedup) (Apache-2.0)
benchmarks MinHash at **ARI 0.7293 in 3.01 s** versus SimHash at **0.6463 in 140.03 s** on
NEWS-COPY — worse quality at ~45× the cost. **Runner-up: `text-dedup` itself**, which wraps
MinHash/SimHash/SuffixArray/Bloom behind TOML configs and has Spark and `datasets` backends.
Switch to it if the corpus exceeds ~1M rows or a distributed run is needed; below that, we want
cluster ids as a first-class column and `datasketch` gives them directly.

**What MinHash will miss, by design.** The fixture's README documents semantic-but-not-lexical
clusters (the Postgres connection-pool cluster) scoring 0.2–0.5. They are not detectable with
character shingles and **must not be chased by lowering the threshold** — that is how you get
the 34.5% collapse at 0.50. If semantic duplicates matter, the tool is embedding-space clustering
over the frozen-encoder vectors (classifier spec §4.1 row 2), which the project already has to
build for a baseline. Specify it as a follow-up diagnostic, not a Phase-1 step.

#### 4.1.8 Step 15: language identification

**Recommended: `lingua-py`** ([Apache-2.0](https://github.com/pemistahl/lingua-py)), restricted
to the `{ru, en}` language set, with `py3langid` (BSD-3) as a cheap cross-check.

Three reasons, in order of weight:

1. **It is the only candidate with span-level multi-language output.** Its documented
   experimental mode "returns contiguous sections with language identification and position
   indices" — which is exactly the operational definition of code-switching this corpus needs
   and the ratio rule cannot express.
2. **Short-text accuracy.** It uses n-grams of size 1–5 rather than trigrams only, and this
   corpus's median description is 186 characters with 262 rows under 80.
3. Restricting the detector to two languages both raises accuracy and cuts the memory footprint.

**Ruled out as primary: fastText `lid.176`.** The model is released under
[CC-BY-SA-3.0](https://fasttext.cc/docs/en/language-identification.html) — a share-alike
obligation on a derived artifact that legal will ask about and that buys nothing here — the full
model is 126 MB (the 917 kB `.ftz` trades accuracy), and it has no span-level output. **Kept as
primary only if** lingua's throughput becomes a problem at ≥1M rows.
**`py3langid` is kept as a second opinion**: [measured] it already resolves the RU/EN binary on
5,006/5,013 rows, it is BSD-licensed and dependency-light, and **disagreement between two
independent LID systems is itself a useful routing signal** for the quarantine queue.

**Two rules that matter more than the library choice:**

- **Run LID on masked prose, after step 6.** Pasted English error strings otherwise drag Russian
  tickets toward `en`: [measured] 82 RU-dominant rows contain an English error-string pattern and
  1,071 contain ≥5% Latin letters.
- **Operational definition of the three-way tag** (replacing the ratio rule in classifier spec
  §6.3):
  - `lang_primary ∈ {ru, en}` — lingua's verdict on masked prose;
  - `code_switched = true` iff lingua's span output assigns ≥1 contiguous span of ≥3 words to the
    *other* language, **outside** protected spans and placeholders;
  - `lang_uncertain = true` iff lingua's confidence is below its own recommended floor, or the
    two LID systems disagree. These rows go to the quarantine queue, not to the `mixed` slice.

  This definition is falsifiable and reproducible; "15% minority script" is neither. **[measured]
  caveat:** on this fixture I cannot validate the code-switch detector against ground truth
  because the CSV carries no code-switch flag — only the README's aggregate count of 237. Report
  the detector's count against 237 as a sanity check, and treat a large gap as a finding about
  the detector, not about the corpus.

#### 4.1.9 Log, traceback and code-block reduction

Applied inside evidence blocks only (a contiguous run of lines that is not prose: no sentence
punctuation, or matching a log/stack pattern).

| Element | Action | Reason |
|---|---|---|
| Exception/error class, error message first line | **preserve** | the label signal |
| Stack frame *module/class* names, top 3 frames | **preserve** | `django.db.utils.OperationalError` names the component |
| File paths, line numbers, memory addresses, thread/goroutine ids, PIDs | delete | zero signal, high hapax rate |
| Repeated log timestamps and level prefixes within a block | delete after the first | constant per block |
| Frames 4+ of a stack trace | replace with `<FRAMES n=k>` | tail frames are framework internals |
| Any evidence block exceeding **64 tokens** | keep first 2 lines, last 2 lines, and every line matching a protected pattern; replace the remainder with `<LOG_TRUNCATED lines=k>` | bounds the worst case |

[measured] **this fixture never triggers the 64-token rule** (max row 228 tokens, max evidence
block well under). The rule exists for the real corpus, where a pasted 500-line traceback is
routine, and it must be written and tested now because it cannot be retrofitted after a snapshot
is frozen.

#### 4.1.10 Step 13: `services`, `labels`, `flags` hygiene

**`services` — the target. Canonicalisation is exactly runbook §2's `norm_services()`, in
Python:** NFC → strip → collapse internal whitespace → lowercase → drop empties → dedupe → sort.
[measured] this resolves all 59 raw tokens to the 20 canonical names and repairs 39 spacing, 38
casing, and 40 duplicate defects.

Three rules beyond that:

- **A token not in `labels.json` is a hard error**, not a silent drop: quarantine the row, count
  it, report it, and require a human decision (it is either a typo, a retired service, or a
  taxonomy change requiring the mapping table from classifier spec §4.8). [measured] 0 such
  tokens here — which is a property of a synthetic fixture and will not hold on a real export.
- `|S| > 4` violates the output contract → quarantine and report. [measured] 0 here.
- `|S| = 0` (391 rows) is **not** an error: it is the untriaged/unactionable population. Tag
  `label_state = 'empty'` and exclude from supervised training by default. **Do not delete
  them** — classifier spec §4.6 makes abstention a first-class output, and these rows are the
  only material available for calibrating it. [measured] they cannot be identified by length:
  median 57 tokens versus 62 for labelled rows.

**`labels` and `flags` — carried as metadata, excluded from the input text by default.** This is
classifier spec §2.7 and it is not the pipeline's call to overturn. Canonicalise them anyway
(split on `;` and `,`, trim, lowercase, dedupe, sort; [measured] 243 distinct `labels` values,
5 `flags` values) so that the feature-ablation arm can be run cheaply if Q2 is ever answered.
**On this corpus the ablation is unanswerable**: the fixture's README states `labels`/`flags` are
as-of-export with no creation-time history, so the target-leakage question cannot be settled
here. Say so in the model card rather than running an ablation whose result would be meaningless.

#### 4.1.11 Step 16: quality filters and quarantine

Nothing is deleted. Every filter sets a reason code and the row stays in the snapshot with
`quarantine_reason`, so counts are reportable and decisions are reversible.

| Filter | Threshold | Measured | Action |
|---|---|---|---|
| Empty after cleaning | title and description both empty | 0 | quarantine `empty_after_clean` |
| Too short | < 8 tokens post-clean | 0 at 8; 8 rows < 16 tokens; 68 < 24 | quarantine `too_short`; **do not raise the threshold to 24** — short tickets are a required evaluation slice (spec §6.3) and 262 rows have descriptions under 80 chars |
| Too long | > 4× `max_len` post-clean | 0 | flag `heavy_truncation` (this is also LLM-fallback trigger case 4) |
| Boilerplate-dominant | > 80% of post-clean tokens came from deleted spans | 0 after the alert footer rule | quarantine `boilerplate_only` |
| Language uncertain | §4.1.8 | 7 rows (py3langid `bg`/`fr`/`sr`) | flag `lang_uncertain`, keep |
| Unknown service token | any | 0 | quarantine `bad_label`, **report and escalate** |
| Structural parse failure | any | 0 | quarantine `parse_error` |

#### 4.1.12 Library stack for Phase 1 — recommendation, runners-up, switch conditions

| Job | **Recommended** | Licence | Why | Runner-up, and when to switch | Weakness on Russian |
|---|---|---|---|---|---|
| Encoding repair | **ftfy** | Apache-2.0 | conservative, will not "fix" correct text; handles layered mojibake | none needed | none material; Cyrillic mojibake is its core case |
| Unicode | **`unicodedata` (stdlib)** | PSF | NFC is all that is required | — | — |
| Transliteration | **none — `unidecode` ruled out** | — | would Latinise 67% of the corpus | — | fatal |
| PII orchestration | **presidio** | MIT | recognizer registry, span+score audit records, stable anonymiser placeholders | hand-rolled span engine, if presidio's dependency weight is unacceptable at inference | **English-only by default**; every RU entity is a recognizer we write; context words must be translated; explicit no-guarantee disclaimer |
| RU NER backend | **slovnet (Natasha)** | MIT | PER F1 0.959 factRuEval, 27 MB / 205 MB RAM / CPU-only, 60× smaller than BERT SOTA at 1–2% cost | spaCy `ru_core_news_lg` if a spaCy pipeline is wanted for other reasons; a RU BERT NER if F1 must go higher and the Tier-A budget allows | ORG F1 0.825 is the weak spot, and ORG is exactly where `ООО «Ромашка-Сервис»` lives — back it with the regex rule, do not rely on NER alone |
| PII, generic | **scrubadub — ruled out** | Apache-2.0 | detectors are US/GB-centric (SSN, GB driving licence, US/GB/CA postcodes); name detection add-ons are English spaCy/Stanford | — | no Russian entity coverage at all; nothing here is reusable |
| Credential / secret detection | **none — the whole tier is removed** | — | §2.3: production `title`/`description` carry no application secrets. `gitleaks`, `detect-secrets` and `trufflehog` were all ruled in by the first revision of this document and are **all ruled out now**. Carrying a credential scanner "just in case" is dead scaffolding: it costs a dependency, a licence review (trufflehog is AGPL-3.0), and a gate that can only ever fire on a false positive | Reinstate only if the premise in §2.3 is withdrawn — that is open question P6 | — |
| RU morphology | **pymorphy3** | MIT | maintained continuation of pymorphy2; needed for the TF-IDF/keyword baselines' lemmatisation and RU service-synonym matching | — | **not needed for the encoder path** — SentencePiece handles morphology; scope it to baselines only |
| Label-error detection | **cleanlab 2.9.0**, multi-label API | Apache-2.0 | confident learning over out-of-sample probabilities; §4.3.3 | — | language-agnostic — it never sees the text, only `pred_probs` |
| Tier-2 model | **scikit-learn** (TF-IDF, OvR logistic regression, `GroupKFold`) | BSD-3 | already required by classifier spec §3 baseline 3; produces the probabilities tiers 3 and 5 and both §6.1 taxonomy reports consume | — | char 3–5 grams handle RU morphology without a lemmatiser |
| Embeddings / projection | **none — `sentence-transformers`, `torch`, `transformers` and `umap-learn` ruled out** | — | The embedding tier they existed for is cut (§4.3.4). Dropping them is [measured, validation §4] **754 MB of `torch` plus a `numba`/`llvmlite` JIT toolchain plus a 471 MB checkpoint** removed from the pipeline environment — the single largest reduction in its dependency surface | Reinstate only through gate G18 | — |
| Language ID | **lingua-py** (`{ru,en}` only) | Apache-2.0 | span-level multi-language output; strong on short text | **py3langid** (BSD-3) kept as a cross-check; **fastText lid.176** only at ≥1M rows | none material for RU/EN; the risk is running it on unmasked text |
| Near-duplicates | **datasketch MinHashLSH** | MIT | cluster ids as a column; 2.1 ms/row [measured] | **text-dedup** (Apache-2.0) above ~1M rows or for a Spark backend; **SimHash never** | char-5-grams are language-agnostic; RU morphology is handled by character n-grams |
| E-mail quote/signature | **own deterministic rules first**; **talon** (Apache-2.0) only if quote markers exceed 2% of rows | Apache-2.0 | [measured] only 10 rows have `^>` quoting here; talon's ML signature classifier was trained on English ENRON mail and has no documented Russian support | `email-reply-parser` (MIT) is simpler but equally English-pattern-driven | **`С уважением` is 154 rows and neither library knows it** — the RU sign-off marker list is ours to write and version either way |
| Row representation | **polars → Parquet**, one file per phase, never in place | MIT / Apache-2.0 (pyarrow) | columnar, typed, hashable, cheap to diff between phases | **pandas** is fine at 5k rows and this choice is aesthetic below ~100k; it stops being aesthetic at 1M | — |
| Training handoff | **HF `datasets`** at the training boundary only | Apache-2.0 | Arrow-backed, integrates with the trainer | — | — |

**Where the honest gap is.** [measured] this fixture contains **zero** bank accounts, БИК codes,
СНИЛС, passport numbers, transaction references, or UUIDs. Every detector for those classes is
therefore **completely untested by this corpus**, and they are precisely the classes the product
owner named as material. Their correctness rests entirely on the seeded-canary protocol in §6.4.
Do not read a green Phase-1 gate on this fixture as evidence that bank-detail handling works.

#### 4.1.13 The honest answer on PII false negatives

**No detector removes all personal data, and neither does Phase 2.** Regex only finds shapes
someone anticipated; NER on Russian is [measured by its authors] PER F1 0.959 and **ORG F1
0.825**, so roughly one company name in six is missed; and an LLM second opinion is a
probabilistic detector with no recall guarantee that — critically — **has already seen the text
by the time it reports.** None of these is a compliance story on its own.

Four layers, and only the first is a guarantee:

1. **Structural destruction where the structure allows it.** Delete signature blocks and quoted
   reply blocks wholesale rather than redacting element by element; drop URL query strings and
   fragments; delete the auto-alert footer. A rule that removes a whole region cannot miss an
   entity inside it, and [measured] the signature-block rule alone covers 226 rows where names,
   phones and job titles cluster.
2. **Seeded canaries with a hard recall gate** (§6.4) — the only quantitative measurement
   available, and the only test the bank/passport detectors get at all (§7.1).
3. **Phase 2's `residual_pii` output** as an independent second opinion (§4.2.3). It protects
   everything downstream of the Phase-2 call; **it cannot protect the Phase-2 call itself**,
   which is why the phase ordering is load-bearing (§4.2.6).
4. **A human read of a stratified sample** (§6.5), plus an **egress self-check** on the finished
   corpus (§6.3) and a standing adversarial exercise: once per snapshot, someone tries to write a
   ticket that defeats the pipeline, and whatever they find becomes a rule and a canary.

And one non-technical layer that outranks all four: **keep the corpus inside the perimeter**
(§4.2.6). Residual personal data in a corpus that never leaves is a risk to manage under our own
controls; the same data sent to an external provider is a disclosure that has already happened
and cannot be undone by improving the redactor afterwards.

---

### 4.2 Phase 2 — at most one LLM call per row

#### 4.2.1 What the call is for, given §0.2(1)

**The call's job is span classification over Phase-1 output, producing rules and metadata —
never replacement text.**

The model receives the Phase-1 text with sentences pre-numbered by code. It returns, per
sentence or per span, a role assignment and a small set of row-level facts. **Code, not the
model, decides what happens next.** Nothing the model emits is ever concatenated into the model
input text.

This is a stricter version of the brief's "bounded rewrite", and it buys three things a rewrite
cannot:

- **Hallucination becomes structurally impossible for the text path.** The model emits indices
  and enum values, not prose. There is no channel through which invented text can reach the
  corpus.
- **Tier A stays deterministic.** The mined rules (blocklist additions, new PII patterns,
  evidence-span regexes) are frozen into `preprocess.json` and run at inference in microseconds.
  No LLM on the serving path, no skew.
- **Failure degrades to nothing.** A refused, malformed, or timed-out call yields no metadata for
  that row and contributes nothing to rule mining. Phase-1 output is untouched and remains a
  complete, valid corpus. This is the fallback requirement the brief asks for, and under this
  design it is satisfied trivially rather than by careful engineering.

#### 4.2.2 Output schema

Structured outputs with `enum` constraints, the same mechanism as
[llm-fallback-policy §4.1](./llm-fallback-policy.md). Sketch:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["sentences", "row_flags"],
  "properties": {
    "sentences": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["idx", "role"],
        "properties": {
          "idx":  {"type": "integer"},
          "role": {"type": "string",
                   "enum": ["symptom", "evidence", "ask", "context",
                            "impact", "pleasantry", "chase_up", "quoted",
                            "signature", "unrelated"]},
          "carries_service_evidence": {"type": "boolean"}
        }
      }
    },
    "row_flags": {
      "type": "object",
      "additionalProperties": false,
      "required": ["is_unactionable", "is_forwarded_thread", "language", "code_switched"],
      "properties": {
        "is_unactionable":     {"type": "boolean"},
        "is_forwarded_thread": {"type": "boolean"},
        "is_multi_issue":      {"type": "boolean"},
        "language":            {"type": "string", "enum": ["ru", "en", "other"]},
        "code_switched":       {"type": "boolean"}
      }
    },
    "residual_pii": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["quote", "kind"],
        "properties": {
          "quote": {"type": "string"},
          "kind":  {"type": "string",
                    "enum": ["person", "org", "email", "phone", "card", "inn",
                             "passport", "address", "account", "bank", "other"]}
        }
      }
    }
  }
}
```

**Allowed to alter:** nothing in the text. It emits indices, enums, booleans, and verbatim
quotes.

**Note the `other` bucket in `residual_pii`.** There is no `secret` value, because §2.3 says
production ticket text has none. If the model nonetheless keeps returning credential-shaped
strings under `other`, that is **evidence to reopen P6** — and it costs nothing, because the
bucket had to exist anyway. This is the one place a removed premise gets a cheap feedback loop
rather than a dormant detector.

**Forbidden, and enforced by a validator rather than by instructions:**

- every `residual_pii.quote` must be a **verbatim substring** of the input; if not, the entry is
  discarded and the row is counted in `phase2_invalid_span`;
- every `idx` must exist; out-of-range indices invalidate the whole response;
- `sentences` must cover every input index exactly once;
- any schema violation → the row's Phase-2 output is dropped entirely. **Partial acceptance is
  not allowed** — a model that got the shape wrong is a model whose judgement on that row is not
  trustworthy.

#### 4.2.3 What is done with the output

| Output | Consumer | Effect |
|---|---|---|
| `role ∈ {pleasantry, chase_up}`, aggregated across rows | **rule mining** | sentences classified as filler in ≥N rows and never as `symptom`/`evidence` are proposed as **blocklist additions**. A human approves the diff (it is a text file; [measured] the ≥20× blocklist is 149 lines). The approved blocklist is Tier A |
| `role ∈ {quoted, signature}` | **rule mining** | proposes new sign-off markers and quote patterns, especially Russian ones the libraries do not know |
| `carries_service_evidence` | **rule mining** | proposes additions to the §4.1.3 protected-span list — this is the mechanism that stops the blocklist from ever eating a signal-bearing sentence |
| `residual_pii` | **redaction audit** | each quote becomes a candidate detector pattern (generalised by a human) and a **canary-set addition**. Rows with residual PII are quarantined pending a rule |
| `is_unactionable` | **row metadata** | training filter and abstention-calibration stratum. Cross-check against `label_state='empty'` ([measured] 391 rows) |
| `is_forwarded_thread`, `is_multi_issue` | **row metadata** | `is_multi_issue` is a genuine cardinality confound worth a slice |
| `language`, `code_switched` | **row metadata** | third opinion alongside lingua and py3langid; three-way disagreement routes to quarantine |

#### 4.2.4 Is one call per row worth it, and how we decide

**Honest position: probably not on this corpus, plausibly yes on a real one, and it is cheap to
find out.** The measurements argue both ways:

*Against:* [measured] the largest available win — removing 48.2% of sentence instances that are
generic filler — is already achievable deterministically with 1.2% collateral damage from a
149-line blocklist. [measured] this corpus has no tracebacks, no forwarded threads to speak of
(10 rows), no mojibake, and no exotic PII. There is very little left for a language model to add.

*For:* the corpus is synthetic and compositional. Its filler is a closed set of hand-written
"moves" (which is exactly why the frequency cut works so well); real customer filler has a long
tail that a frequency threshold will not reach. The rule-mining framing means Phase 2's value is
**one-off and permanent** — the rules it mines keep working for every future snapshot, so the
5,013 calls are paid once, not per retrain.

**The decision rule, and it is the ablation the brief asks for:**

> Train the recommended checkpoint on **B1 (Phase-1 text)** and on **B3 (Phase-1 + Phase-2-mined
> rules)**. Identical splits, identical hyperparameters, identical decode, **5 seeds each, same
> seed list** (classifier spec §5.3, §5.6). Compare validation `R@P90` as **mean ± sd**.
> **Ship Phase 2 only if the mean improvement is ≥ 1.0pp and the seed intervals do not overlap.**
> Below that, Phase 2's rule-mining pass runs **once, on a 1,000-row sample, as a discovery
> exercise**, its approved rules are merged into Phase 1, and the per-row call is switched off
> permanently.

The 1.0pp bar is deliberately low relative to the 5pp ship criterion in classifier spec §6.6,
because Phase 2's cost is one-off and its risk — under the rule-mining design — is bounded to
"we added some blocklist rules a human approved". It is not low enough to be free: a phase that
moves nothing gets removed.

**Corollary worth stating: because the output is rules, Phase 2 does not need every row.** If the
budget is tight, run it on a stratified 1,000-row sample (random, plus the multiline stratum,
plus the placeholder-touched stratum, plus the quarantine strata) and mine from that. The per-row
metadata (`is_unactionable`, `residual_pii`) is the only output that genuinely wants full
coverage, and `residual_pii` is the one with a compliance argument behind it. **Recommendation:
full coverage for the first snapshot** (the residual-PII sweep justifies it on its own), sampled
thereafter.

#### 4.2.5 Cost and caching

The binding constraint here is **not money.** Working, at Sonnet-tier list pricing from
[llm-fallback-policy §7](./llm-fallback-policy.md) ($3/MTok in, $15/MTok out, cache reads at
roughly a tenth of input price):

| Component | Tokens | Cost/call |
|---|---|---|
| Cached instruction + role-definition prefix | ~1,800 | ~$0.0005 |
| Uncached ticket (mean 71 tokens [measured] + sentence numbering ≈ 110) | 110 | $0.0003 |
| Output (sentence roles + flags) | ~200 | $0.0030 |
| **Per call** | | **≈ $0.0038** |
| **5,013 rows** | | **≈ $19** [estimate] |
| **50,000 rows** | | **≈ $190** [estimate] |

**Prompt caching applies and is worth wiring** ([llm-fallback-policy §4.3](./llm-fallback-policy.md)):
the role definitions and instructions are a stable prefix, the ticket is the varying suffix. The
same gotcha applies — **count the real prefix before choosing a tier**, because the minimum
cacheable prefix is 512 tokens on Opus 5, 1,024 on Sonnet 5, and **4,096 on Haiku 4.5**, and a
~1,800-token prefix silently does not cache on Haiku with no error, just full price.

**Model tier: mid-tier is correct for Phase 2.** The task is span classification into 10 enum
values with a schema constraint — the least demanding job in the pipeline. Unlike Phase 3
(llm-fallback-policy §7 recommends the top tier there because a precision regression on an
agent-visible field is expensive), Phase 2's errors are caught by a human approving the mined
rule diff. Run a 200-row tier comparison against human-adjudicated span roles and pick on
measured agreement; **do not default to the expensive tier out of caution when a human gate
already exists.**

The real costs of Phase 2 are: engineering time, one more artifact to version, and the risk that
someone later "simplifies" it into a rewrite and reintroduces skew. Weigh those, not the $19.

#### 4.2.6 Where the model runs, and why the phase ordering is load-bearing

**Default and recommendation: a self-hosted model inside the production perimeter for Phases 2
and 3.** Three reasons, and the first is the one that decides it:

1. **The deterministic redactor's false-negative rate on personal data is not zero and cannot be
   proven to be zero** (§4.1.13). Sending "redacted" text to an external provider is a bet that
   the redactor is perfect. The NER backend's own published **ORG F1 is 0.825**, so roughly one
   company name in six survives; names in oblique Russian cases are the next-largest gap. Do not
   take that bet with customer text.
2. **It removes a dependency on an approval we do not have.** Ticket text is personal data, and
   whether pseudonymised text may leave the perimeter is a governance decision owned by legal and
   the DPO (§2.9, open question P1). Defaulting to self-hosted means **the pipeline is not
   blocked on that decision** and does not have to be redesigned whichever way it goes.
3. **Feasibility is settled by volume, exactly as in llm-fallback-policy §3.** This is a
   **one-off batch of ~5k–50k rows with no latency requirement**, which is a far easier ask than
   the production fallback path that document already found feasible on CPU. A dataset build that
   takes six hours instead of forty minutes costs nothing.

**Phase ordering is the load-bearing control, and it must be stated as a rule:** Phase 1's
deterministic redaction runs **before** any LLM call, without exception. Phase 2's
`residual_pii` output is a *downstream* protection — it protects the training corpus, the logs,
Phase 3, and the human queue. It cannot protect the Phase-2 call itself, because by then the text
has already been sent. Under a self-hosted deployment inside the perimeter, that residual
exposure never crosses a boundary at all, which is the fourth reason self-hosting is the default.

**Upgrade path.** If the governance decision (P1) permits pseudonymised text to leave the
perimeter, run a **200-row head-to-head** — self-hosted versus hosted frontier — scored against
human-adjudicated span roles, and switch if the hosted model's agreement is materially better.
Record which model produced every row's metadata in the provenance record either way, because a
mid-snapshot model switch is otherwise invisible and irreproducible.

---

### 4.3 Phase 3 — label-error detection as a cheap-first cascade

#### 4.3.0 The cascade, and why the ordering is what it is

Phase 3 is **a cascade, cheapest and most-grounded first**, each tier narrowing the pool the next
one sees. The LLM judge is late and small; on this corpus it sees **[measured] 6.2% of labelled
rows**.

**Tier numbering is stable and tier 4 is deliberately left empty.** It held an embedding-kNN
flag and a UMAP diagnostic, both **cut on measurement** (§4.3.4, and
[`knn-tier-validation.md`](./knn-tier-validation.md)). The slot is kept rather than renumbered so
that cross-references in this document, the architecture document and the proposal stay valid,
and so the next person to propose an embedding tier finds the measurement instead of a gap.

| Tier | Instrument | Cost | Contamination risk | Rows it selects on this fixture [measured] |
|---|---|---|---|---|
| **1** | Deterministic label conflicts inside duplicate groups (§4.3.2) | free, seconds | **none** — no model involved | ~14 routed, ≤4 masked |
| **2** | TF-IDF + OvR logistic regression, grouped CV (§4.3.3) | **113 s CPU** | **none** — produces probabilities, not opinions | (produces the input to 3, 5, and the §6.1 taxonomy reports) |
| **3** | cleanlab confident learning over tier-2 probabilities (§4.3.3) | seconds | **none** — routing signal only, never auto-applied | **161** at `score < 0.20`; **313** at the recommended widened selector (§4.3.11) |
| ~~**4**~~ | ~~Embedding kNN + UMAP~~ — **cut, see `knn-tier-validation.md`** | ~~`torch` + `umap-learn`, ~1.9 review-hours/snapshot~~ | — | **−9.4pp recall at matched budget** (§4.3.4) |
| **5** | **LLM judge**, constrained adjudication (§4.3.5–4.3.9) | ~375–600 calls, ~$2–4 | **real** — every verdict is model provenance | **288 rows judged** (union + audit) |
| **6** | Human review of the residual (§4.3.10–4.3.11) | ~8 person-hours | none | **~200 rows** [estimate] |

**Why this order, argued rather than asserted.** Four reasons, in order of weight:

1. **Grounding.** A tier-1 conflict is *evidence*: two texts that a character-5-gram signature
   says are the same carry different labels, and at least one of them is wrong or the taxonomy is
   ambiguous. No model asserted anything. An LLM verdict is an *opinion* with an evidence span
   attached. Prefer evidence to opinion when both are available, and they are.
2. **Contamination.** Tiers 1–3 never emit a label and never change one; they emit a *ranking of
   which rows deserve attention*. Under the fence in §4.3.13 that ranking still may not touch the
   gold set — but it creates no model-provenance labels, so nothing it flags is disqualified from
   val/test on provenance grounds. Every LLM verdict does create model provenance. **Doing the
   same job with a cheaper instrument that carries less permanent cost is the whole argument.**
3. **Cost, which is not close.** [measured] the tier-2 + tier-3 pass is **113 seconds of CPU and
   $0** for the whole corpus. The first revision of this document specified ~6,000 LLM calls for
   the same job. Even at $30 that is a bad trade when the cheap instrument is better grounded.
4. **Cleanlab is structurally bad at exactly one thing, and that thing is what the LLM is for.**
   Confident learning assumes a **class-conditional noise process** — labels flip with a
   probability that depends on the true class
   ([Northcutt, Jiang & Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels",
   JAIR 2021](https://arxiv.org/abs/1911.00068)). When `auth` and `access-control` overlap because
   the *taxonomy* is ambiguous, there is no flip process to estimate: both labels can be right,
   and which one is intended depends on a written guideline the model has never seen. That is a
   definitional question, and the LLM judge — with the decision rule in its prompt (§4.3.6) — is
   the only automated instrument in this pipeline that encodes the guideline. **That is its
   non-redundant job, and it is the only one it should be given.**

#### 4.3.1 What each tier may and may not do

One rule, applied uniformly, and it is the thing that keeps the cascade safe:

> **No tier in Phase 3 changes a label automatically. Tiers 1–3 route. Tier 5 routes and, under
> the narrow unanimity condition in §4.3.9, may auto-apply a *removal*. Only tier 6 (a human)
> may add a label.**

The empirical justification is in §4.3.3: [measured] dropping the 161 cleanlab-flagged rows from
training *hurt* macro average precision by 0.79pp on this corpus. Flagged rows were hard, not
wrong. A pipeline that had auto-applied cleanlab's verdict would have silently degraded the
dataset and reported a green gate.

#### 4.3.2 Tier 1 — deterministic label conflicts

**Input:** the `dup_cluster_id` columns already produced by §4.1.7 (two signatures, unioned),
plus exact-match groups on normalised `title`, on `description`, and on `title+description`.

**The conflict grade**, computed over the label sets of a group's labelled rows. Strict set
inequality is the wrong rule — it fires on legitimate variation — so the rule is graded:

| Grade | Definition | Interpretation | Action |
|---|---|---|---|
| `identical` | all pairs equal | nothing to see | none |
| `nested` | every pair satisfies `A ⊆ B` or `B ⊆ A` | **incompleteness, not contradiction** — one annotator recorded fewer services, which is the runbook §3 "reliable positives, unreliable negatives" problem in miniature | **mask** the missing services out of the negative set for the smaller rows (§4.3.8); do not queue |
| `partial` | pairwise overlap, but neither nested nor disjoint | genuine ambiguity, *or* a template whose instances legitimately differ | queue, **but cap at 5 rows per cluster** |
| `disjoint` | some pair shares no service | **hard contradiction, highest-precision signal in the pipeline** | queue every row, always |

**The cluster-size cap is not a nicety.** [measured] the single `partial` cluster at Jaccard 0.80
is the 98-row auto-alert clique, whose members carry `[managed-postgres, monitoring]` versus
`[compute, monitoring]` because they are alerts on different resources. A strict rule would put
98 correct rows in a human queue. The cap routes 5 and records the cluster id so a reviewer can
pull the rest if the sample looks wrong.

**Yield on this fixture [measured]: ~14 rows queued** (3 `disjoint` at Jaccard 0.60 + 5 sampled
from the capped 98-row `partial` cluster + 6 from exact-`description` groups) **and up to 4 rows
masked** (2 `nested` groups, which may overlap between the title and near-dup groupings — the
implementation must de-duplicate across grouping methods before acting). That is a real finding
and it must not be generalised: the fixture's labels are author-assigned from a scenario
bank, so 564 of 565 same-title groups agree exactly. **On a real corpus with a
CatBoost-contaminated `services` column this tier is where the cheapest wins live.** Re-run it on
real data before concluding anything about tier 1's value; it costs seconds.

**Report per snapshot:** conflicting groups and rows by grade, by grouping method, and by
threshold. A rising `disjoint` count between snapshots is an annotation-quality alarm.

#### 4.3.3 Tiers 2 and 3 — TF-IDF probabilities and confident learning

**Tier 2 is the baseline classifier spec §3 item 3 already requires**, computed once and used
three times.

Specification:

- **Features:** word 1–2 grams (`min_df=2`, sublinear TF) ∪ char\_wb 3–5 grams (`min_df=3`,
  sublinear TF, 200k features cap). Character n-grams handle Russian morphology without a
  lemmatiser, which is classifier spec §3's own reasoning.
- **Model:** one-vs-rest logistic regression, `class_weight='balanced'`, on the **Phase-1 text**.
- **Cross-validation: `GroupKFold`, 5 folds, grouped on the §4.1.7 near-duplicate cluster id.**
  This is the subtlety that decides whether the whole tier works. If a row's near-duplicate sits
  in the training fold, the model predicts that row with inflated confidence, its label-quality
  score rises, and **the exact error you are hunting is hidden.** [measured] the effect is
  large and in the expected direction:

| CV scheme | macro-AP | micro-AP | micro-F1@0.5 | macro-F1@0.5 |
|---|---|---|---|---|
| **`GroupKFold` on dup clusters** | **0.9384** | **0.9751** | **0.9397** | **0.8807** |
| Plain shuffled `KFold` | 0.9338 | 0.9807 | 0.9488 | 0.8944 |

  Ungrouped folds inflate micro-F1 by **+0.91pp** and micro-AP by **+0.56pp** — pure duplicate
  leakage. (Macro-AP moves the other way because the leakage concentrates in the large template
  clusters, which are head-service rows.)

- **Runtime [measured]: 113 s** for 5 folds × 20 services × 4,622 rows, single machine, no GPU.

**Two required metrics fall out of tier 2 for free**, and both must be reported (§6.1):

- **ΔA — the text-pipeline delta.** Same labels, same folds, raw text versus Phase-1 text.
  [measured] macro-AP **0.9351 → 0.9384 (+0.33pp)**, micro-F1 **0.9335 → 0.9397 (+0.62pp)**.
  This is a **GPU-free proxy for the B0-vs-B1 rung** of the §5.2 ladder, available in minutes
  instead of hours. It does not replace the encoder ablation; it tells you early whether the text
  pipeline is helping or hurting.
- **ΔB — the label-noise delta.** Same text, folds fixed, tier-3-flagged rows dropped from the
  **training** folds only, evaluated on **untouched** held-out folds. The evaluation set must not
  be cleaned, or you are measuring that you deleted the hard examples rather than that you
  removed noise. [measured, approximate — see §9.7] macro-AP **0.9381 → 0.9302 (−0.79pp)**,
  micro-F1 **0.9403 → 0.9406 (flat)**.

  **Read that result carefully, because it is the most useful negative result in this document.**
  On this corpus, removing the rows confident learning flags makes the model *worse*, and worse
  specifically on the macro metric — i.e. on the rare services. The flagged rows are not noise;
  they are hard-but-correct examples, which is exactly what you would expect from a fixture whose
  labels were author-assigned and internally consistent. **A pipeline that auto-dropped
  cleanlab's flags would have degraded the corpus and passed every gate.** Hence §4.3.1.

**Tier 3 — cleanlab, with the correct multi-label API.** Verified against
[cleanlab 2.9.0 docs](https://docs.cleanlab.ai/stable/cleanlab/multilabel_classification/index.html)
(Apache-2.0, Python ≥ 3.10) and run locally at that version:

| Purpose | Function | Notes |
|---|---|---|
| Flag rows | `cleanlab.multilabel_classification.filter.find_label_issues(labels, pred_probs, return_indices_ranked_by=None, …)` | `labels` is an **iterable of iterables of class indices** (`[[1,2],[1],[0],…]`), **not** a binary matrix; `pred_probs` is `(N, K)`. Returns a boolean mask by default, or indices ranked by `self_confidence` / `normalized_margin` / `confidence_weighted_entropy` |
| Per-service flags | `…filter.find_multilabel_issues_per_class(labels, pred_probs)` | needed for the rare-tail guard below |
| Rank rows | `…rank.get_label_quality_scores(labels, pred_probs, method='self_confidence', aggregator_kwargs={'method':'exponential_moving_average','alpha':0.8})` | one score per row; `…get_label_quality_scores_per_class` returns K arrays |
| Corpus-level | `…dataset.common_multilabel_issues`, `rank_classes_by_multilabel_quality`, `overall_multilabel_health_score`, `multilabel_health_summary` | these are the **taxonomy diagnostics**, and they are more valuable here than the per-row flags |

> **The wrong entry point is `cleanlab.filter.find_label_issues`.** That is the multi-class API;
> it expects one integer label per example and will either raise or silently mis-handle a
> multi-label target. Use the `multilabel_classification` namespace.

[measured] on this corpus, tier 2 probabilities into cleanlab 2.9.0:

| Selector | Rows | % of 4,622 |
|---|---|---|
| `find_label_issues` mask | **95** | 2.06% |
| `get_label_quality_scores < 0.20` | 126 | 2.73% |
| `< 0.30` | 313 | 6.77% |
| `< 0.50` | 670 | 14.50% |
| **Union of mask and score < 0.20 (the recommended selector)** | **161** | **3.48%** |

Per-service flag counts [measured] are the interesting output — they read as a taxonomy report,
not as a list of mistakes:

`console-ui` **25** · `networking` 17 · `compute` 10 · `auth` 6 · `api-gateway` 6 · `billing` 6 ·
`notifications` 6 · `cdn` 4 · `integrations` 4 · `managed-redis` 3 · `access-control` 2 ·
`dns` 2 · `terraform-provider` 2 · `logging` 1 · `managed-postgres` 1 · `subscriptions` 1 ·
`backups` 0 · `message-queue` 0 · `monitoring` 0 · `object-storage` 0

And `common_multilabel_issues` gives the *direction*: `networking` is flagged 17 times as
"in given label, not in suggested" and **0** times the other way; `console-ui` is flagged **13
one way and 12 the other**. A class that is wrong in both directions is the signature of genuine
definitional overlap rather than noise — which is precisely §2.8's `console-ui` problem,
recovered automatically.

**Cleanlab's assumptions, stated honestly:**

- **Class-conditional noise.** Confident learning estimates the joint distribution of noisy and
  true labels under the assumption that flips depend on the true class (JAIR 2021). Ambiguity
  that depends on the *text* rather than on the class violates this. It is why tier 5 exists.
- **Out-of-sample probabilities are required.** The documentation is explicit that scores are
  "most accurate when computed based on out-of-sample `pred_probs`". In-sample probabilities make
  every row look clean.
- **Rare classes are fragile.** `find_label_issues` takes `min_examples_per_class=1` by default,
  so it will happily prune a class down to one example. [measured] it flags **2 of
  `terraform-provider`'s 15 positives (13%)** and 4 of `cdn`'s 38 (11%). Classifier spec §2.8
  says a service below ~50 positives cannot be learned or evaluated reliably; letting an
  automated tier erode those further is unrecoverable. **Guard: any flag on a service with < 50
  positives routes to a human (rule R2, §4.3.10) and is never auto-applied.**
- **Tier 3 is complementary to tier 1, not a superset.** [measured] only **27 of the 95** flagged
  rows sit in a duplicate group of size > 1. The tiers find different things and both are worth
  running.

#### 4.3.4 Why there is no embedding tier — **cut, with evidence**

**This slot previously held an embedding-kNN label-agreement flag plus a UMAP taxonomy
diagnostic. Both are cut.** An independent adversarial validation
([`knn-tier-validation.md`](./knn-tier-validation.md)) measured the proposal against the cascade
it was meant to complete, and it lost on the one measurement that decides the question. On
injected label errors with known ground truth, at a **matched reviewer budget of 185 rows**,
spending the entire budget on tier 3's ranking recovers **0.683 ± 0.022** of the injected errors,
while splitting it as tier-3 top-129 + tier-4 top-56 recovers **0.589 ± 0.015** — tier 4 costs
**9.4pp of recall on the ambiguous-pair noise process and 5.5pp on the random one**, with the
sign consistent across 3 seeds, both representations, and every budget from 25 to 500 rows
(validation §3.7). The ambiguous-pair process was constructed to be tier 4's *best* case, because
§4.3.0 point 4 argues confident learning is structurally weak there — and tier 4 still lost by
more than 2× on it (0.143 vs 0.309 at 56 rows). It is not that the kNN signal is noise: 0.276
recall at 56 rows against a 1.2% chance rate is real. It is **strictly dominated** signal, and a
dominated instrument on a capped review budget is worse than no instrument, because the slots it
consumes are slots tier 3 would have used better. Two supporting results close the remaining
doors: the tier-4-only flags fail the same ablation control that reframed tier 3 — dropping the
41 e5-only rows from training costs **−0.42pp** macro-AP versus **−0.28pp** for 41 *random* rows
(validation §3.8), i.e. they are ordinary training data — and hand review of 32 of them put
strict precision at ~5%, with 29% of the list being one auto-alert template (validation §3.6).
**"Corroborating only" does not rescue it.** My routing rule added no rows to the queue, only
sort order; but the queue is capped (§4.3.11), so sort order decides which rows a human reaches,
and reordering a capped list by a dominated ranking is the same trade with the cost hidden.

**The UMAP diagnostic fails separately, and its replacement is free.** Its purpose was "do `auth`
and `access-control` occupy the same region", read off a 2-D projection. On this corpus the
inter-service centroid distances a human would read off that plot correlate with the same
distances in the real 384-d space at Spearman **0.011 / 0.208 / 0.239** across three seeds, and
the plot's own geometry moves between seeds at Spearman 0.484–0.608 (validation §3.9) — the
reading is neither faithful to the space nor stable. It is also 20 overlapping multi-label
classes on one scatter at mean cardinality 1.90, so most points carry two or more colours and
there is no honest colouring. **Per-service-pair confusion computed from tier 2's out-of-fold
probabilities answers the same question better and costs nothing**, because those probabilities
are already computed: it is directional (`P(b | a-only)` and `P(a | b-only)` separately),
conditional on base rate, and diffable between snapshots — and it recovers the fixture's planted
ambiguities, topped by `monitoring`→`notifications` at 0.223. It is now a required §6.1 output
alongside the label co-occurrence matrix. Neighbour purity, if anyone still wants it, is a table
over the full-dimensional vectors and needs no projection — but the validation notes it tracks
class frequency almost perfectly, so it restates "rare classes are rare", which tier 2's
per-service AP already says with a decision attached.

**I accept this verdict without reservation, and one part of it is a lesson about my own
process.** My §9.7 caveat flagged that the tier-4 numbers were a TF-IDF proxy and that
"every tier-4 threshold must be re-derived". That instinct was right and I under-reacted to it:
the real `multilingual-e5-small` flags **36 rows at the specified ≥0.95 cut, not 56**, and its
top-56 overlaps my proxy's top-56 by only **34%** (validation §3.2, §3.3). Two TF-IDF variants
agree with *each other* at 73% overlap while neither agrees with the encoder — so the flag list
was tracking the representation, not the labels. **Every threshold I published in this section
was a threshold on a distribution that does not exist.** The correct handling was not a caveat;
it was a gate: an instrument whose selection is that sensitive to an arbitrary representation
swap should not have entered the spec before someone measured what it selects. That gate now
exists in general form in §6.2 (G18).

**What is not claimed, and the conditions that would reopen this.** Embeddings are not useless
here and kNN label agreement is not a bad idea in general; the claim is narrow and local — on the
only corpus we can measure, it returns less per reviewer-row than widening tier 3, which is
already built. Validation §7 lists the flip conditions and they are inherited verbatim: a **real**
adjudicated error set on which tier 4's ranking beats tier 3's at matched budget; a corpus whose
labels come from CatBoost or from multiple annotators rather than one author; a reformulated flag
that is null-normalised, not mean-Jaccard (which is cardinality-biased — [measured] `|S|=1` rows
are 34% of the corpus and 54% of the flag list), and template-clique-aware; or the encoder
arriving in the pipeline for another reason, which collapses the marginal cost to a kNN pass.
**Note the last one does not by itself reopen the tier:** negative value at matched budget is not
fixed by becoming cheap. If it is reproposed under any of these, it enters on the gate in §6.2
(G18), not on a routing rule.

One related requirement must not be lost in the deletion. An embedding-distance-to-centroid
signal was doing double duty as the out-of-distribution detector for
[`llm-fallback-policy.md` §2 case 4](./llm-fallback-policy.md). **That requirement is real and it
survives, but it does not belong in this pipeline**: it is a *serving-time* signal, its threshold
is set on the production model's validation set, and the production classifier is already an
encoder. It is handed to the classifier spec and the fallback policy as an explicit requirement —
tracked, not deleted.

#### 4.3.5 Tier 5 — what the LLM judge is scoped to

The judge runs on the **union of four strata, and nothing else**:

| Stratum | Definition | Rows [measured] |
|---|---|---|
| **S1 — cheap-tier flags** | tier 1 conflicts (queued grades) ∪ tier 3 selector | ~14 ∪ 161 |
| **S2 — ambiguous-pair swap candidates** | row carries exactly one member of an ambiguous pair, and the tier-2 out-of-sample probability of the *other* member both exceeds the carried one and exceeds 0.5 | **6** (`auth`/`access-control` 3, `integrations`/`notifications` 1, `cdn`/`networking` 2) |
| **S3 — `console-ui` candidates** | carries `console-ui` with `P < 0.30`, or lacks it with `P > 0.70` | **65** |
| **S4 — random audit** | 2% of labelled rows, drawn by seeded hash, judged regardless of any flag | **92** |
| **Union** | | **288 rows = 6.2% of labelled rows** |

S2 and S3 are what §2.8's "59.1% of rows touch an ambiguous family" becomes once it is made
*measurable*: instead of routing 2,730 rows because they contain an ambiguous service, route the
71 where a model trained on the corpus disagrees with the stored label in the specific direction
the taxonomy is known to be weak. **This is the single largest reduction in the revised design**,
and it is only possible because tier 2 exists.

S4 is not optional. Without an unflagged stratum there is no way to estimate the judge's
behaviour on rows the cheap tiers considered clean, and therefore no way to estimate the cascade's
recall.

#### 4.3.6 Verdict schema and the constrained-adjudication pattern

The pattern is [llm-fallback-policy §4.1 constrained adjudication](./llm-fallback-policy.md),
reused rather than reinvented, applied to stored labels instead of to a classifier shortlist. The
shortlist is the row's existing `services` set; the verdict is accept/reject per candidate; the
output is schema-constrained with an `enum` over service names, so an invalid service name is not
a failure mode that has to be handled. Requiring an `evidence` span is a grounding constraint and
gives reviewers something to audit — same argument, same schema, one addition.

The general caution from
[Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)](https://arxiv.org/pdf/2306.05685)
applies and is why §4.3.9 shuffles candidate order: LLM judges exhibit **position bias, verbosity
bias and self-enhancement bias**. That paper's headline agreement-with-humans result is for
*pairwise preference* judging, not for taxonomy label adjudication. **Do not import their
agreement number as an expectation for this task.** Measure ours on the pilot.

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["verdicts"],
  "properties": {
    "verdicts": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["service", "accept", "evidence", "basis"],
        "properties": {
          "service":  {"type": "string", "enum": ["access-control", "api-gateway", "auth", "..."]},
          "accept":   {"type": "boolean"},
          "evidence": {"type": "string"},
          "basis":    {"type": "string",
                       "enum": ["explicit_mention", "error_string", "symptom_implies",
                                "no_support_in_text", "contradicted_by_text"]}
        }
      }
    },
    "proposed_additions": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["service", "evidence"],
        "properties": {
          "service":  {"type": "string", "enum": ["access-control", "..."]},
          "evidence": {"type": "string"}
        }
      }
    }
  }
}
```

**Validation, enforced in code:**

- `evidence` must be a **verbatim substring of the Phase-1 text**. A non-verbatim span
  invalidates that verdict — the model is claiming support that is not there. Count these as
  `judge_ungrounded`; a rate above 2% is a prompt bug, not a data finding (§6.2).
- An `accept: true` with `basis: no_support_in_text` is self-contradictory → invalid.
- `verdicts` must cover the row's stored services exactly.
- **Evidence spans for `reject` verdicts are the interesting case:** the model must quote the
  span it believes the label was mistakenly derived from, or return the empty string with
  `basis: no_support_in_text`. This is what makes a rejection reviewable in seconds instead of
  minutes, and it is the difference between a 2.5-minute and a 6-minute human queue item.

#### 4.3.7 The three hard cases the taxonomy deliberately contains

The judge prompt carries the **annotation guideline's own definitions** (classifier spec §2.2)
plus an explicit decision rule per confusable pair. Definitions alone are what produced the
ambiguity; rules are what resolve it. The rules below must be authored by the product owner, not
by me — I am specifying their form and the monitoring that catches them being wrong.

**`auth` vs `access-control`** — [measured] 213 rows carry both, out of 1,216 carrying either
(17.5%), and `access-control`+`auth` is the second most common co-occurring pair in the corpus.

> Rule: `auth` = proving *who you are* (login, SSO/SAML, 2FA, session lifetime, token issuance
> and expiry, password reset). `access-control` = what an established identity *may do* (roles,
> permissions, scope, membership, offboarding). Tiebreakers: the user could not get a session →
> `auth`; the user had a session and was denied an action → `access-control`; an API key was
> rejected because it *expired* → `auth`; because its *scope* was insufficient → `access-control`.
> **Both is a correct and common answer** and the judge must not be pushed toward exclusivity.

**`monitoring` vs `logging`** — [measured] they co-occur on only 37 of 877 rows (4.2%), so they
are largely disjoint in practice.

> Rule: `monitoring` = metrics, dashboards, alert rules, thresholds, notifications from alerts.
> `logging` = log ingest, search, retention, log delivery.
> **Monitor:** if the judge's post-adjudication co-occurrence rate for this pair rises materially
> above the stored 4.2%, the judge has started hedging by accepting both. That is a measurable
> failure and it triggers the §6.2 gate.

**`console-ui` vs the broken component** — [measured] 793 rows carry `console-ui`, of which only
156 carry it alone and **637 co-label it with the component**. It is 9.1% of all positives.

> Rule: `console-ui` applies only when the defect is *in the console itself* — rendering,
> navigation, form validation, a control that does nothing, a message that vanishes too fast. If
> the console correctly displays a backend failure, the label is the backend component **only**.
> **Guard:** a judge applying this rule strictly would strip `console-ui` from a large share of
> the 637 co-labelled rows. That may be right and it may be the model over-applying a rule it
> was just handed. §4.3.10 caps it: **no service's reject rate may exceed 15% [estimate —
> calibrate on the pilot] without explicit human sign-off on that service's whole reject list.**
> **Corroboration [measured]:** cleanlab independently flags `console-ui` more than any other
> service (25 rows) and flags it in *both* directions (13 given-not-suggested, 12
> suggested-not-given). Two instruments agreeing that this service is the corpus's weak point,
> from completely different evidence, is the strongest signal in the pipeline that the *guideline*
> needs work rather than the labels.

#### 4.3.8 The missing-label direction

The judge can only reject what is present. The missing-label problem is real and runbook §3
already identifies its consequence: labels with reliable positives and **unreliable negatives**,
trained with plain BCE, actively teach the model that correct services are negatives.

**Recommendation: allow proposals, never auto-apply them as positives, and prefer masking to
adding.**

| Option | Effect on the label set | Effect on provenance | Verdict |
|---|---|---|---|
| Auto-apply proposals as positives | recall of labels ↑ | **every added positive is LLM-authored** — model provenance, and the exact distillation failure of proposal §3 | **No** |
| Route proposals to the human queue | recall ↑ where a human agrees | `human_after_judge` (anchored, §4.3.13) | Yes, for high-value cases only (§4.3.10) |
| **Mask the proposed service out of the negative set** | recall unchanged; the model is simply not told "this is a negative" | the *label* is unchanged, so provenance is unchanged | **Recommended default** |
| Ignore proposals | nothing | nothing | The fallback if the mask ablation shows no gain |

The masking option is the one worth arguing for. It only requires the judge to say *"I am not
confident this is a negative"* — a far weaker and far safer claim than *"this is a positive"* —
and it maps directly onto machinery classifier spec §5.1 already specifies: per-service masking
in the BCE loss, or Asymmetric Loss, which was chosen partly for tolerance of mislabelled
samples ([Ridnik et al., ICCV 2021](https://arxiv.org/abs/2009.14119)). **Effect on precision:**
masking cannot lower label precision because it adds no positives; it can only reduce the
gradient pressure that pushes a genuinely-correct service down. The cost is a slightly weaker
negative signal, which is exactly what the ablation measures.

**Tier 1 produces the same masking signal for free.** A `nested` conflict group (§4.3.2) is a
missing-label observation with no model in it at all: two duplicate texts, one carrying `{auth,
api-gateway}` and one carrying `{auth, api-gateway, object-storage}`. Mask `object-storage` out
of the smaller row's negatives. [measured] this fires on 4 rows here; on a real corpus it should
be the dominant source of masking, and it is strictly better evidence than a judge proposal.

**The ablation (4 arms, 5 seeds, identical everything else, evaluated once on the blind gold
test set):** (i) Phase-1 labels; (ii) + judge rejections applied; (iii) (ii) + proposals masked;
(iv) (ii) + proposals applied as positives. Arm (iv) is included **only** as a diagnostic and is
**not shippable** regardless of its score, because it puts LLM-authored positives in the training
set — llm-fallback-policy §5. If (iv) wins by a lot, that is a finding about label recall to
report to the annotation effort, not a licence to ship it.

#### 4.3.9 Self-consistency, and the call budget after the cascade

**The ≤1-call-per-row constraint is scoped per phase, and I am stating that explicitly rather
than assuming it.** A judge call is a second LLM call for a row that Phase 2 already saw; a
2-of-3 self-consistency scheme is a third and fourth. If the constraint were global the judge
could not exist at all, which is clearly not the brief's intent. After the cascade the question
is close to moot: the judge sees **6.2% of labelled rows**, so Phase 3's *corpus-wide* call rate
is **0.08 calls per labelled row**, an order of magnitude under the Phase-2 budget.

**Escalating self-consistency, over the §4.3.5 strata only.**

1. **One call per judged row** with the candidate services in stored order.
   [measured] 288 rows → **288 calls**.
2. **Escalate to two more calls with the candidate order shuffled**, only where the first call
   produced ≥1 `reject` or ≥1 `proposed_addition`. Because the pool is pre-filtered to
   *suspicious* rows, the escalation rate is much higher than it would be corpus-wide —
   [estimate] 40–60% rather than 10–20%.
3. Decide: **3-of-3 unanimous reject → eligible for auto-apply (subject to §4.3.10 R1–R7).
   2-of-3 → human queue. 1-of-3 → ignore**, and log it.

Order shuffling rather than temperature is the correct mechanism and llm-fallback-policy §4.1
already says why: temperature is rejected with a 400 on current models, so sampling variance has
to come from varying the prompt. Shuffling is also the direct mitigation for the **position
bias** documented by Zheng et al.

**Call count and cost, revised [measured rows, estimated escalation]:**

```
judged rows                                   288   [measured]
first-pass calls                              288
escalated rows @ 50%                          144   [estimate]
escalation calls (2 each)                     288
------------------------------------------------------
total Phase-3 LLM calls                     ~ 576   → but see below
```

At the more conservative 30% escalation rate the total is ~460; at 60% it is ~634. **Plan on
~375–600 calls**, against the first revision's ~6,000 — a **10–16× reduction**. Cost at the
§4.2.5 rates with the ~2,000-token cached service-definition prefix: **≈ $2–4** [estimate] for
the whole corpus, down from $30–45.

**The cost saving is not the point and should not be argued as one.** $30 was never a constraint.
The point is that **576 model opinions carry an order of magnitude less contamination surface
than 6,000**, and that the rows the judge does see were selected by instruments that carry none.

**On fusing Phases 2 and 3 into one call.** The argument *for* is real: it halves the calls, the
model sees the ticket once, and evidence spans could be shared between the two jobs. **The
argument against is decisive and there are two of them:**

1. **The contamination fence makes fusion impossible.** Phase 2 **must** run on the gold val/test
   rows — their text must be preprocessed identically to training text or there is skew inside
   our own evaluation. Phase 3 **must never** run on the gold rows (llm-fallback-policy §5). One
   fused call cannot both run and not run on the same rows. Splitting the fused call by row
   population reintroduces two prompts, at which point nothing was fused.
2. **Fusion destroys the ablation.** With one call you cannot attribute a change to text cleaning
   or to label repair, so B2/B2b and B3 in §3 collapse into B4 and the decision rules in §4.2.4
   and §4.3.8 have nothing to compare.

There is a third, softer reason: a model asked to clean text and judge labels in one pass can
rationalise one to fit the other. Keeping them independent keeps the two signals independent.

A fourth reason arrives with the cascade: **the two phases now run on different-sized
populations.** Phase 2 sees every row; Phase 3 sees 6.2% of them. Fusing would drag the judge
onto 4,622 rows purely to keep the call count down, which is the opposite of what §4.3.0 is for.

**Do not fuse.**

#### 4.3.10 The routing rule — which rows go to a human

A row is routed to the human-supervision queue if **any** of the following fires. Rules are
ordered by expected volume and each carries its measured or estimated population.

| # | Condition | Rationale | Population |
|---|---|---|---|
| **R1** | Reject on a row where `|S| = 1` — applying it would empty the label set | An emptied row silently leaves the training corpus; that is too consequential for an unreviewed model verdict | [measured] 1,569 rows have `|S| = 1` (34% of labelled rows) |
| **R2** | Reject on a service in the rare tail (`terraform-provider` 15, `message-queue` 31, `cdn` 38, `dns` 41, `managed-redis` 98) | Classifier spec §2.8: below ~50 positives a service cannot be learned or evaluated. Losing rare positives is unrecoverable and cannot be detected by any aggregate metric | [measured] 212 rows carry a rare-tail service |
| **R3** | 2-of-3 self-consistency (not unanimous) | The judge is not sure; a human is the tiebreak | [estimate] ~40% of escalated rows |
| **R4** | `proposed_additions` with ≥2 entries, or one that would take `|S|` to 5 (violating the `≤4` contract) | Aggressive expansion is either a real missing-label finding or a hallucinating judge; both need eyes | [estimate] ~1–2% of judged rows |
| **R5** | A "swap" within an ambiguous pair: reject `auth`, propose `access-control` (or the reverse), same for `monitoring`/`logging`, `console-ui`/component | These are exactly the distinctions the taxonomy admits are contested; a model resolving them unilaterally is out of its authority | [measured] the S2/S3 strata are **71 rows**, and only the *flip* subset of those routes |
| **R6** | Any service whose reject rate over the judged pool exceeds **15%** [estimate] | A systematic prompt or guideline problem, not many independent label errors. **The whole service's reject list is held** pending sign-off | Whole-service, rare |
| **R7** | Random **2%** audit sample regardless of verdict (stratum S4) | The only way to measure the auto-applied arm's error rate **and** the cascade's recall on rows the cheap tiers called clean. **Non-negotiable** | [measured] 92 rows |
| **R8** | Tier-1 `disjoint` conflict, and tier-1 `partial` up to the 5-row cluster cap | Direct evidence, no model. These route whether or not the judge agrees — the judge's verdict is attached as context, not as a gate | [measured] ~14 rows |

**Not routed:** `judge_ungrounded` verdicts (evidence not verbatim). Those are a pipeline defect,
not a label question; discard the verdict, count it, and if the rate exceeds 2% fix the prompt
(§6.2). Also not routed: tier-1 `nested` conflicts, which go to masking (§4.3.8).

**Queue sort order: tier-3 rank, and nothing else.** The first revision used
`knn_disagreement` as a corroborating secondary sort key. It is removed with tier 4 — and the
removal matters more than "corroborating" made it sound. The queue is capped, so sort order
decides which rows a reviewer actually reaches, and promoting rows by a ranking that
[measured, validation §3.7] recovers fewer injected errors per slot demotes rows that recover
more. Sort by the tier-3 quality score ascending; break ties by `|S| = 1` first (rule R1 is the
most consequential), then by rare-tail membership (R2).

#### 4.3.11 Queue size and person-hours — revised for the cascade

The cascade changes this number substantially, and it supersedes the ~650-row / 46-hour figure in
the first revision of this document and the reconciliation in
[`finetuning-dataset-pipeline.md` §7.1](../proposals/finetuning-dataset-pipeline.md).

**Spend the reviewer budget deeper in tier 3, not sideways.** The tier-4 deletion does not
literally free 56 queue slots — under the old "corroborating only" rule tier 4 routed no rows, it
only reordered (§4.3.10). What it frees is ~1.9 review-adjacent hours per snapshot, a dependency
stack, and a worse sort key. But validation §3.7 measures something more useful than a refund:
**the marginal reviewer row is worth more further down tier 3's ranking than anywhere else
available.** On the ambiguous-pair noise process, recall of injected errors goes
**0.309 (56 rows) → 0.543 (129) → 0.683 (185) → 0.826 (300)**; on the random process,
**0.377 → 0.720 → 0.862 → 0.952**. The curve is still climbing steeply at the budget we are at.

**Recommendation: widen the tier-3 selector from `mask ∪ score < 0.20` to `mask ∪ score < 0.30`.**
[measured] that takes the selector from **161 to 315 rows** (3.48% → 6.82% of labelled rows), and
it is the natural stopping point: at `< 0.30` the `find_label_issues` mask contributes only 2 rows
the score threshold has not already taken, so beyond it the two selectors have merged and the
mask stops adding independent evidence. Both sizings are given below; the second is recommended.

```
labelled rows                                        4,622  [measured]

TIER SELECTION (before any LLM call)          score<0.20   score<0.30 (recommended)
  T1 conflicts: disjoint + capped partial + desc-dups
                                                     ~14          ~14   [measured]
  T1 nested → masking, not queued                      ≤4           ≤4   [measured]
  T3 cleanlab, mask ∪ quality-score cut               161          315   [measured]
  S2 ambiguous-pair swap candidates                     6            6   [measured]
  S3 console-ui candidates                             65           65   [measured]
  S4 2% random audit                                   92           92   [measured]
  --------------------------------------------------------------------
  union judged by the LLM                             288        ~430   [estimate for the
                                                    (6.2%)      (9.3%)   widened union]

QUEUE (after judging, widened selector)
  reject rate on a pre-filtered pool              40–60%     [estimate — the pool is
                                                              selected to be suspicious;
                                                              expect it to fall as the
                                                              selector widens]
  take 45% of 430                                    ~194 rows with ≥1 reject/proposal
  of those, 3-of-3 unanimous            ~60%         ~116  auto-apply-eligible
  of those, 2-of-3 → R3                 ~40%          ~78  queued
  R1 (|S|=1 rejects, 34% of rows)                     ~66  (overlaps R3)
  R2 (rare-tail rejects)                              ~15  [measured: 65 rare-tail rows
                                                            in the widened selector vs 43
                                                            at <0.20 — R2 grows, and that
                                                            is the point of R2]
  R4 (aggressive proposals)                           ~10
  R5 (ambiguous swaps, from the 71-row S2∪S3 pool)    ~30
  R7 (2% audit, queued regardless)                     92  [measured]
  R8 (tier-1 conflicts, queued regardless)             14  [measured]
  --------------------------------------------------------------------
  union, de-overlapped                            ~250–320 rows [estimate]
  midpoint                                            ~285 rows = 6.2% of labelled rows
```

**Person-hours, and this is the number that should appear in the proposal:**

| Item | Volume | Rate | Hours |
|---|---|---|---|
| Human review queue, **widened selector** | **~285 rows** [estimate] | 2.5 min | **11.9** |
| *(queue at the narrow `< 0.20` selector, for comparison)* | *~200 rows* | *2.5 min* | *8.3* |
| Phase-3 prompt calibration pilot | 100 verdicts | 2.5 min | 4.2 |
| Phase-1 validation sample (§6.5) | 200 rows | 1.5 min | 5.0 |
| Phase-2 validation sample (§6.5) | 150 rows | 1.5 min | 3.8 |
| Phase-3 validation sample (§6.5) | 100 verdicts | 1.5 min | 2.5 |
| Blocklist review (§6.5) | 149 entries, once per version | — | ~1.0 |
| **Total, recommended** | | | **~28 person-hours** [estimate] |

**The widening is an increase of ~3.6 h against a measured recall gain**, and I am flagging it as
a trade rather than burying it: it is the difference between roughly the 185-row and the 300-row
points on validation §3.7's curve, i.e. **+14pp of injected-error recall on the ambiguous process
and +9pp on the random one**. It also stays comfortably inside the architecture document's
600-row queue cap. If P4 comes back with a hard ~25-hour ceiling, take the narrow selector and say
which recall was bought back; do not restore tier 4 to fill the gap, since [measured] it buys
negative recall at any budget.

**Reconciliation with the two other documents.** The architecture doc costed the queue at
600 rows × 75 s = 12.5 h; the first revision of this document costed 650 × 2.5 min = 27 h plus
19 h of validation = 46 h; the proposal §7.1 split the difference at ~44 h. **All three are now
superseded.** The rate stays at 2.5 min — that argument was about the *difficulty* of a queue
item, not its count, and the cascade makes queue items *harder* on average, not easier, because
the easy ones were filtered out by instruments that did not need a human. What changed is the
count: **~285 rows, not 600–650.** The architecture doc's 600-row cap remains a sensible capacity
control and is still non-binding after the widening.

**Single figure for the proposal: ~28 person-hours, of which ~12 h is the review queue.**

**Scaling caveat, and it is the one that matters for budgeting.** These counts come from a
fixture whose labels are author-assigned and internally consistent — cleanlab flags 2.06% of
rows. On a real corpus with a CatBoost-contaminated `services` column [estimate] the flag rate is
**10–20%**, giving 460–920 tier-3 rows at the same corpus size, a judged pool of ~600–1,100, and
a queue of **~400–700 rows ≈ 17–29 h**. **Do not commit a queue budget from this fixture.** Run
tiers 1–3 on the real export first; they cost [measured] under four minutes of CPU and produce
the exact number (open question P10).

**This is in addition to the runbook §4.2 gold-set budget of ~70 person-hours, and it must not
be taken out of it.** Four reasons, and the first alone settles it:

1. **The queue is a filtered set, and the filter is a model's disagreement.** That is precisely
   the failure runbook §0 exists to prevent: "you cannot build the test set by filtering on
   tickets a human corrected." Substituting the queue for the gold set rebuilds the Case-A bias
   with an LLM in place of CatBoost.
2. **Queue reviewers are not blind.** They see the judge's verdict and evidence. Classifier spec
   §2.2 requires blindness because anchoring is real and inflates measured agreement — the same
   automation-bias mechanism, and llm-fallback-policy §5 says there is no safe version of
   pre-annotation.
3. **The queue cannot surface silent errors** — rows where the stored label and the judge agree
   and both are wrong. Only blind re-annotation of a random sample finds those.
4. **No agreement statistic.** One reviewer per item means no double annotation, no Krippendorff
   α, and therefore **the α ≥ 0.67 taxonomy gate cannot be computed.** [measured] 59.1% of
   labelled rows touch an ambiguous-family service; on this taxonomy that gate is the one most
   likely to fire, and the queue would leave it unmeasured.

**Priority statement, in case the budget is ever forced:** if only 70 hours exist, **spend all of
it on the blind gold set and skip Phase 3 entirely.** The gold set is a prerequisite for
concluding anything; the queue is an optimisation of training-label quality that improves a
number you cannot yet measure. The cascade makes this trade-off less painful than it was — tiers
1–3 cost four minutes of CPU and zero person-hours, so **even under a zero-human budget you can
run them and get the taxonomy reports** (§4.3.3, §6.1); you simply cannot act on the per-row
flags.

#### 4.3.12 When to drop the LLM judge entirely

This is a live option, not a rhetorical one, and it should be decided on a measurement rather
than on taste. The cheap tiers carry no contamination cost; the judge carries a permanent one
(§4.3.13). So the judge has to earn its place.

**Measure this on the pilot (100 human-adjudicated verdicts, §4.3.11):**

- **`P_cheap`** — precision of the cheap tiers: of the rows tiers 1+3 flagged, the fraction a
  human agrees are genuinely mislabelled.
- **`P_judge|cheap`** — of the rows tiers 1+3 flagged, the fraction on which the judge's verdict
  matches the human's.
- **`Yield_judge`** — of the rows the judge flags in the **S4 random audit stratum** (which the
  cheap tiers did *not* flag), the fraction a human agrees are genuinely mislabelled. This is the
  judge's *marginal recall*, and it is the number that decides the question.

**The decision rule:**

| Condition | Reading | Action |
|---|---|---|
| `P_cheap ≥ 0.80` **and** `Yield_judge < 0.10` | The cheap tiers are precise and the judge finds little they missed. Its verdicts are mostly re-deriving conclusions a human could reach from the flag alone | **Drop the LLM judge.** Route tier-1 and tier-3 flags straight to humans. The queue grows by the ~30 R5 rows the judge would have triaged, and the corpus gains zero model-provenance rows |
| `P_cheap < 0.50` | The cheap tiers are noisy; humans would spend most of their time rejecting false alarms | **Keep the judge as a triage filter.** This is its strongest case: it reduces human load rather than adding opinions |
| `Yield_judge ≥ 0.10` | The judge finds real errors the cheap tiers structurally cannot — most likely the taxonomy-ambiguity class of §4.3.0 point 4 | **Keep the judge**, scoped to S2/S3 plus the cheap-tier residual, as specified |
| Ambiguous / pilot too small to separate | | **Keep the judge but disable auto-apply**, so every verdict is advisory to a human and no model-provenance label is created |

**My prior, for what it is worth, and it is a prior not a finding:** on this fixture I expect
`P_cheap` to be *low* — [measured] dropping the flagged rows hurt macro-AP by 0.79pp, which is
consistent with the flags being hard-but-correct rows rather than errors. That argues for
*keeping* the judge as a triage filter here. On a real CatBoost-contaminated corpus I expect
`P_cheap` to be much higher and the judge's marginal yield to be concentrated entirely in the
ambiguous pairs. **Both readings are testable with 100 human decisions; make them before
committing to the judge.**

**What is not negotiable either way:** the taxonomy reports (§4.3.3 `common_multilabel_issues`,
and the §6.1 pair-confusion and co-occurrence tables) ship regardless. They cost nothing, they need no human, and they
answer a question — "is this taxonomy learnable?" — that no per-row instrument answers.

#### 4.3.13 The contamination fence — mechanics

**The rule (llm-fallback-policy §5): the LLM must never touch the gold set — not as annotator,
not as pre-annotator, not as a pre-filter on what gets annotated.** Here is exactly how that is
enforced.

**Pipeline ordering is the fence.** Phase 3 cannot start until the gold ID list exists:

```
Phase 1  (all rows)
   ↓
dedup clusters, language tags, quality flags      (all rows)
   ↓
Phase 2  (all rows — text metadata + rule mining; NO labels involved)
   ↓
GOLD SAMPLE DRAWN  — runbook §4.1, seeded hash, stratified, blind
   ↓
gold_ids.json  FROZEN, content-hashed
   ↓
Phase 3, ALL TIERS  input row set := all_rows \ gold_ids   ← hard assertion
   ├─ tier 1  conflicts
   ├─ tier 2  TF-IDF + grouped CV  ← trained on the non-gold pool only
   ├─ tier 3  cleanlab
   ├─ tier 4  (cut — §4.3.4)
   ├─ tier 5  LLM judge
   └─ tier 6  human queue
```

**The fence covers the whole cascade, not just the LLM.** llm-fallback-policy §5 names the LLM
because that is the document's subject, but the *principle* it protects — a model must not choose
what a human annotates — applies to any model, and cleanlab's ranking is exactly such a choice.
Extending the fence costs nothing: the cheap tiers have no reason to look at gold rows, and
excluding them removes a whole class of argument about whether tier 3 "counts".

Two narrow exceptions, both after the fact and neither feeding sampling:

- **Tier 1 may be run over the *adjudicated* gold set as a data-quality report to the annotation
  lead.** "Two byte-identical tickets received different labels from your annotators" is a
  legitimate finding about annotation quality, it involves no model, and it arrives after
  adjudication so it cannot influence what was annotated.
- **Tier 2 may be *evaluated* on the gold set** — that is just the classifier spec §3 baseline
  doing its normal job. It must be **trained** on the non-gold pool only, which is the same rule
  every other model in the project follows.

Five mechanical requirements:

1. **A hard assertion, not a filter.** The judge stage asserts
   `set(input_ids) ∩ set(gold_ids) == ∅` and **aborts the run** on violation. A filter that
   silently drops overlap hides the bug; an assertion surfaces it. The gold ID list is
   content-hashed and its hash is recorded in the snapshot manifest.
2. **Deny by default.** The judge stage takes a *complement* of the gold list, not an inclusion
   list, so a row added to the corpus after the fence was built is judged only if it is not gold —
   the same "filter on provenance, not on model name" principle llm-fallback-policy §5 applies to
   the training query.
3. **Phase 3 output never reaches the annotation tool.** Judge verdicts, evidence spans, and
   proposals are stored in a separate table from the annotation payload. Annotators for the gold
   set see title + description + priority only (classifier spec §2.2). The queue reviewers are a
   *different task with a different UI* and their output carries a different provenance value.
4. **Sampling independence.** No Phase-3 tier may influence *which* rows get annotated. The gold
   sample is drawn by seeded hash over strata that contain no Phase-3-derived field. Concretely:
   `dup_cluster_id`, `lang_primary`, `created_at` month, and cardinality are permitted strata;
   `judge_verdict`, `judge_disagreement`, **`cleanlab_flag`, `label_quality_score`,
   and `is_unactionable` (Phase-2-derived) are **not**.
   `dup_cluster_id` is permitted and `conflict_grade` is **not** — the first is a property of the
   text, the second is a property of the labels, and stratifying on the second would bias the
   test set toward contested rows.
5. **Tier-2 folds respect the fence.** The `GroupKFold` in §4.3.3 is computed over the non-gold
   pool. If gold rows were included, their out-of-sample probabilities would be produced by a
   model trained on the rest of the corpus, which is fine for evaluation but would then tempt
   someone to run cleanlab over them. Keep the pool clean and the temptation does not arise.

**Provenance markers.** Every row carries a structured `label_provenance`:

```json
{
  "label_source": "author_assigned | human_blind | human_after_judge |
                   llm_judge_confirmed | llm_judge_modified | catboost | transformer",
  "judge_touched": true,
  "judge_run_id": "...", "judge_model": "...", "judge_prompt_version": "...",
  "verdict_ids": ["..."], "changed_at": "...", "reviewer_id": "... | null",

  "cascade": {
    "conflict_grade": "identical | nested | partial | disjoint | null",
    "dup_cluster_id": "...",
    "cleanlab_flagged": false,
    "label_quality_score": 0.79,
    "selected_by": ["T1", "T3", "S2", "S3", "S4"],   // no T4 — the tier is cut
    "tier2_run_id": "...", "cleanlab_version": "2.9.0"
  }
}
```

`judge_touched` is set for **every row the judge scored, including rows it fully confirmed.**
Confirmation is not neutral: a confirmed row has passed through a model's opinion, and if we ever
discover the judge was systematically wrong about a service, we need to find every row it looked
at — not just the ones it changed.

**The `cascade` block is diagnostic metadata, not provenance.** A `cleanlab_flagged: true` row
whose label was never changed has exactly the provenance it started with — a routing signal is
not an opinion about the label, and treating it as one would disqualify rows for no reason. The
block exists so that (a) the queue can be re-derived without re-running the tiers, (b) a later
discovery that tier 3 was mis-configured can be traced to the rows it affected, and (c) the
§6.1 metrics can be sliced by which tier selected a row.

**A row whose label was changed on the judge's advice is model provenance and can never be
Case A. Confirmed, and the reasoning is not close.** Case A is defined
([classifier spec §2.1.1](./ticket-services-classifier.md), runbook §2.1) as *"a human
demonstrably edited the field"*. A judge edit is a model edit; treating it as Case A would put
LLM output into the trustworthy-label pool and reproduce, with a second model, exactly the
feedback loop proposal §3 exists to escape. Concretely:

- `llm_judge_modified` → **excluded from val/test unconditionally.** Usable in training at
  reduced weight, subject to the §4.3.8 ablation, and reported as its own slice.
- `human_after_judge` (a reviewer agreed with a judge proposal) → **also excluded from val/test.**
  This is the subtle one and it is worth being firm about: the reviewer saw the model's opinion,
  so this is anchored annotation, which classifier spec §2.2 and llm-fallback-policy §5 both
  reject. It is *better* than `llm_judge_modified` and gets a higher training weight, and it is
  still not gold.
- `llm_judge_confirmed` → the label is unchanged, so its provenance is whatever it was before,
  **plus** `judge_touched: true`. It gains no trust from having been confirmed.
- If the audit tables from proposal §5.4 ever exist, a judge-driven write must produce
  `actor_type = 'model'` with `model_kind = 'llm'`, exactly as llm-fallback-policy §5 specifies
  for production LLM output. **The training query filters on provenance, not on model name**, so
  a new model kind is excluded by default rather than by someone remembering.

---

## 5. Execution plan

### 5.1 What is held fixed for comparability

Classifier spec §5.3 already requires an identical data snapshot, splits, PII redaction, text
construction, thresholds, decode and metric code across arms. The pipeline adds four artifacts
that must be pinned identically across every ablation arm, or the comparison is meaningless:

`preprocess_version` · `blocklist_hash` · `detector_inventory_version` · `gold_ids_hash`, plus
`judge_prompt_version` and `judge_model` for arms that use tier 5, and **`tier2_run_id`,
`cleanlab_version` and the tier-3 selector definition** for arms that use the cascade. A cleanlab
minor-version bump changes which rows are flagged and therefore which rows a human saw; it is a
snapshot-invalidating change.

Every one of these goes in the snapshot manifest and in `model_card.md`.

### 5.2 The ablation ladder

Same 5-seed list throughout (10 seeds if the usable label count lands under 5,000 — classifier
spec §5.4). Reported as **mean ± sd**; a single-seed number is not a result.

| Rung | Arm | Compared to | Ship the rung if |
|---|---|---|---|
| 0 | B0 raw passthrough | — | (reference floor) |
| 1 | B1 Phase-1 only | B0 | any improvement; if none, investigate before proceeding |
| 2 | B3 Phase 1 + Phase-2-mined rules | B1 | **≥ 1.0pp mean val `R@P90` and non-overlapping seed intervals** (§4.2.4) |
| 3 | **B2 Phase 1 + cheap-tier label repair** (tier-1 conflicts applied, tier-1/tier-5-free masking, human decisions on the tier-1 queue) | B1 | ≥ 1.0pp, same test |
| 3b | B2 + proposals masked | B2 | ≥ 0.5pp; masking is nearly free, so a smaller bar is justified |
| 3c | **B2b = B2 + LLM judge** | **B2** | ≥ 1.0pp **over B2, not over B1**. This is the rung that decides §4.3.12 |
| 4 | B4 full | max(B2b, B3) | ≥ 1.0pp over the better single phase, else ship the single phase |

Rung 3c is the addition that the cascade forces and it is the one most likely to be skipped.
**Comparing the judge against B1 rather than against B2 would credit the judge with everything
the free tiers achieved** — a version of the same attribution error §4.2.4 avoids for Phase 2.

Diagnostic-only, never shippable: proposals applied as positives (§4.3.8 arm iv).

**The cheap pre-check that comes first.** Before any GPU touches this, run the §4.3.3 tier-2
model over each text variant and compare out-of-sample macro-AP. [measured] on this fixture the
B0→B1 text delta is **+0.33pp macro-AP / +0.62pp micro-F1**, obtained in under four minutes of
CPU. That is not a substitute for the encoder ladder — a linear model over character n-grams may
respond differently to preprocessing than a fine-tuned transformer — but a variant that makes the
linear model *worse* is worth investigating before spending GPU hours on it.

**All rungs are decided on validation.** The blind gold test set is scored **once per shipped
candidate** (classifier spec §6.4). An ablation ladder is a hyperparameter search over
preprocessing; running it on test would be exactly the error §6.4 forbids.

### 5.3 Cost and wall-clock inputs for the architecture document

I state the inputs; `dataset-pipeline-architecture.md` owns the cost model and the schedule.

| Stage | Unit cost | 5,013 rows | 50,000 rows |
|---|---|---|---|
| Phase 1 regex tier | **0.09 ms/row** [measured], single core | 0.5 s | 5 s |
| NER (if Tier A) | ≤ 40 ms/row [estimate], must be measured | ~3 min | ~33 min |
| MinHash, 2 signatures | **2.1 ms/row/signature** [measured] | 21 s | 3.5 min |
| Language ID | ~1 ms/row [estimate] | 5 s | 50 s |
| Phase 2 | 1 call/row, ~1,900 in / ~200 out, cacheable prefix | ~$19 [estimate] | ~$190 [estimate] |
| **Phase 0** — label profile (§4.0.1) | pure dataframe aggregation; **0.15 s** [measured] for the counts, split projection and per-quarter breakdown, plus ~5 s if the language tag is computed here rather than reused from §4.1.8 | **< 6 s** | < 60 s |
| **P3 tier 1** — conflict grading over dup clusters | reuses §4.1.7 clusters; O(n) grouping | **< 5 s** [measured] | ~1 min |
| **P3 tier 2** — TF-IDF + OvR LR, 5-fold `GroupKFold`, 20 services | **113 s** [measured], single machine, no GPU | **113 s** | ~25 min [estimate, superlinear in vocabulary] |
| **P3 tier 3** — cleanlab 2.9.0 over `(N, 20)` probabilities | seconds | **< 10 s** [measured] | < 2 min |
| ~~**P3 tier 4**~~ — embeddings + kNN + UMAP | **cut** (§4.3.4). Would have added `torch` (754 MB), `umap-learn` + a JIT toolchain, a 471 MB fp32 checkpoint, ~60 s of embedding wall-clock at 5k rows, and an embedding-matrix artifact class | **removed** | **removed** |
| **P3 tier 5** — LLM judge | **~430 first-pass calls** at the widened selector [estimate], ~2,100 in / ~150 out, cached prefix | **~$3–6** [estimate] | ~$30–60 [estimate] |
| **P3 tier 6** — human queue | 2.5 min/item | **~12 h** on ~285 rows [estimate] | scales with flag rate, not rows |

**The whole label-error cascade below the LLM costs about two minutes of CPU and zero dollars.**
That is the fact that should govern the design review. The deterministic phases are free, the LLM
phases now cost less than a lunch, and **no part of this pipeline is cost-constrained** — so every
decision in it should be made on correctness, contamination risk and human time, not on budget.
Any argument that starts with "to save on LLM calls" should be treated with suspicion; the
argument for the cascade is grounding and contamination, and the 10–16× call reduction is a
side effect.

---

## 6. Evaluation

### 6.1 Metrics that must be reported per snapshot

Emitted as `pipeline_report.json` and rendered as a table in the model card. No snapshot is
usable without it.

**Phase 0 (§4.0.1)** — the label profile, in full, as the *first* section of the report. It is
reproduced rather than referenced because it is the section a stakeholder reads: the one-line
summary, Tables 1–5, the T-tier assignment per service, and the diff against the previous
snapshot. §6.1's later class-balance figures are computed on the **assembled** corpus and must be
reported **beside** the Phase-0 input figures, never instead of them — the delta between the two
is exactly how much of the tail the pipeline itself removed, and a phase that quietly costs
`cdn` 8 of its 38 positives is something the build must surface.

**Phase 1**

- rows in / out / quarantined, by reason code;
- chars and tokens before/after, per deletion class, with per-rule attribution ("the boilerplate
  blocklist removed X tokens across Y rows");
- token-length distribution before/after (mean, p50, p90, p95, p99, max, share > `max_len`);
- placeholder counts by class, absolute and per 1,000 rows;
- **redaction recall and precision against the canary set** (§6.4) — the headline safety number;
- near-duplicate clusters at 0.80 and 0.70, both signatures, with size histogram;
- language distribution and the three-way LID disagreement rate;
- blocklist size and hash; detector inventory version.

**Phase 2**

- call success / schema-invalid / refused / timeout counts;
- `phase2_invalid_span` rate (quotes not verbatim);
- rules proposed, rules approved by a human, rules rejected — and **the approval rate**, which is
  the phase's real quality signal;
- residual-PII findings by kind, and how many became new detectors or canaries.

**Phase 3 — cascade tiers 1–3**

- **tier 1:** conflicting groups and rows by grade (`identical`/`nested`/`partial`/`disjoint`) ×
  grouping method × threshold; rows queued; rows masked. A rising `disjoint` count between
  snapshots is an annotation-quality alarm;
- **tier 2:** out-of-sample macro-AP, micro-AP, micro-F1, macro-F1 under `GroupKFold`, **plus
  ΔA (raw vs Phase-1 text) and ΔB (with vs without flagged rows in training)** as defined in
  §4.3.3, and the grouped-vs-ungrouped gap as a leakage diagnostic;
- **tier 3:** flagged rows by selector (report **both** the `< 0.20` and `< 0.30` cuts, so the
  §4.3.11 widening decision can be revisited per snapshot); per-service flag counts against
  per-service positive counts; `common_multilabel_issues` with both directions;
  `overall_multilabel_health_score`; overlap with tier 1.

**Phase 3 — the two taxonomy reports (required, and they cost nothing)**

These replace the UMAP deliverable that was cut with tier 4 (§4.3.4). Both are computed from
artifacts that already exist — tier 2's out-of-fold probability matrix and the label column — so
they add no dependency, no stage and no wall-clock. **They are the deliverable for the product
owner**, and their value is that they are diffable between snapshots, which a scatter plot is not.

1. **Per-service-pair confusion, both directions.** For every ordered pair `(a, b)` of the 190
   unordered service pairs, report `P(b | a-only)` — the rate at which the tier-2 model asserts
   `b` above 0.5 on rows that carry `a` and not `b` — and `P(a | b-only)` separately.
   **The asymmetry is the finding**, and averaging the two directions destroys it: a pair that is
   confused in one direction only is a guideline gap, while a pair confused in both is genuine
   definitional overlap. [measured, validation §3.9] on this fixture the top rows are
   `monitoring`→`notifications` **0.223** (against 0.019 the other way),
   `cdn`→`api-gateway` 0.184 / 0.000, `cdn`↔`console-ui` 0.143 / 0.000,
   `compute`↔`networking` 0.020 / 0.095, `access-control`↔`console-ui` 0.054 / 0.035,
   `billing`↔`subscriptions` 0.014 / 0.056, `access-control`↔`auth` 0.026 / 0.040 — which recovers
   the fixture README's deliberately-planted ambiguities, and surfaces one
   (`monitoring`→`notifications`) that the README does not list.
2. **Label co-occurrence matrix**, raw counts and normalised by the smaller class's base rate.
   Report it beside the confusion table: **confusion conditioned on a base rate is interpretable,
   confusion alone is not.** `billing`↔`subscriptions` co-occurs 273 times and is barely confused;
   `cdn`↔`api-gateway` never co-occurs and is confused 18% of the time in one direction. Those are
   opposite findings and only the pair of tables distinguishes them.

Report both **per snapshot**, with a diff against the previous snapshot. A pair whose confusion
rate moves materially between snapshots is either a taxonomy change nobody recorded or a training
distribution shift, and both are worth a question.

**Phase 3 — tiers 5 and 6**

- rows judged, by selecting stratum (S1/S2/S3/S4), and the overlap matrix between strata;
- verdicts issued; accept / reject / ungrounded counts; **calls made** (first-pass + escalated);
- **reject rate per service** (the §4.3.10 R6 gate);
- escalation rate; 3-of-3 / 2-of-3 / 1-of-3 distribution;
- proposals by service; masked vs queued vs ignored;
- queue size by routing rule, and the overlap matrix between rules;
- **human agreement with the judge on the R7 2% audit sample** — this is the number that says
  whether the auto-apply path was safe, and it also yields `Yield_judge` for §4.3.12;
- post-adjudication co-occurrence rates for the three ambiguous pairs, versus the pre-adjudication
  rates (`access-control`/`auth` 17.5%, `logging`/`monitoring` 4.2%, `console-ui` co-label 637)
  [measured].

#### 6.1.1 Agreement statistics: what to use, and the wrong answer to avoid

Several of the numbers above are **agreement between two parties on a set-valued label** —
judge vs stored, judge vs human, reviewer vs reviewer. The instinct is Cohen's κ. **Cohen's κ is
not defined for this target and must not be used.** It assumes each item receives exactly one of
K mutually exclusive categories; here each item receives a *subset* of 20 services with
`|S| ∈ 1..4`.

**The two ways people reach for it anyway, and why both produce a meaningless number:**

- **Flatten to `(row, service)` binary cells and call `sklearn.metrics.cohen_kappa_score`.**
  [measured] the label matrix is 8,771 positives in 4,622 × 20 = 92,440 cells — **90.51% of cells
  are agreed negatives.** The statistic is then dominated by the two parties agreeing that
  `terraform-provider` does not apply, and the chance-correction term is computed against a
  marginal that averages 20 wildly different base rates (851 positives for `billing`, 15 for
  `terraform-provider`). The number will look reassuringly high and will move for reasons that
  have nothing to do with the disagreements you care about.
- **Treat the whole label set as one category.** That is up to 2²⁰ categories, most observed once,
  so expected agreement `p_e → 0` and `κ → raw agreement`. The chance correction — the entire
  reason to use κ — does nothing.

**What to report instead, in this order:**

1. **Krippendorff's α with the MASI distance** — the standard instrument for set-valued
   annotation ([Passonneau 2006](https://www.researchgate.net/publication/228528415_Measuring_agreement_on_set-valued_items_MASI_for_semantic_and_pragmatic_annotation)),
   and **already the project's chosen metric** for annotator agreement in runbook §4.2 and
   classifier spec §2.2. Using it here means one agreement statistic across the whole project,
   comparable between the judge and the humans. MASI weights partial set overlap and penalises
   the nested case less than the disjoint case — which is exactly the distinction §4.3.2 makes.
2. **Per-service Cohen's κ as a vector, never averaged into one number.** *Within* one service
   the decision genuinely is binary and mutually exclusive, so κ is well-defined. Report all 20
   values with `n_positives` beside each, plus the minimum. This is the view that locates *which*
   services the disagreement lives in — and [measured] we already expect that to be `console-ui`
   and `networking`.
3. **Set-level descriptive companions:** exact-set-match rate, mean Jaccard, and mean MASI
   similarity. These are uncorrected for chance, so report them **alongside** α, never instead of
   it.

**Gate wording follows from this:** the α ≥ 0.67 threshold in classifier spec §2.2 applies to
*annotator* agreement on the gold set and is **not** transferable to judge-vs-stored agreement,
which measures something different (a model's agreement with a possibly-contaminated column).
Report judge-vs-stored α as a descriptive number with no gate attached, and gate only on the
human-adjudicated quantities in §6.2.

### 6.2 Quality gates, thresholds, and what happens when one fails

**A failed gate stops the snapshot.** It does not produce a warning that someone reads later. The
snapshot is not written, and the run is reported.

| # | Gate | Threshold | On failure |
|---|---|---|---|
| G1 | Structural parse failures | ≤ 0.1% of rows | Stop. The export is malformed; fix the export, not the parser |
| G2 | Canary redaction recall, structured PII classes (email, phone, card, ИНН/ОГРН, bank account, БИК, СНИЛС, passport, transaction reference) | **≥ 0.99** | Stop. Add the missing pattern, re-run, add the miss to the permanent canary set |
| G3 | Canary redaction recall, NER classes (person, org) | **≥ 0.90** | Stop below 0.90; between 0.90 and 0.95, ship with the gap recorded in the model card and a follow-up ticket |
| G4 | Over-redaction of protected technical strings | **≤ 0.5%** of protected tokens | Stop. This is the §2.3 failure mode (45% of person-shaped matches are error strings) and it silently destroys the label signal |
| G5 | Total tokens removed by the boilerplate blocklist | ≤ 60% of corpus tokens | Stop and review the blocklist. [measured] the ≥20× cut removes 48.2% of sentence instances; anything approaching 60% means the cut is eating content |
| G6 | Rows quarantined | ≤ 5% | Investigate before proceeding; a quarantine spike is usually one broken rule |
| G7 | Near-dup rows at Jaccard 0.80 | ≤ 10% of corpus | [measured] 2.8% here. Above 10%, the effective sample size is far below the row count and the confidence intervals in classifier spec §6.4 are wrong |
| G8 | Out-of-taxonomy service tokens | **0** | Stop. Either a typo, a retired service, or a taxonomy change requiring a mapping table (classifier spec §4.8). Never auto-drop |
| G9 | Phase-2 schema-invalid rate | ≤ 5% | Fix the prompt or the schema. Above 20%, the model tier is wrong |
| G10 | `judge_ungrounded` rate | ≤ 2% | Fix the prompt. This is a grounding failure, not a data finding |
| G11 | Per-service judge reject rate over the judged pool | ≤ 15% [estimate, calibrate on the pilot] | Hold that service's entire reject list for human sign-off (R6) |
| G12 | Human agreement with auto-applied judge verdicts (R7 sample) | ≥ 0.90 | Below 0.90, **revert all auto-applied rejects** for that run and route everything to the queue. The auto-apply path is a convenience, not a requirement |
| G13 | Gold-set contamination assertion, **all Phase-3 tiers** | **exactly 0 overlap** | Abort the run. Non-negotiable |
| **G14** | **Tier-2 CV is grouped** — assert that no `dup_cluster_id` appears in more than one fold | **0 violations** | Abort tier 2. Ungrouped folds inflate confidence on duplicated rows and hide the errors tier 3 exists to find; [measured] the leakage is +0.91pp micro-F1 |
| **G15** | Tier-3 flag rate | **≤ 15% of labelled rows** | Above that, either `pred_probs` are broken (check the tier-2 metrics first) or the corpus has a systemic labelling problem that a queue cannot absorb. Stop and diagnose; do not size a queue from it |
| **G16** | Tier-3 flags on services with < 50 positives | **0 auto-applied** | Route to a human (R2). Eroding a rare service below evaluability is unrecoverable and invisible to every aggregate metric |
| **G17** | ΔB (label-noise delta, §4.3.3) | **report, do not gate** | [measured] it is **−0.79pp macro-AP** on this fixture, i.e. removing flagged rows *hurts*. A negative ΔB is a finding, not a failure — it says the flags are hard-but-correct rows and that auto-dropping must stay off (§4.3.1) |
| **G18** | **Admission gate for any new row-selection tier** — including any future re-proposal of an embedding tier (§4.3.4) | On a labelled error set (injected or human-adjudicated): **`tier-3 top-N ∪ new-tier top-M` must beat `tier-3 top-(N+M)` on error recall, mean over ≥ 3 seeds, with non-overlapping spreads.** Report the ablation control too: dropping the new tier's *unique* flags from training must cost materially more macro-AP than dropping the same number of random rows | Do not admit the tier. [measured] the embedding kNN fails this by **5.5–9.4pp** and its ablation is indistinguishable from random (−0.42pp vs −0.28pp control). **"Corroborating only" does not exempt a tier from this gate** — it is the same gate with the cost moved into reviewer sort order on a capped queue |
| **G19** | **Phase 0: a service in `labels.json` with zero positives in the export** | **exactly 0 such services** | **Hard stop before Phase 1 runs.** The service occupies an index in the label space and cannot be learned, so every macro metric silently averages in a class that can only score 0. It is not a modelling problem: either the export is incomplete (wrong window, wrong filter, a join that dropped rows) or the taxonomy contains a service the product does not use. Requires a human answer — a mapping-table entry (classifier spec §4.8) or a corrected export — before the build continues. [measured] 0 on this fixture |
| **G20** | **Phase 0: projected test positives per service** (§4.0.1.2) | **warn and report** at < 30; **the report must name every service below the line in its one-line summary** | Do not stop the build — a thin tail is a legitimate corpus property, not a defect (§4.0.1.5). But the services below 30 must be marked T-1, excluded from macro aggregates (with macro reported both ways), and given "insufficient data" rather than a precision figure. **Silently reporting a precision computed on 1 positive is the failure this gate exists to prevent** — [computed] its 95% interval is ±48.8pp. [measured] 5 of 20 services fail this on the fixture |
| **G21** | **Phase 0: prevalence drift against the previous snapshot** | **warn** at any service whose prevalence moves by ≥ 3×, or any T-tier transition | Investigate before training. Three causes and all matter: a product change (legitimate, retrain), a taxonomy change nobody recorded (invalidates historical labels — classifier spec §4.8 requires a mapping table), or a broken export (stop). Not a stop by itself, because the first cause is normal; a stop if the same service also fails G19 |

### 6.3 Egress gate on the finished corpus

Run once per snapshot, out of band, before the corpus is used for anything or moved anywhere.

**Say plainly what this gate is: a control against leaking personal data, not credentials.** The
credential scanners (`trufflehog`, `gitleaks`) that the first revision of this document put here
are removed along with the rest of the secrets tier (§2.3) — they detect developer credentials,
which the product owner states are not present, and they detect no Russian PII at all. Keeping
them would be dead scaffolding that can only ever fire a false positive.

What the gate actually is:

- **Re-run the full §4.1.4 detector inventory over the *output* corpus and assert zero
  detections.** This is a self-check, and its value is specific and limited: it cannot find PII
  classes we do not detect, but it **does** catch the failure mode that actually happens — a row
  that bypassed redaction because of an ordering bug, a protected-span sentinel that was never
  restored, a partition written before the detector version was bumped. That class of bug is
  silent in every other gate.
- **Grep for the raw value of every canary** injected in §6.4. Any survivor is a G2/G3 failure the
  recall computation somehow missed, which is itself a bug worth finding.
- **Assert that no quarantined row and no `is_gold` row leaked into the training partition.**
- **Assert the corpus contains no free-text column other than the Tier-A `model_input_text`** —
  in particular, that the raw `description` is not carried into an artifact that leaves the
  perimeter. Whether raw text is retained at all, and where, is the architect's and the DPO's
  decision (P9); the pipeline's job is to make the answer checkable.

### 6.4 Measuring redaction recall without a labelled PII set

**The problem the brief names is real: we have no ground-truth PII annotations, and building
them would cost more than the pipeline.** The answer is seeded canaries — inject PII we control,
at offsets we recorded, and measure how much comes back out. The idea is borrowed from
memorisation research, where deliberately-inserted canaries measure what a system retains
([Carlini et al., "The Secret Sharer", USENIX Security 2019](https://arxiv.org/abs/1802.08232));
here we use it to measure what a redactor removes.

**Protocol.**

1. **Build a canary generator** producing K = 500 instances across every class in the §4.1.4
   inventory, and deliberately covering the surface variation that breaks naive detectors:
   - RU phones in ≥6 formats (`+7 900 000-16-63`, `+7(900)0001663`, `8 900 000 16 63`,
     `7-900-000-16-63`, `+7 900 000 95 68`, `тел. 9000001663`);
   - cards spaced, unspaced, hyphenated, and **partially masked** (`4276 38** **** 5678`) —
     [measured] a plain `\d{13,19}` regex finds **0** of this corpus's 21 card-like rows;
   - ИНН 10- and 12-digit, with and without the `ИНН` keyword, valid and invalid check digits;
   - **RU person names in nominative, genitive, dative and instrumental case** — inflection is
     the single most likely NER failure mode and it must be in the canary set;
   - Latin names, including ones shaped like error strings, to probe over-redaction;
   - **bank and transactional details** — 20-digit settlement accounts (р/с, к/с) with and
     without spacing, БИК, СНИЛС, payment and order references, invoice numbers — **none of which
     occur in this fixture** (§4.1.12), so canaries are the *only* test those detectors will ever
     get, and they are the classes the product owner named as material;
   - паспорт in both `45 03 123456` and `4503 123456` forms, with and without the keyword.

   **No application-secret canaries.** Per §2.3 there are no secrets in production ticket text, so
   a secret canary would measure a detector we do not ship against a risk we do not have.
2. **Inject** into a copy of the real corpus at recorded offsets, half inside prose and half
   inside evidence blocks (detectors behave differently in each, and the §4.1.3 protected-span
   masking makes that difference large).
3. **Run the pipeline.** `recall = fraction of injected spans replaced by the correct
   placeholder`. A span replaced by the *wrong* placeholder counts as recalled for G2/G3 and is
   reported separately as a misclassification.
4. **Over-redaction precision** is measured on a different set: a fixed list of ~300 **protected
   technical strings** (HTTP reason phrases, error class names, SQL plan nodes, service names,
   resource id shapes) embedded in carrier sentences. `over_redaction = fraction of protected
   tokens that got replaced`. Gate G4 ≤ 0.5%. [measured] this is not hypothetical: 45% of
   English person-name-shaped matches in this corpus are technical strings.
5. **The canary set is versioned, append-only, and grows.** Every real miss found by a human
   (§6.5), by the egress self-check (§6.3), or by Phase 2's `residual_pii` becomes a permanent
   canary. This is what stops the same class of miss from recurring.

**The honest caveat, which must appear in the model card verbatim.** Canary recall measures the
detectors against *the distribution we invented*, which is the distribution we wrote the
detectors for. **It is an upper bound and it cannot find unknown unknowns.** It is a regression
test, not an assurance. The complements are the human sample (§6.5), the egress self-check
(§6.3), Phase 2's independent second opinion (§4.2.3), and the standing adversarial exercise
(§4.1.13). Anyone quoting "0.99 redaction recall" without that sentence attached is
misrepresenting it.

### 6.5 Human validation of the pipeline itself

**Sample sizes are set by the rule of three:** with *n* observations and zero defects, the upper
95% bound on the defect rate is ~3/*n*. n = 200 gives a 1.5% bound, which is the resolution
needed to defend a "≤ 2% defect" claim. Below n = 100 the bound exceeds 3% and the exercise
cannot support any gate.

| Phase | n | Sampling | Reviewer checks | Gate |
|---|---|---|---|---|
| **Phase 1** | **200** | 50 uniform random · 50 from the placeholder-touched stratum (922 rows) · 50 from the multiline/evidence stratum (374 rows) · 50 from rows where the boilerplate blocklist or template footer fired (99 alert rows + blocklist hits) | (a) is any PII left? (b) was any label-bearing span destroyed? (c) is the text still readable and grammatical? (d) does the placeholder match what it replaced? | ≥ 2 defects of type (a) or (b) → fix and re-sample |
| **Phase 2** | **150** | 50 uniform · 50 with the largest fraction of sentences marked filler · 50 where the validator rejected the response | (a) was anything invented? (b) was a signal-bearing sentence marked `pleasantry`? (c) is `is_unactionable` right? | ≥ 3 defects of type (b) → the mined blocklist diff is rejected wholesale, not row by row |
| **Phase 3, tiers 1+3** | **100 rows** | 30 tier-1 conflicts (all grades) · 40 tier-3 flags sampled across the quality-score range · 30 from the S4 random-audit stratum that the tiers did **not** flag | (a) is the row genuinely mislabelled? (b) if yes, is it noise or taxonomy ambiguity? (c) for tier-1 groups, is the grade right? | This sample yields **`P_cheap`** and **`Yield_judge`** for the §4.3.12 decision. `P_cheap < 0.30` → stop and fix tier 2 before spending any human time on the queue |
| **Phase 3, tier 5** | **100 verdicts** | 40 rejects · 30 accepts · 30 from the 2-of-3 disagreement pool | (a) is the verdict right? (b) does the evidence span support it? (c) for the ambiguous pairs, is the decision rule being applied as written? | judge precision on rejects < 0.85 → auto-apply disabled, everything routes to the queue |
| **Blocklist review** | **all** | the full blocklist, once per version | is any entry signal-bearing? | any signal-bearing entry → remove it and add it to the protected-span list |

**On statistical power, stated plainly:** n = 40 gives a proportion to roughly ±10pp at 95%. That
is enough for a go/no-go on the auto-apply path and on §4.3.12, and **not** enough for a
per-service number. Do not publish per-service judge precision or per-service tier-3 precision
from these samples; if a per-service number is needed, it needs its own sample per service, and at
that point the annotation budget conversation from §4.3.11 reopens.

**The 30 unflagged rows in the tiers-1+3 sample are the most important 30 items in this table.**
They are the only estimate of the cascade's **recall** — how many genuinely mislabelled rows the
cheap tiers missed entirely. Everything else in §6 measures precision, and a precision-only
evaluation of an error-detection system is exactly how you convince yourself a corpus is clean
when it is not.

---

## 7. Risks and failure modes

### 7.1 What this pipeline does not fix, and what a dataset built from this file is not good for

This section is the one most likely to be skipped and most likely to matter.

**Labels are author-assigned. There is no `ticket_messages`. Cases A/B/C/D cannot be
reconstructed from this CSV.** That has consequences that no amount of text processing touches:

- **The provenance partition does not exist here.** Every row's `label_provenance.label_source`
  is `author_assigned`. The Case A/B/C/D machinery in runbook §2 and classifier spec §2.1.1
  cannot be exercised, the correction rate that sizes the entire project (runbook §5.2) cannot be
  computed, and **the "label provenance" evaluation slice in classifier spec §6.3 is
  unavailable.** Any pipeline code that reads a provenance column will run and will be reading a
  constant.
- **There is no inter-annotator agreement, therefore no Krippendorff α, therefore the α ≥ 0.67
  taxonomy gate cannot be evaluated.** [measured] 59.1% of labelled rows touch an
  ambiguous-family service, so this is the gate most likely to fire on real data and it is
  precisely the one this file cannot test.
- **This file cannot stand in for the gold set** (2,000 test + 1,000–1,500 val, runbook §4.2).
  Any accuracy number measured on it describes the generator, not the world.
- **Phase 3 does not repair this, and the cascade does not either.** The judge is another model.
  A corpus whose labels were author-assigned and then judge-adjudicated has *two* non-human
  provenance classes, not zero. The cheap tiers add no provenance class, but they add no ground
  truth either — cleanlab tells you a row is *surprising*, not that it is *wrong*, and
  [measured] on this fixture the two are not the same thing: dropping the flagged rows cost
  0.79pp of macro-AP.
- **The cascade's fixture yields are not transferable.** [measured] tier 1 finds ~14 reviewable
  rows and tier 3 flags 2.06% of rows, *because* the generator assigned labels from a scenario
  bank and 564 of 565 same-title groups agree exactly. On a real CatBoost-contaminated column
  both numbers should be several times larger. **Every queue and person-hour figure in §4.3.11
  must be re-derived on the real export before any budget is committed** — it costs four minutes
  of CPU (P10).
- **Phase 0 confirms a known-bad number here; it does not discover one.** The fixture's imbalance
  is **deliberate**: per [`data/raw/README.md`](../../data/raw/README.md), "the rare tail is
  deliberately NOT scaled proportionally … Instead: terraform-provider 15, message-queue 31, cdn
  38, dns 41 — four services below 50 absolute positives … held down by weighting, not by lack of
  scenarios (7–19 distinct scenarios each)." The tail was constructed so that "services that
  cannot be learned or evaluated reliably" is a live property to design against. **So the §4.0.1
  report reproduces a designed-in result, and it must not be presented as a finding.** Its real
  value is the *first run against a production export*, where the numbers are unknown and the
  scoping decision is live. What the fixture run does prove is that the report, the tiers and the
  gates behave correctly on a corpus whose answer is known in advance — which is the right thing
  to test them on.
- **Dedup thresholds tuned here will be wrong on real data.** [measured] the corpus is
  compositional: 149 sentence types cover 48.2% of sentence instances, and TTR is 0.047–0.054.
  Lexical statistics are systematically lower than a real corpus of this size. The Jaccard 0.80
  choice must be re-validated on real data against a human-labelled duplicate sample.
- **The 391 empty-`services` rows are ambiguous between "untriaged" and "genuinely
  unactionable"** and [measured] cannot be separated by length (median 57 vs 62 tokens). The
  pipeline can flag them; it cannot resolve them.
- **Whole detector classes are untested.** Zero bank accounts, БИК, СНИЛС, passport numbers,
  transaction references, UUIDs, or real tracebacks appear in this file — and bank/transactional
  details are exactly the class the product owner named as material. A green Phase-1 gate here
  says nothing about them; only the canary set (§6.4) exercises them at all.
- **The fixture over-states one risk class and under-states another.** It carries application
  secrets that production does not (§2.4) and carries none of the bank details that production
  does. Read both directions before generalising any §6 number.

**What this file *is* good for, and it is genuinely a lot:** exercising every stage end to end at
realistic volume; measuring throughput and cost; building and human-reviewing the blocklist,
protected-span list and detector inventory; validating the two-signature dedup design and the
temporal-split leakage measurement; and producing meaningful canary-based redaction metrics —
meaningful precisely because we inject the canaries ourselves, so they do not depend on the
corpus being real.

### 7.2 Risk table

| # | Risk | How it manifests | Detection |
|---|---|---|---|
| 1 | **Train/serve skew** — a Tier-A rule that cannot run at inference | Offline metrics fine, shadow-mode precision collapses. Invisible in every offline number | The tier classification is enforced in code review; the shadow gate (classifier spec §6.7) is the backstop. **Add a CI test that runs the full Tier-A path on a single ticket with no corpus statistics and no network** |
| 2 | **Signal destruction by over-redaction** | Precision drops on RU/EN slices with pasted evidence; error analysis shows the model never learns error-string features | G4, the §6.5 Phase-1 human sample, the B0-vs-B1 ablation (if B1 loses to raw passthrough, the pipeline is eating signal) |
| 3 | **Residual personal data in the corpus** | Nothing, until it is in a model artifact, a log, or a snapshot that moves | G2/G3, §6.3 egress self-check, Phase-2 `residual_pii`, the adversarial exercise. **Assume this is nonzero and design storage accordingly** — this is a requirement for `system-architect`, not a hope |
| 4 | **Blocklist eats signal** | A whole class of tickets loses its distinguishing sentence; one service's recall collapses | G5, the full blocklist review, `carries_service_evidence` from Phase 2, per-service recall in the ablation |
| 5 | **Judge distillation** — the corpus starts to encode the judge's opinions | Model agrees suspiciously well with the judge and no better with humans | `judge_touched` slice in every metrics report; the R7 audit sample; the ablation arms |
| 6 | **Anchoring in the human queue** | Reviewers rubber-stamp the judge; agreement looks great and means nothing | Interleave **20 blind items** (no verdict shown) into the queue and compare agreement. A large gap proves anchoring — the same A/B classifier spec §7 risk 8 specifies for the gold set |
| 7 | **Dedup collapses the corpus** | Effective sample size far below row count; confidence intervals wrong | G7; report cluster-size histograms; **never lower the threshold to chase semantic duplicates** ([measured] 0.50 collapses 34.5% of rows) |
| 8 | **Leakage via near-duplicates across the split boundary** | Random-vs-temporal validation gap; test precision far above shadow precision | [measured] 9 clusters / 120 rows straddle at 0.80 — the splitter must drop the later copy and report the count |
| 9 | **Language slice built on a broken rule** | English or code-switched degradation invisible; classifier spec §6.3's RU/EN gate measures the wrong thing | Three-way LID agreement; the code-switch detector's count against the README's 237; §4.1.8's operational definition |
| 10 | **Preprocessing version drift** | A model trained on snapshot v3 served with preprocessing v4; silent quality loss | `preprocess_version` in the model artifact **and** in the serving artifact, with a startup assertion that they match. Requirement for `system-architect` |
| 11 | **Judge bias on the ambiguous pairs** | Systematic stripping of `console-ui` (9.1% of all positives) or hedging toward accepting both members of a pair | G11 per-service reject rate; post-adjudication co-occurrence rates vs the measured baselines (17.5% / 4.2% / 637) |
| 12 | **Rare-tail erosion** | `terraform-provider` (15 positives) drops below evaluability; no aggregate metric notices | R2 routes every rare-tail reject to a human; per-service positive counts reported before and after every phase |
| 13 | **Phase 2 quietly becomes a rewrite** | Someone "simplifies" the span-selection design into text generation; risk 1 follows | The tier table in §4.0 is a contract. **CI test: assert the Tier-A output is byte-identical whether or not Phase 2 ran** |
| 14 | **Tier-2 CV folds stop respecting duplicate clusters** | A refactor swaps `GroupKFold` for `KFold`; duplicated rows get inflated confidence; tier 3's flag rate silently collapses and the corpus looks clean | **G14 asserts it directly.** [measured] the leakage signature is micro-F1 rising ~0.9pp while macro-AP falls — an odd combination that is itself diagnostic |
| 15 | **Cleanlab flags treated as ground truth** | Someone auto-drops or auto-relabels the flagged rows; rare services erode; macro metrics fall while micro metrics hold | §4.3.1 forbids it; G16 blocks it for rare services; **G17 reports ΔB, and [measured] ΔB is negative on this fixture** — the evidence that the failure is real, not theoretical |
| 16 | **The cascade's precision is measured and its recall is not** | Every gate is green, the queue is small, and the corpus is still full of errors nobody looked for | The 30 unflagged rows in the §6.5 tiers-1+3 sample, and the S4 random-audit stratum (R7). Both exist solely to estimate recall; **neither may be dropped for budget** |
| 17 | **Model/library version drift in the cascade** | A cleanlab or scikit-learn upgrade changes which rows were flagged, so a snapshot cannot be reproduced and a past human review cannot be re-derived | `cleanlab_version` and `tier2_run_id` in the snapshot manifest (§5.1); pinned versions (§9.3) |
| 18 | **A selection instrument enters on plausibility rather than on measurement** | It consumes reviewer slots or sort position, looks reasonable in review, and quietly lowers the recall per reviewer-hour. This already happened once: the embedding tier was specified with thresholds derived from a proxy that [measured] shares only 34% of its list with the real instrument | **Gate G18** — matched-budget recall against tier 3 plus the random-drop ablation control, before admission. The precedent and the full working are in [`knn-tier-validation.md`](./knn-tier-validation.md) |
| 19 | **A requirement disappears with the stage that incidentally provided it** | The OOD detector for llm-fallback-policy §2 case 4 was riding on tier 4's embedding distances; cutting the tier silently removes it and nobody notices until the fallback path misbehaves | **P13 tracks it as an explicit hand-off to the classifier spec, not a deletion.** The general rule: when a stage is cut, enumerate what else was consuming its outputs before the deletion lands |
| 20 | **Unmeasured imbalance** — the model never predicts the tail and micro metrics look healthy | [measured] the 4 sub-50 services carry 1.43% of positives, so never predicting any of them still leaves a **98.57% micro-recall ceiling against an 80.0% macro ceiling**. Micro-F1 would read fine while 20% of the taxonomy is dead | The Phase-0 report (§4.0.1) makes the size of the blind spot known *before* training; macro-AP as the selection metric (classifier spec §6.2); per-service prediction counts; G20 |
| 21 | **Someone resamples to "fix" the imbalance** | Rare-service prevalence rises as intended and the co-occurring services' prevalence rises with it; the joint label distribution is quietly distorted and calibration degrades where it matters | §4.0.1.4 forbids it with the arithmetic: [measured] oversampling `cdn` 10× adds 342 target positives and **396 collateral**. Detection: compare the label marginal and co-occurrence matrix of the *training sample* against the Phase-0 profile — any divergence that is not an explicit weighting decision is this bug |

---

## 8. Open questions

| # | Question | Owner | Why it blocks |
|---|---|---|---|
| **P1** | **May pseudonymised ticket text leave the production perimeter — to a hosted model provider, or to a training environment?** | Legal / DPO (classifier spec Q7) | Decides self-hosted vs hosted for Phases 2 and 3. §4.2.6 defaults to self-hosted so no work is blocked on the answer, but the quality ceiling differs. **This is a governance decision, not an engineering one; the spec states the requirement and does not attempt the reasoning** |
| **P2** | **Is there a self-hosted LLM available inside the perimeter, and of what tier?** | Infra / `system-architect` | If not, and P1 says no, Phases 2 and 3 cannot run at all and the pipeline is Phase 1 only — which §3's B1 baseline says may be sufficient anyway |
| **P3** | **Who authors the decision rules for `auth`/`access-control`, `monitoring`/`logging`, `console-ui`?** | Product owner | §4.3.7. These are taxonomy decisions, not ML decisions. Without them the judge is guessing and so are the annotators |
| **P4** | **Can ~28 person-hours be allocated for the queue and validation, *in addition to* the ~70 h gold budget?** (down from 46 h by the cascade; ~25 h if P14 keeps the narrow selector — §4.3.11) | Support lead (classifier spec Q9) | §4.3.11. If the answer is no, tier 5 and the queue are dropped and the gold set is protected; **tiers 1–3 still run, since they need no human at all** and still produce the §6.1 taxonomy reports. **This must not be resolved by taking hours from the gold set** |
| **P5** | **What is the NER latency on real tickets?** | Whoever runs the benchmark, week 1 | §4.1.5 gate. Decides whether name redaction is Tier A (runs at inference) or Tier B (training corpus only, with documented skew) |
| **P6** | **Confirm the premise: does production `title`/`description` really contain no application secrets, API keys or program keys?** Who checked, over what window, and how? | Product owner + security | §2.3 takes this as authoritative and removes the entire credential-detection tier on the strength of it. If it is an assumption rather than a measurement, the cheap check is to run a rule pack over one month of real tickets **once**, and record the result. Reinstating the tier later is a preprocessing version bump and a re-run, not a redesign |
| **P6b** | **Is residual personal data in the training corpus an acceptable risk, and what is the containment plan?** | Security / DPO | §4.1.13 says the FN rate is nonzero and cannot be proven zero. The honest ask is for a containment requirement (encryption at rest, access control, retention), which is `system-architect`'s to design |
| **P7** | **Does `preprocess.json` have a size budget?** Classifier spec §9.1 says < 50 KB; [measured] the ≥20× blocklist alone is 149 entries ≈ 9 KB here, and a real corpus's could be far larger | `system-architect` | If the budget holds, ship the blocklist as a separate hash-referenced artifact. Either way it must be versioned with the model |
| **P8** | **Is the judge allowed to see `labels` and `flags`?** | Product owner + ML | They are excluded from the *model* input (classifier spec §2.7), but they are legitimate context for a *judge*. My recommendation is **no** — a judge with more context than the annotator produces verdicts the annotator cannot reproduce, and the queue reviewer then cannot adjudicate. Confirm |
| **P9** | **Retention: how long are Phase-2/Phase-3 raw LLM responses kept?** | `system-architect` + DPO | They contain quoted ticket text including any residual PII. Needed for audit and for re-running the ablation, dangerous to keep forever |
| **P10** | **On real data, what do tiers 1–3 actually flag?** | Measurement — **four minutes of CPU, no LLM, no human, week 1** | §4.3.11's sizing rests on a **[measured] 2.06% cleanlab flag rate on a fixture whose labels are internally consistent by construction**. At a realistic 10–20% the judged pool is ~600–1,100 rows and the queue is ~400–700 rows / 17–29 h, which changes P4's answer. **This is the cheapest unanswered question in the project — run it first** |
| **P11** | **Does the LLM judge survive its own decision rule (§4.3.12)?** | Measurement, after the 100-verdict pilot | If `P_cheap ≥ 0.80` and `Yield_judge < 0.10`, drop tier 5 entirely: the corpus then contains **zero** model-provenance labels from this pipeline, which is a materially better position for every future model generation. Do not skip this test to save the pilot's 4 hours |
| ~~**P12**~~ | ~~Which encoder is used for the tier-4 embeddings?~~ | — | **Closed — there is no embedding tier** (§4.3.4). Kept as a numbered entry so the other documents' references to P12 resolve rather than dangle |
| **P13** | **Where does the out-of-distribution signal for [`llm-fallback-policy.md`](./llm-fallback-policy.md) §2 case 4 now live?** It was being provided incidentally by tier 4's embedding-distance-to-centroid | ML + `system-architect` | **This is a real requirement that must not vanish with the tier.** It is a *serving-time* signal whose threshold belongs on the production model's validation set, and the production classifier is already an encoder — so it belongs in the classifier spec and the fallback policy, not in the dataset pipeline. Track it as a hand-off, not a deletion |
| **P15** | **Is the service taxonomy in scope as-is, or are `terraform-provider` (15), `message-queue` (31), `cdn` (38) and `dns` (41) descoped, merged, or accepted as report-only?** | **Product owner**, on the Phase-0 report (§4.0.1) | [measured] these four cannot be evaluated at the planned 2,000-ticket gold test size — reaching 50 test positives would need 15,625 / 7,463 / 6,098 / 5,618 tickets respectively. **This is a scoping decision, not a modelling one, and it should be taken in week 1 from the Phase-0 report rather than discovered in week 5 from an evaluation table.** Classifier spec §4.8 already requires a mapping table for any merge |
| **P16** | **Is a longer collection window or targeted enrichment sampling available for the rare tail?** | Product owner + backend | The only two levers that create rare positives; resampling cannot (§4.0.1.4). Enrichment needs its own reweighting and **cannot be the main test set** (runbook §0), so its cost is not just annotation |
| **P14** | **Widen the tier-3 selector to `score < 0.30` (~285-row queue, ~28 h) or keep `< 0.20` (~200 rows, ~25 h)?** | Support lead + ML, with P4 | §4.3.11. The widening buys [measured, validation §3.7] roughly +14pp injected-error recall on the ambiguous process for ~3.6 h. **Decide on the real export's flag rate (P10), not on this fixture's** |

---

## 9. Implementation notes and inference requirements

These are constraints for `dataset-pipeline-architecture.md` and for the model-serving design.
**I am not specifying the stage DAG, artifact layout, caching strategy, concurrency model, retry
policy, cost model, or queue plumbing** — those are the architecture document's, and it should
contradict me freely on any of them.

### 9.1 Artifacts this pipeline must produce

| Artifact | Contents | Size |
|---|---|---|
| `corpus.parquet` (or equivalent) | one row per ticket: `row_uid`, raw fields, Tier-A `model_input_text`, canonical `services`, `label_provenance`, `dup_cluster_id`, `lang_primary`, `code_switched`, quality flags, per-phase audit counters | ~5 MB at 5k rows [estimate] |
| `preprocess.json` | **the Tier-A contract**: detector inventory + patterns, placeholder set, protected-span list, folding rules, input template, `max_len`, truncation strategy, blocklist hash, `preprocess_version` | see P7 |
| `boilerplate_blocklist.txt` | frozen sentence blocklist, derived **from the training split only**, human-reviewed | 9 KB at 149 entries [measured] |
| **`label_profile.md` + `label_profile.json`** | the Phase-0 report (§4.0.1.6): one-line summary, per-service table with tiers and projected split counts, cardinality histogram, per-quarter and per-language breakdowns, co-occurrence by reference, and the diff against the previous snapshot. **Linked from the snapshot manifest and rendered, not just serialised** | < 100 KB |
| `gold_ids.json` | the frozen gold-set ID list, content-hashed | small |
| **`label_conflicts.parquet`** | tier-1 output: group id, grouping method, threshold, member row ids, member label sets, `conflict_grade`, action taken | small |
| **`tier2_pred_probs.npy` + `tier2_folds.json`** | the `(N, 20)` out-of-sample probability matrix and the exact `GroupKFold` assignment. **Both are required for reproducibility** — cleanlab's output is a pure function of these, so keeping them means a flag list can be re-derived without re-training | ~740 KB at 4,622 rows, fp64 [computed] |
| **`label_issues.parquet`** | tier-3 output: per row, `cleanlab_flagged`, `label_quality_score`, per-service flags, and the selector definition used | small |
| **`taxonomy_diagnostics/`** | `common_multilabel_issues` table, `rank_classes_by_multilabel_quality`, `overall_multilabel_health_score`, **the per-service-pair confusion table (both directions) and the label co-occurrence matrix** (§6.1), each with a diff against the previous snapshot. **This directory is the deliverable for the product owner**, not for the trainer. No plot, no embedding matrix, no UMAP coordinates — all cut with tier 4 | **< 200 KB** — it is all tables |
| `judge_verdicts.parquet` | every verdict, evidence span, basis, run id, prompt version, model, self-consistency votes, **and the selecting stratum (S1–S4)** | ~200 KB [estimate] at 288 judged rows |
| `review_queue.parquet` | routed rows with the firing rule, judge evidence, and reviewer outcome | small |
| `canary_set.json` + `canary_report.json` | versioned canaries and the recall/precision measurement | small |
| `pipeline_report.json` | every metric in §6.1 | small |
| `snapshot_manifest.json` | content hashes of everything above, library versions (**including `cleanlab_version` and `scikit-learn` version**), `preprocess_version`, `judge_prompt_version`, `judge_model`, `tier2_run_id` | small |

**None of the Phase-3 artifacts ship with the model.** They are dataset-construction evidence.
The only artifacts that cross into serving are `preprocess.json` and the blocklist (§9.2).

### 9.2 The inference contract — what must run at serve time

**Requirement, not a suggestion:** the Tier-A path is a single shared code path used by training,
evaluation and inference, loaded from `preprocess.json`. Classifier spec §9.2 already requires
that all preprocessing runs inside the model service; this document specifies *what* that
preprocessing is.

- **Input:** `{ticket_id, title, description, priority, created_at}` — unchanged from classifier
  spec §9.2.
- **Output of the Tier-A path:** the `model_input_text` string, plus a redaction receipt
  (placeholder counts by class) for logging. **The receipt, not the text, is what gets logged.**
- **No network calls, no corpus statistics, no LLM, no database lookup** anywhere in Tier A.
- **Latency budget:** [measured] 0.09 ms/row for the regex tier. NER, if Tier A, must come in
  ≤ 25 ms p95 (§4.1.5). Total preprocessing ≤ 30 ms p95, against the 250 ms p95 model-call
  budget in classifier spec §9.3.
- **Memory:** regex tier negligible; slovnet adds ~205 MB RAM and a 27 MB artifact if Tier A —
  **which would roughly double the 503 MB RSS figure in classifier spec §4.5 for the base model,
  and must be in the architect's per-replica sizing.**
- **Version pinning:** the serving process must assert that its `preprocess_version` matches the
  model artifact's at startup and refuse to serve on mismatch (risk 10).

### 9.3 Reproducibility

- **Snapshot immutability.** Every artifact content-hashed; the manifest hash is what a training
  run records. Never train against a live query or a mutable file (classifier spec §9.5).
- **Pinned versions** for every library in §4.1.12, in the lockfile and in the model card.
  Detector behaviour changes between library versions and that silently changes the corpus.
- **Seeds** for the MinHash permutations, the gold-set hash salt, the canary injection offsets,
  and the Phase-3 candidate shuffles. All recorded.
- **LLM reproducibility is not achievable and must not be assumed.** Record model id, prompt
  version, and every raw response (subject to P9) so a run is *auditable* even though it is not
  bit-reproducible. This is another reason the Tier-A/Tier-B split matters: **the reproducible
  part of the pipeline is exactly the part that touches the model input.**

### 9.4 Requirements handed to `system-architect`

Stated as constraints, with the design left open:

0. **Phase 0 (§4.0.1) runs immediately after ingest and before Phase 1**, reads only `services`,
   `created_at` and the text, writes only a report, and **G19 stops the build from its result**.
   It must be runnable standalone against a raw CSV with no other stage configured — that is its
   whole point. Phase numbering is deliberately 0 so existing cross-references stay valid.
1. Phase ordering is a correctness property, not a performance choice: deterministic redaction
   **before** any LLM call; gold sample drawn **before** any Phase-3 tier; the gold-set assertion
   aborts the run (§4.3.13).
2. Phase 1 must be re-runnable standalone and must produce a complete, valid corpus on its own.
   Phases 2 and 3 are strictly additive.
3. **Phase 3 is a cascade with a strict dependency order** — tier 2 needs the dup-cluster ids from
   §4.1.7, tier 3 needs tier 2's probability matrix, tier 5 needs tiers 1, 3 and the tier-2
   probabilities to form its strata. **Tiers 1–3 must be runnable without any LLM configured at
   all**, because that is the mode the pipeline runs in if P1/P2/P11 come back negative, and it is
   the mode it runs in on every re-snapshot where the judge is not re-run.
3b. **Tier 4 is cut (§4.3.4).** Everything it required comes out of the runtime with it: the
   embedding stage and its `audit_t4` table, the `t4_*` columns, the UMAP artifacts, the
   checkpoint and embedding-matrix pinning entries, the model-cache directory and pull step, and
   the `sentence-transformers` / `torch` / `transformers` / `umap-learn` dependencies. **One thing
   must not be silently lost with it:** the embedding-distance-to-centroid signal was doubling as
   the out-of-distribution detector for llm-fallback-policy §2 case 4. That requirement is real,
   it is a *serving-time* signal, and it moves to the classifier spec — see P13. Do not retain an
   embedding stage in the dataset pipeline to provide it.
4. **Tier 2's cross-validation folds are grouped on `dup_cluster_id` and this is a gate (G14)**,
   not a convention. It is the single easiest thing in this pipeline to break by accident.
5. Raw LLM responses contain quoted ticket text — storage, access control and retention are
   yours (P9).
6. The human review queue needs: the ticket, **which tier(s) selected the row**, the judge's
   verdict and evidence if any, the firing routing rule, a decision control, **and 20 blind
   interleaved items per batch** for the anchoring check (risk 6). Queue items now arrive from
   tiers that produce no verdict at all (R8 tier-1 conflicts), so the UI cannot assume a judge
   verdict is present.
7. `preprocess.json` (plus the blocklist) ships **with the model artifact**, and the serving
   process asserts version equality at startup.
8. If NER is Tier A, per-replica memory rises by ~205 MB over the classifier spec §9.4 figure.
9. Every phase writes a new artifact; nothing is mutated in place. The pipeline must be
   re-runnable from any phase given the prior phase's output. `tier2_pred_probs.npy` in
   particular must be persisted, so tier 3 can be re-run under a new cleanlab version or a new
   selector threshold without re-training tier 2.

### 9.5 Sequencing

0. **Day 1, before anything else — Phase 0 on the real export** (§4.0.1). Seconds of pandas, no
   dependencies, no budget approval. It answers whether the corpus can support the taxonomy at
   all, and **a T-0 service (G19) or a tail that cannot reach 30 projected test positives is a
   scoping conversation with the product owner, not a pipeline run.** Running this after the
   build is the mistake this phase exists to prevent: the answer costs seconds and it gates
   ~$25 of LLM calls and ~28 person-hours.
1. **Week 1** — Phase 1 end to end on the fixture. Canary set v1. Human sample n=200. Gates
   G1–G8. **This alone produces a usable corpus**, and per §3 it may be all that is needed.
2. **Week 1** in parallel — the NER latency benchmark (P5) and the B0-vs-B1 ablation.
3. **Week 1, and this is the change from the first revision — run Phase-3 tiers 1–3 on the real
   export as soon as it exists.** [measured] four minutes of CPU, no LLM, no human, no budget
   approval needed. It answers P10, sizes the queue, and produces the taxonomy diagnostics that
   the annotation guideline (classifier spec §2.2) should be written *against*. **Doing this
   before the guideline is written is worth more than doing it after.**
4. **Week 2** — Phase 2 on a 1,000-row sample as a discovery exercise. Human-review the mined
   rule diff. Merge approved rules into Phase 1. Run the §4.2.4 decision rule before committing
   to full coverage.
5. **Week 2** — the §6.5 tiers-1+3 human sample (100 rows, ~4 h). This yields
   `P_cheap` and `Yield_judge` and therefore **decides whether tier 5 is built at all** (§4.3.12,
   P11).
6. **Week 2–3** — if tier 5 survives: the 100-verdict judge pilot, calibrating the per-service
   cap (G11) and the queue size before anyone commits person-hours.
7. **Week 3** — the queue, the ablation ladder including rung 3c, the snapshot freeze.

**Nothing in weeks 2–3 blocks the classifier work.** The week-1 Phase-1 corpus is a complete
input to the training sweep, and the ablation ladder is designed so that later phases are
additive comparisons against it rather than prerequisites.

### 9.6 A note on ordering relative to the runbook

This pipeline is the **executable text-processing layer under
[`dataset-construction-runbook.md`](./dataset-construction-runbook.md)**, not a replacement for
it. The runbook's order of operations still governs: audit internal messages → build
`work_provenance` → check Case C → measure balance → **deduplicate, then draw the gold sample**
→ annotate blind → adjudicate → compute Case A precision/recall → freeze the snapshot. This
document supplies steps 4–5's mechanics (cleaning, dedup clusters, language tags, quality flags)
and adds two stages the runbook does not have (Phases 2 and 3), both of which sit **after** the
gold sample is drawn and are fenced from it (§4.3.13).

One addition the runbook should absorb: **its step 7 ("compute Case A label precision/recall and
Case B agreement, and decide how to train on them") now has a cheap companion that runs at step
4.** Tiers 1–3 give a label-quality picture — conflict counts, per-service flag rates, the
`common_multilabel_issues` direction table — **before** the annotation guideline is written, for
four minutes of CPU and no annotation budget. Runbook §7 orders the annotation first because it
assumes the only instrument is a human; that assumption is no longer true for the *diagnostic*
question, though it remains true for every *ground-truth* question.

On this repository's fixture the runbook's steps 1–3 cannot be run at all, because there is no
`ticket_messages` table (§7.1). That is a property of the fixture, not a gap in the runbook.

### 9.7 Scratch work backing this spec

Throwaway measurement scripts, not part of any deliverable and not to be imported by anything:
`/tmp/claude-0/-home-user-bert-poc/351181d2-9b69-560c-8223-87b870e37c96/scratchpad/m1.py`
through `m17.py`, plus the downloaded `xlmr_tokenizer.json` used for token counts. Libraries used
for measurement only: `pandas` 3.0.5, `tokenizers` 0.23.1, `datasketch` 2.0.0, `py3langid` 0.3.0,
`scikit-learn` 1.9.0, `cleanlab` 2.9.0. No files were added to the repository outside
`docs/specs/`.

Two measurement caveats that belong with the numbers they qualify:

- **ΔB (§4.3.3, G17) is approximate.** The cleanlab flags used to filter the training folds were
  computed from the *global* out-of-sample probability matrix, so a flag on a training row was
  derived from a model that had seen some held-out rows. The proper protocol computes flags with
  an inner CV inside each outer training fold. The measured effect (−0.79pp macro-AP) is large
  enough and in a consistent enough direction that the conclusion — flags are not noise on this
  fixture — survives the approximation, but **the implementation must use the nested protocol**
  and re-measure. Independently reproduced at **−1.10pp** in validation §3.8 under the same
  approximation, which strengthens the reading.
- **The tier-4 caveat that used to sit here is superseded and worth keeping as a process note.**
  It said the kNN numbers were a TF-IDF character-cosine proxy whose thresholds "must be
  re-derived", and treated the routing rule as making that safe to defer. It was not safe to
  defer. [measured, validation §3.2–3.3] the real encoder flags **36 rows at the ≥0.95 cut, not
  56**, and its top-56 overlaps the proxy's by **34%**, while two TF-IDF variants overlap each
  other by 73% — the list was tracking the representation, not the labels. **The lesson is
  general and is now encoded as gate G18:** a caveat is not a substitute for a gate, and a tier
  whose selection is that sensitive to an arbitrary representation swap should not enter a
  specification before someone measures what it selects. Full working in
  [`knn-tier-validation.md`](./knn-tier-validation.md).
