# Fine-tuning dataset construction pipeline: text processing, LLM cleaning, and label adjudication

Status: specification (not implemented)
Author: ml-researcher
Date: 2026-08-16
Input: [`data/raw/tickets_export.csv`](../../data/raw/tickets_export.csv) (5,013 rows)
Output: a versioned training corpus for the multi-label ticket→services classifier

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

## 0. Repo state, and three disagreements with the brief stated up front

### 0.1 Existing assets

`/home/user/bert-poc` contains four specification documents, `data/raw/tickets_export.csv`
(5,013 rows, 2.4 MB), and its README. **There is no code of any kind** — no pipeline, no
notebooks, no preprocessing module, no `preprocess.json`, no model artifacts, no
`ticket_messages` companion table. Everything below is specified from scratch. Nothing is
being reused because there is nothing to reuse.

### 0.2 Three places where I do not agree with the brief

These are the load-bearing arguments in this document, so they go first rather than buried.

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
policy). A fused call cannot satisfy both. There is a second, independent reason in §4.3.4.

**(3) Placeholder normalisation is not a token-budget play on this corpus.** Classifier spec
§4.6 estimates that placeholder normalisation "typically recovers 10–30% of the token budget".
**[measured]** on this corpus it recovers **2.1%** overall and **8.3%** on the 922 rows that
contain anything to replace. The justification for placeholders here is compliance and
skew-stability, not tokens. What *does* recover budget is boilerplate removal: 5.5% corpus-wide,
and **65.1%** on the 99 auto-alert rows. Section §4.1.6 has the table. I am not asking anyone to
change §4.6 — its estimate was explicitly flagged as one to measure per corpus. This is the
measurement, and this corpus does not support it.

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
2. **No leakage of PII or secrets into the training corpus, model artifacts, or logs.**
   Measured against seeded canaries (§6.4), gated at recall ≥ 0.99 for structured classes.
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

### 2.3 PII, secret and identifier inventory [measured]

Counts are over `title + "\n" + description`. "Rows" = rows containing ≥1 instance.

| Class | Rows | Instances | Detectable by | Verdict (§4.1.4) |
|---|---|---|---|---|
| E-mail address | 262 | 265 (189 distinct) | regex | `<EMAIL>` |
| URL | 210 | 210 (110 with a query string) | regex | `<URL:{host_class}>`, query dropped |
| **URL-borne secret** (`?token=`, `&sig=`, `key=`) | 109 | **81 values, 80 distinct** | **structure, not entropy** | `<URL:...>` (destroyed by construction) |
| Phone `+7…` | 126 | — | regex (6 surface formats present) | `<PHONE>` |
| Phone `+1…` | 53 | — | regex | `<PHONE>` |
| Card-like (`4111 11** **** 1111`) | 21 | — | regex; **partially masked already**, so a naive 13–19-digit-run regex matches **0** | `<CARD>` |
| ИНН / ОГРН / КПП keyword | 143 | — | keyword + value regex | `<INN>` on the value |
| ИНН with a 10/12-digit value | 24 | 24 | regex, checksum-validatable | `<INN>` |
| Legal entity `ООО «…»` | 32 | 33 | regex | `<ORG>` |
| Person name, RU (two capitalised words) | 359 | 370 | NER required | `<PERSON>` |
| Person name, EN (`Firstname Lastname` shape) | 265 | 282 (145 distinct) | NER required | `<PERSON>` — **see the trap below** |
| IPv4 | 17 | — | regex | `<IP>` |
| ISO timestamp | 100 | — | regex | `<TS>` |
| Host/resource id (`i-55ba20`, `db-01`, `lb-prod`, `api-gw-1`) | 182 | 219 | regex | **preserved** (§4.1.4) |
| base64-ish run ≥32 chars | 1 | 1 | regex | `<B64>` |
| UUID · JWT · AWS key · SSH private key · connection string · `Bearer` · `password=` | **0** | 0 | regex | rules kept as insurance |

**The trap, and it is the single most important measurement in this document.** Of the 282
English `Firstname Lastname`-shaped matches, **126 (45%) are technical strings, not people**:
`Service Unavailable` (34), `Too Many [Requests]` (25), `Bad Gateway` (10), `Internal Server`
(9), `Gateway Timeout` (5), `Method Not [Allowed]` (5), `Payload Too [Large]` (5),
`Unprocessable Entity` (3), `Seq Scan` (6), plus `Host Loss` / `Snt Last` / `Avg Best` from
pasted `mtr` output. A PERSON detector — regex *or* NER — pointed at raw ticket text will
redact exactly the HTTP reason phrases and PostgreSQL plan nodes that are the strongest label
signal in the corpus. This is why §4.1.3 makes **protected-span masking strictly precede
detection**, and why §6.4 gates on over-redaction precision and not only on recall.

### 2.4 Why entropy-based secret detection is not usable here [measured]

The 81 secret values found in URL query strings:

- length **3–14 characters** (median 12);
- Shannon entropy **1.58–3.81 bits/char** (median 3.08);
- **0 of 81** exceed `detect-secrets`' default `Base64HighEntropyString` limit of **4.5**;
- **49 of 81 (60%)** exceed its default `HexHighEntropyString` limit of **3.0**, so even the
  permissive detector has a **40% false-negative rate** on this corpus
  ([Yelp/detect-secrets](https://github.com/Yelp/detect-secrets), Apache-2.0; the project itself
  states it "is not meant to be a sure-fire solution to prevent secrets from entering the
  codebase").

Short opaque tokens in customer-pasted URLs are *structurally* obvious and *statistically*
invisible. The design consequence is in §4.1.4: **redact the container, not the value.** Drop
the entire URL query string unconditionally and you destroy 81/81 by construction, with no
detector accuracy involved. That is a property, not a measurement, and it is the only kind of
guarantee worth having about secrets.

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
is not a queue, it is the corpus. §4.3.6 specifies a rule that routes ~14%.

### 2.9 Licensing and residency

The data is first-party; there is no dataset licence question. The binding constraint is
**152-FZ** (classifier spec §2.9, open question Q7). It decides §4.2.6 (self-hosted vs hosted
LLM) and it is not optional. This document assumes the worst case — text may not leave the
Russian perimeter — and specifies a pipeline that works under it, with the hosted path as a
documented upgrade if legal clears it.

---

## 3. Baselines

Runbook §0's governing principle applies to pipelines too: **every stage must beat doing
nothing, and the cheap comparison comes first.** The ablation ladder in §5.2 is evaluated
against these.

| # | Baseline | What it is | Why it must exist |
|---|---|---|---|
| **B0** | **Raw passthrough** | `title + "\n" + description`, no processing at all beyond CSV parsing and `services` canonicalisation | The floor. If B1 does not beat this on the gold test set, the entire pipeline is ceremony. Cheap to run |
| **B1** | **Phase 1 only** | §4.1 deterministic pipeline, no LLM anywhere | **The real baseline.** This is the trivial-baseline slot from classifier spec §3, transposed to the data layer. Phase 2 and Phase 3 must each beat *this*, not B0 |
| **B2** | **Phase 1 + Phase 3 (judge), no Phase 2** | labels adjudicated, text deterministic | Isolates label repair from text cleaning |
| **B3** | **Phase 1 + Phase 2, no judge** | text improved via mined rules, labels untouched | Isolates text cleaning from label repair |
| **B4** | Full pipeline | | Must beat max(B2, B3) or the extra phase is dropped |

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
| Regex placeholder substitution (email, phone, URL, card, ИНН, IP, TS, IDs, secrets) | A |
| Log/traceback skeleton reduction | A |
| Input template assembly, truncation | A |
| NER-based `<PERSON>` / `<ORG>` redaction | A **only if** the budget in §4.1.5 is met; otherwise B |
| `services` / `labels` / `flags` canonicalisation | B (labels, not text) |
| Language identification and code-switch tagging | B |
| Near-duplicate clustering | B |
| Length/emptiness/quality filters and quarantine | B |
| **Phase 2 (LLM), all of it** | **B** |
| **Phase 3 (judge + human queue), all of it** | **B** |

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
| URL query strings and fragments | everything after `?` or `#` | 110 of 210 URLs | **The secret story, §2.4.** 81/81 URL secrets die by construction, no detector involved |
| Whitespace runs, zero-width, CSV artefacts | §4.1.2 | 207 rows | Noise, no information |

**REPLACED by a stable placeholder** (the placeholder is a special token, §4.1.6):

| Placeholder | Detector | Measured | Why keep a marker at all |
|---|---|---|---|
| `<EMAIL>` | regex | 265 instances / 262 rows | Presence of an address is weak signal (`notifications`, `auth` invite flows). Deletion would break sentence structure |
| `<PHONE>` | regex, 6 RU + 2 EN surface formats | 179 rows | Same |
| `<URL:{class}>` | regex; `{class}` from a **first-party host allowlist** (`docs`, `api`, `logs`, `portal`, `cdn`, `app`, `hooks`) else `other` | 210 URLs; allowlist covers 208/210 | `hooks.*` → `integrations`, `cdn.*` → `cdn` is real signal. The host is first-party and therefore not PII; path and query are dropped |
| `<CARD>` | regex over digit runs **including `*` masks** — a plain `\d{13,19}` regex matches **0** here because the fixture pre-masks them | 21 rows | Presence of a card is signal for `billing` |
| `<INN>` | keyword-anchored 10/12-digit value + control-digit validation | 143 rows keyword, 24 with value | Presence signals `billing`/accounting documents |
| `<ORG>` | `(ООО|АО|ЗАО|ПАО|ИП)\s*[«"]…[»"]` | 32 rows | Legal-entity names are identifying; the *fact* of one is billing signal |
| `<IP>` | regex, both v4 and v6 | 17 rows | Networking signal; the address itself is customer infrastructure |
| `<TS>` | ISO-8601 and common RU/EN date forms | 100 rows | High-entropy, zero-signal, per §4.6's list |
| `<ID>` | `req_[0-9a-f]+`, `rule_\d+`, `i-[0-9a-f]{6}`, ticket/order ids | 182 rows / 219 instances | **Placeholder, not preserve.** The *class* of identifier is signal (a `req_` id means the user pasted an API response); the value is unique per row, so preserving it adds one hapax per row and zero generalisation |
| `<UUID>`, `<B64>`, `<JWT>`, `<SECRET>`, `<KEY>` | regex; `<SECRET>` also fires on `(?i)(token|secret|ключ|пароль|password|api[- ]?key)\s*[:=]\s*\S+` outside URLs | 0 / 1 / 0 / — here | Insurance. Absent from this fixture, certain to appear in a real export |
| `<PERSON>` | NER — see §4.1.5 | 359 RU + 265 EN rows | Names carry no service signal; the placeholder keeps grammar intact |

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

[measured] under XLM-R, angle-bracket placeholders cost **3–5 tokens each**: `<EMAIL>` →
`['▁<','E','MAIL','>']` (4), `<SECRET>` → 5, `<URL>`/`<PHONE>`/`<ID>`/`<TS>` → 3.

Classifier spec §4.6 says to add special tokens "only if the redaction placeholders are
themselves over-segmented". **They are.** Recommendation: **add the placeholder set as special
tokens and resize the embedding matrix** — 11–14 tokens, each becoming a single unsplittable id.
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
| Secret regex pack | **gitleaks rules** (ported), + **detect-secrets** keyword/plugin detectors | MIT / Apache-2.0 | gitleaks is MIT, config-driven TOML rules with per-rule entropy and stopword allowlists, and scans arbitrary directories | — | rule packs target developer credentials, not customer-pasted URL tokens — **our own URL/keyword rules do the real work** |
| Secret verification (audit only) | **trufflehog** | **AGPL-3.0** | 700+ detectors with *live credential verification*, which nothing else offers | — | **Do not link it into the pipeline** — AGPL-3.0 on a proprietary internal service is a question for legal that we do not need to ask. Run it out-of-band as a CLI over the finished corpus as a release gate (§6.3) |
| Entropy detection | **not used as a primary detector** | — | [measured] 0/81 secrets at the 4.5 default, 49/81 at 3.0 | — | — |
| RU morphology | **pymorphy3** | MIT | maintained continuation of pymorphy2; needed for the TF-IDF/keyword baselines' lemmatisation and RU service-synonym matching | — | **not needed for the encoder path** — SentencePiece handles morphology; scope it to baselines only |
| Language ID | **lingua-py** (`{ru,en}` only) | Apache-2.0 | span-level multi-language output; strong on short text | **py3langid** (BSD-3) kept as a cross-check; **fastText lid.176** only at ≥1M rows | none material for RU/EN; the risk is running it on unmasked text |
| Near-duplicates | **datasketch MinHashLSH** | MIT | cluster ids as a column; 2.1 ms/row [measured] | **text-dedup** (Apache-2.0) above ~1M rows or for a Spark backend; **SimHash never** | char-5-grams are language-agnostic; RU morphology is handled by character n-grams |
| E-mail quote/signature | **own deterministic rules first**; **talon** (Apache-2.0) only if quote markers exceed 2% of rows | Apache-2.0 | [measured] only 10 rows have `^>` quoting here; talon's ML signature classifier was trained on English ENRON mail and has no documented Russian support | `email-reply-parser` (MIT) is simpler but equally English-pattern-driven | **`С уважением` is 154 rows and neither library knows it** — the RU sign-off marker list is ours to write and version either way |
| Row representation | **polars → Parquet**, one file per phase, never in place | MIT / Apache-2.0 (pyarrow) | columnar, typed, hashable, cheap to diff between phases | **pandas** is fine at 5k rows and this choice is aesthetic below ~100k; it stops being aesthetic at 1M | — |
| Training handoff | **HF `datasets`** at the training boundary only | Apache-2.0 | Arrow-backed, integrates with the trainer | — | — |

**Where the honest gap is.** [measured] this fixture contains **zero** JWTs, AWS keys, SSH keys,
connection strings, `Bearer` headers, or `password=` assignments. Every detector for those
classes is therefore **completely untested by this corpus**. Their correctness rests entirely on
the seeded-canary protocol in §6.4. Do not read a green Phase-1 gate on this fixture as evidence
that secret handling works.

#### 4.1.13 The honest answer on secret false negatives

**No deterministic detector has an acceptable false-negative rate for a secret, and neither does
Phase 2.** [measured] entropy misses 100% at the standard base64 threshold and 40% at the hex
threshold; regex packs only find shapes someone anticipated; an LLM second opinion is a
probabilistic detector with no recall guarantee and — critically — **it has already seen the
text by the time it reports.** Nobody should present any of these as a compliance story.

The defence-in-depth answer has six layers, and only the first is a guarantee:

1. **Structural destruction, not detection.** Drop URL query strings and fragments
   unconditionally; drop the entire value after a secret-ish key name; drop signature blocks
   wholesale. [measured] this kills 81/81 of this corpus's secrets **by construction**. Prefer a
   rule that cannot miss over a detector that usually does not.
2. **Regex packs** (gitleaks + detect-secrets keyword plugins) for known credential shapes.
3. **Seeded canaries with a hard recall gate** (§6.4) — the only quantitative measurement
   available.
4. **Phase 2's `residual_pii` output** as an independent second opinion (§4.2.3). It protects
   everything downstream of the Phase-2 call; **it cannot protect the Phase-2 call itself**,
   which is precisely why the phase ordering is load-bearing (§4.2.6).
5. **An out-of-band trufflehog scan of the finished corpus** as a release gate, with live
   verification on any hit (§6.3).
6. **A human read of a stratified sample** (§6.5), plus a standing adversarial exercise: once
   per snapshot, someone tries to write a ticket that defeats the pipeline, and whatever they
   find becomes a rule and a canary.

And one non-technical layer that outranks all six: **keep the corpus inside the perimeter.**
A residual secret in a corpus that never leaves is an incident waiting to happen; the same
secret in a corpus posted to a foreign API is an incident that has already happened.

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
                             "passport", "address", "secret", "account", "other"]}
        }
      }
    }
  }
}
```

**Allowed to alter:** nothing in the text. It emits indices, enums, booleans, and verbatim
quotes.

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

#### 4.2.6 152-FZ, and why the phase ordering is load-bearing

**Default and recommendation: a self-hosted model inside the Russian perimeter for Phases 2 and
3.** Three reasons, and the first is the one that decides it:

1. **The deterministic redactor's false-negative rate on secrets is not zero and cannot be
   proven to be zero** (§4.1.13). Sending "redacted" text to a foreign API is a bet that the
   redactor is perfect. [measured] on the fixture's own secrets, standard entropy detection is
   0-for-81. Do not take that bet with customer text.
2. **152-FZ** requires that recording, storage and extraction of Russian citizens' personal data
   occur in databases located in Russia (classifier spec §2.9; open question Q7). Pseudonymised
   text arguably falls outside the regime — but "arguably" is a legal opinion we do not have, and
   the pipeline should not be blocked on obtaining it.
3. **Feasibility is settled by volume, exactly as in llm-fallback-policy §3.** This is a
   **one-off batch of ~5k–50k rows with no latency requirement**, which is a far easier ask than
   the production fallback path that document already found feasible on CPU. A dataset build that
   takes six hours instead of forty minutes costs nothing.

**Phase ordering is the load-bearing control, and it must be stated as a rule:** Phase 1's
deterministic redaction runs **before** any LLM call, without exception. Phase 2's
`residual_pii` output is a *downstream* protection — it protects the training corpus, the logs,
Phase 3, and the human queue. It cannot protect the Phase-2 call itself, because by then the text
has already been sent. Under a self-hosted deployment inside the perimeter, that residual
exposure is not a disclosure event at all, which is the fourth reason self-hosting is the
default.

**Upgrade path.** If legal (Q7) clears pseudonymised text leaving the perimeter, run a
**200-row head-to-head** — self-hosted versus hosted frontier — scored against human-adjudicated
span roles, and switch if the hosted model's agreement is materially better. Record which model
produced every row's metadata in the provenance record either way, because a mid-snapshot model
switch is otherwise invisible and irreproducible.

---

### 4.3 Phase 3 — LLM-as-a-judge, and the human queue

#### 4.3.1 The pattern, reused not reinvented

This is [llm-fallback-policy §4.1 constrained adjudication](./llm-fallback-policy.md), applied
to stored labels instead of to a classifier shortlist. The shortlist is the row's existing
`services` set; the verdict is accept/reject per candidate; the output is schema-constrained with
an `enum` over service names, so an invalid service name is not a failure mode that has to be
handled. Requiring an `evidence` span is a grounding constraint and gives reviewers something to
audit — same argument, same schema, one addition.

The general caution from
[Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)](https://arxiv.org/pdf/2306.05685)
applies and is why §4.3.5 exists: LLM judges exhibit **position bias, verbosity bias and
self-enhancement bias**, and the paper's headline agreement-with-humans result is for *pairwise
preference* judging, not for taxonomy label adjudication. **Do not import their agreement number
as an expectation for this task.** Measure ours on the pilot.

#### 4.3.2 Verdict schema

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

#### 4.3.3 The three hard cases the taxonomy deliberately contains

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
> was just handed. §4.3.6 caps it: **no service's reject rate may exceed 15% [estimate — calibrate
> on the pilot] without explicit human sign-off on that service's whole reject list.**

#### 4.3.4 The missing-label direction

The judge can only reject what is present. The missing-label problem is real and runbook §3
already identifies its consequence: labels with reliable positives and **unreliable negatives**,
trained with plain BCE, actively teach the model that correct services are negatives.

**Recommendation: allow proposals, never auto-apply them as positives, and prefer masking to
adding.**

| Option | Effect on the label set | Effect on provenance | Verdict |
|---|---|---|---|
| Auto-apply proposals as positives | recall of labels ↑ | **every added positive is LLM-authored** — model provenance, and the exact distillation failure of proposal §3 | **No** |
| Route proposals to the human queue | recall ↑ where a human agrees | `human_after_judge` (anchored, §4.3.8) | Yes, for high-value cases only (§4.3.6) |
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

**The ablation (4 arms, 5 seeds, identical everything else, evaluated once on the blind gold
test set):** (i) Phase-1 labels; (ii) + judge rejections applied; (iii) (ii) + proposals masked;
(iv) (ii) + proposals applied as positives. Arm (iv) is included **only** as a diagnostic and is
**not shippable** regardless of its score, because it puts LLM-authored positives in the training
set — llm-fallback-policy §5. If (iv) wins by a lot, that is a finding about label recall to
report to the annotation effort, not a licence to ship it.

#### 4.3.5 Self-consistency, and reconciling it with the ≤1-call budget

**The ≤1-call constraint is scoped per phase, and I am stating that explicitly rather than
assuming it.** A judge call is a second LLM call per row; a 2-of-3 self-consistency scheme is a
second, third and fourth. If the constraint were global the judge could not exist at all, which
is clearly not the brief's intent.

That said, 3× on every row is wasteful, so:

**Escalating self-consistency.**

1. **One call per row** with the candidate services in stored order. [measured] 4,622 rows have
   ≥1 service → 4,622 calls.
2. **Escalate to two more calls, with the candidate order shuffled**, only for rows where the
   first call produced ≥1 `reject` or ≥1 `proposed_addition`.
3. Decide: **3-of-3 unanimous reject → auto-apply. 2-of-3 → human queue. 1-of-3 → ignore**,
   and log it.

Order shuffling rather than temperature is the correct mechanism and llm-fallback-policy §4.1
already says why: temperature is rejected with a 400 on current models, so sampling variance has
to come from varying the prompt. Shuffling is also the direct mitigation for the **position
bias** documented by Zheng et al.

**[estimate]** at a 10–20% escalation rate: 4,622 + (0.15 × 4,622 × 2) ≈ **6,000 calls**, or
**1.30 calls per labelled row**. Cost at the §4.2.5 rates with a larger prompt (the ~20 service
definitions are the cached prefix, ~2,000 tokens): **≈ $30–45** [estimate] for the full corpus.
Again, cost is not the constraint.

**On fusing Phases 2 and 3 into one call.** The argument *for* is real: it halves the calls, the
model sees the ticket once, and evidence spans could be shared between the two jobs. **The
argument against is decisive and there are two of them:**

1. **The contamination fence makes fusion impossible.** Phase 2 **must** run on the gold val/test
   rows — their text must be preprocessed identically to training text or there is skew inside
   our own evaluation. Phase 3 **must never** run on the gold rows (llm-fallback-policy §5). One
   fused call cannot both run and not run on the same rows. Splitting the fused call by row
   population reintroduces two prompts, at which point nothing was fused.
2. **Fusion destroys the ablation.** With one call you cannot attribute a change to text cleaning
   or to label repair, so B2 and B3 in §3 collapse into B4 and the decision rules in §4.2.4 and
   §4.3.4 have nothing to compare.

There is a third, softer reason: a model asked to clean text and judge labels in one pass can
rationalise one to fit the other. Keeping them independent keeps the two signals independent.

**Do not fuse.**

#### 4.3.6 The routing rule — which rows go to a human

A row is routed to the human-supervision queue if **any** of the following fires. Rules are
ordered by expected volume and each carries its measured or estimated population.

| # | Condition | Rationale | Population |
|---|---|---|---|
| **R1** | Reject on a row where `|S| = 1` — applying it would empty the label set | An emptied row silently leaves the training corpus; that is too consequential for an unreviewed model verdict | [measured] 1,569 rows have `|S| = 1` (34% of labelled rows) |
| **R2** | Reject on a service in the rare tail (`terraform-provider` 15, `message-queue` 31, `cdn` 38, `dns` 41, `managed-redis` 98) | Classifier spec §2.8: below ~50 positives a service cannot be learned or evaluated. Losing rare positives is unrecoverable and cannot be detected by any aggregate metric | [measured] 212 rows carry a rare-tail service |
| **R3** | 2-of-3 self-consistency (not unanimous) | The judge is not sure; a human is the tiebreak | [estimate] ~40% of escalated rows |
| **R4** | `proposed_additions` with ≥2 entries, or one that would take `|S|` to 5 (violating the `≤4` contract) | Aggressive expansion is either a real missing-label finding or a hallucinating judge; both need eyes | [estimate] ~1–2% of rows |
| **R5** | A "swap" within an ambiguous pair: reject `auth`, propose `access-control` (or the reverse), same for `monitoring`/`logging`, `console-ui`/component | These are exactly the distinctions the taxonomy admits are contested; a model resolving them unilaterally is out of its authority | [measured] 2,730 rows touch an ambiguous family, but only the *flip* subset routes |
| **R6** | Any service whose corpus-wide reject rate exceeds **15%** [estimate] | A systematic prompt or guideline problem, not 700 independent label errors. **The whole service's reject list is held** pending sign-off | Whole-service, rare |
| **R7** | Random **2%** audit sample regardless of verdict | The only way to measure the auto-applied arm's error rate. **Non-negotiable** — without it, the auto-apply path is unmeasured | [measured] 92 rows |

**Not routed:** `judge_ungrounded` verdicts (evidence not verbatim). Those are a pipeline defect,
not a label question; discard the verdict, count it, and if the rate exceeds 2% fix the prompt
(§6.2).

#### 4.3.7 Queue size and person-hours

Arithmetic, all inputs either [measured] on the fixture or flagged [estimate]:

```
labelled rows                                        4,622  [measured]
(ticket, service) verdicts                           8,771  [measured]

base reject rate on this fixture                    3–8%    [estimate]
  (labels are author-assigned and internally consistent;
   a real CatBoost-contaminated column would be 10–20%)
  take 8%                                        → ~700 rejects

unanimous 3-of-3 (auto-apply)               ~60%   → ~420   [estimate]
2-of-3 → R3 queue                           ~40%   → ~280   [estimate]

R1: rejects landing on |S|=1 rows      0.34 × 700 → ~240    (overlaps R3)
R2: rejects on rare-tail rows          0.08 × 212 → ~17
R4: ≥2 proposals or |S|→5                          → ~50–70 [estimate]
R5: ambiguous-pair swaps                           → ~80–120[estimate]
R7: 2% random audit                                → 92     [measured]

union, de-overlapped                               → ~550–750 rows [estimate]
midpoint                                           → ~650 rows = 14% of labelled rows
```

**Person-hours:** ~2.5 min per item — the evidence span makes most decisions fast, and R5 swaps
are the slow ones. **650 × 2.5 min ≈ 27 person-hours** [estimate]. Add the Phase-3 prompt
calibration pilot (200 human-adjudicated verdicts, ~8 h) and the per-phase pipeline validation
samples (§6.5: 200 + 150 + 100 items, ~1.5 min each ≈ 11 h). **Total ≈ 46 person-hours**
[estimate].

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
number you cannot yet measure.

#### 4.3.8 The contamination fence — mechanics

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
Phase 3  input row set := all_rows \ gold_ids      ← hard assertion
```

Four mechanical requirements:

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
4. **Sampling independence.** The judge must not influence *which* rows get annotated. The gold
   sample is drawn by seeded hash over strata that contain no judge-derived field. Concretely:
   `dup_cluster_id`, `lang_primary`, `created_at` month, and cardinality are permitted strata;
   `judge_verdict`, `judge_disagreement`, and `is_unactionable` (Phase-2-derived) are **not**.

**Provenance markers.** Every row carries a structured `label_provenance`:

```json
{
  "label_source": "author_assigned | human_blind | human_after_judge |
                   llm_judge_confirmed | llm_judge_modified | catboost | transformer",
  "judge_touched": true,
  "judge_run_id": "...", "judge_model": "...", "judge_prompt_version": "...",
  "verdict_ids": ["..."], "changed_at": "...", "reviewer_id": "... | null"
}
```

`judge_touched` is set for **every row the judge scored, including rows it fully confirmed.**
Confirmation is not neutral: a confirmed row has passed through a model's opinion, and if we ever
discover the judge was systematically wrong about a service, we need to find every row it looked
at — not just the ones it changed.

**A row whose label was changed on the judge's advice is model provenance and can never be
Case A. Confirmed, and the reasoning is not close.** Case A is defined
([classifier spec §2.1.1](./ticket-services-classifier.md), runbook §2.1) as *"a human
demonstrably edited the field"*. A judge edit is a model edit; treating it as Case A would put
LLM output into the trustworthy-label pool and reproduce, with a second model, exactly the
feedback loop proposal §3 exists to escape. Concretely:

- `llm_judge_modified` → **excluded from val/test unconditionally.** Usable in training at
  reduced weight, subject to the §4.3.4 ablation, and reported as its own slice.
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

`preprocess_version` · `blocklist_hash` · `detector_inventory_version` ·
`gold_ids_hash`, plus `judge_prompt_version` and `judge_model` for arms that use Phase 3.

Every one of these goes in the snapshot manifest and in `model_card.md`.

### 5.2 The ablation ladder

One table, five arms, one decision rule per rung. Same 5-seed list throughout (10 seeds if the
usable label count lands under 5,000 — classifier spec §5.4). Reported as **mean ± sd**;
a single-seed number is not a result.

| Rung | Arm | Compared to | Ship the rung if |
|---|---|---|---|
| 0 | B0 raw passthrough | — | (reference floor) |
| 1 | B1 Phase-1 only | B0 | any improvement; if none, investigate before proceeding |
| 2 | B3 Phase 1 + Phase-2-mined rules | B1 | **≥ 1.0pp mean val `R@P90` and non-overlapping seed intervals** (§4.2.4) |
| 3 | B2 Phase 1 + judge rejections | B1 | ≥ 1.0pp, same test |
| 3b | B2 + proposals masked | B2 | ≥ 0.5pp; masking is nearly free, so a smaller bar is justified |
| 4 | B4 full | max(B2, B3) | ≥ 1.0pp over the better single phase, else ship the single phase |

Diagnostic-only, never shippable: proposals applied as positives (§4.3.4 arm iv).

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
| Phase 3 | ~1.30 calls/labelled row, ~2,100 in / ~150 out | ~$30–45 [estimate] | ~$300–450 [estimate] |
| Human queue | 2.5 min/item | ~27 h [estimate] | scales with reject count, not rows |

The deterministic phases are free. The LLM phases cost less than a day of engineering time. **No
part of this pipeline is cost-constrained**, which means every decision in it should be made on
correctness and risk, not on budget — and any argument in a design review that starts with "to
save on LLM calls" should be treated with suspicion.

---

## 6. Evaluation

### 6.1 Metrics that must be reported per snapshot

Emitted as `pipeline_report.json` and rendered as a table in the model card. No snapshot is
usable without it.

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

**Phase 3**

- verdicts issued; accept / reject / ungrounded counts;
- **reject rate per service** (the §4.3.6 R6 gate);
- escalation rate; 3-of-3 / 2-of-3 / 1-of-3 distribution;
- proposals by service; masked vs queued vs ignored;
- queue size by routing rule, and the overlap matrix between rules;
- **human agreement with the judge on the R7 2% audit sample** — this is the number that says
  whether the auto-apply path was safe;
- post-adjudication co-occurrence rates for the three ambiguous pairs, versus the pre-adjudication
  rates (`access-control`/`auth` 17.5%, `logging`/`monitoring` 4.2%, `console-ui` co-label 637)
  [measured].

### 6.2 Quality gates, thresholds, and what happens when one fails

**A failed gate stops the snapshot.** It does not produce a warning that someone reads later. The
snapshot is not written, and the run is reported.

| # | Gate | Threshold | On failure |
|---|---|---|---|
| G1 | Structural parse failures | ≤ 0.1% of rows | Stop. The export is malformed; fix the export, not the parser |
| G2 | Canary redaction recall, structured classes (email, phone, card, ИНН, URL-secret, JWT, AWS key) | **≥ 0.99** | Stop. Add the missing pattern, re-run, add the miss to the permanent canary set |
| G3 | Canary redaction recall, NER classes (person, org) | **≥ 0.90** | Stop below 0.90; between 0.90 and 0.95, ship with the gap recorded in the model card and a follow-up ticket |
| G4 | Over-redaction of protected technical strings | **≤ 0.5%** of protected tokens | Stop. This is the §2.3 failure mode (45% of person-shaped matches are error strings) and it silently destroys the label signal |
| G5 | Total tokens removed by the boilerplate blocklist | ≤ 60% of corpus tokens | Stop and review the blocklist. [measured] the ≥20× cut removes 48.2% of sentence instances; anything approaching 60% means the cut is eating content |
| G6 | Rows quarantined | ≤ 5% | Investigate before proceeding; a quarantine spike is usually one broken rule |
| G7 | Near-dup rows at Jaccard 0.80 | ≤ 10% of corpus | [measured] 2.8% here. Above 10%, the effective sample size is far below the row count and the confidence intervals in classifier spec §6.4 are wrong |
| G8 | Out-of-taxonomy service tokens | **0** | Stop. Either a typo, a retired service, or a taxonomy change requiring a mapping table (classifier spec §4.8). Never auto-drop |
| G9 | Phase-2 schema-invalid rate | ≤ 5% | Fix the prompt or the schema. Above 20%, the model tier is wrong |
| G10 | `judge_ungrounded` rate | ≤ 2% | Fix the prompt. This is a grounding failure, not a data finding |
| G11 | Per-service judge reject rate | ≤ 15% [estimate, calibrate on the pilot] | Hold that service's entire reject list for human sign-off (R6) |
| G12 | Human agreement with auto-applied judge verdicts (R7 sample) | ≥ 0.90 | Below 0.90, **revert all auto-applied rejects** for that run and route everything to the queue. The auto-apply path is a convenience, not a requirement |
| G13 | Gold-set contamination assertion | **exactly 0 overlap** | Abort the run. Non-negotiable |

### 6.3 Release gates on the finished corpus

Run once per snapshot, out of band, before the corpus is used for anything:

- **trufflehog filesystem scan with verification** over the exported corpus. Any *verified*
  finding is a security incident, not a data-quality issue: destroy the snapshot, rotate the
  credential, and add the pattern to Phase 1 and to the canary set. Unverified findings go to the
  §6.5 human sample. (Run as an out-of-band CLI — see the AGPL-3.0 note in §4.1.12.)
- **gitleaks `dir` scan** with the project ruleset as a cheaper second pass.
- **A grep for the raw values of every canary** injected in §6.4 — any survivor is a G2/G3
  failure that the recall computation somehow missed, which would itself be a bug worth finding.

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
   - URL secrets in the measured length band (3–14 chars, entropy 1.58–3.81) **and** above it;
   - JWTs, AWS keys, SSH private key headers, connection strings, `Bearer` headers — **none of
     which occur in this fixture**, so canaries are the *only* test those detectors will ever get
     (§4.1.12);
   - паспорт (`45 03 123456`), СНИЛС, БИК, account numbers — absent here, present in real
     support traffic.
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
   (§6.5), by trufflehog (§6.3), or by Phase 2's `residual_pii` becomes a permanent canary. This
   is what stops the same class of miss from recurring.

**The honest caveat, which must appear in the model card verbatim.** Canary recall measures the
detectors against *the distribution we invented*, which is the distribution we wrote the
detectors for. **It is an upper bound and it cannot find unknown unknowns.** It is a regression
test, not an assurance. The complements are the human sample (§6.5), the out-of-band verified
scan (§6.3), Phase 2's independent second opinion (§4.2.3), and the standing adversarial exercise
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
| **Phase 3** | **100 verdicts** | 40 rejects · 30 accepts · 30 from the 2-of-3 disagreement pool | (a) is the verdict right? (b) does the evidence span support it? (c) for the ambiguous pairs, is the decision rule being applied as written? | judge precision on rejects < 0.85 → auto-apply disabled, everything routes to the queue |
| **Blocklist review** | **all** | the full blocklist, once per version | is any entry signal-bearing? | any signal-bearing entry → remove it and add it to the protected-span list |

**On the Phase-3 sample's statistical power, stated plainly:** n = 40 rejects gives judge
precision to roughly ±10pp at 95%. That is enough for a go/no-go on the auto-apply path and
**not** enough for a per-service number. Do not publish per-service judge precision from this
sample; if a per-service number is needed, it needs its own sample per service, and at that point
the annotation budget conversation from §4.3.7 reopens.

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
- **Phase 3 does not repair this.** The judge is another model. A corpus whose labels were
  author-assigned and then judge-adjudicated has *two* non-human provenance classes, not zero.
- **Dedup thresholds tuned here will be wrong on real data.** [measured] the corpus is
  compositional: 149 sentence types cover 48.2% of sentence instances, and TTR is 0.047–0.054.
  Lexical statistics are systematically lower than a real corpus of this size. The Jaccard 0.80
  choice must be re-validated on real data against a human-labelled duplicate sample.
- **The 391 empty-`services` rows are ambiguous between "untriaged" and "genuinely
  unactionable"** and [measured] cannot be separated by length (median 57 vs 62 tokens). The
  pipeline can flag them; it cannot resolve them.
- **Whole detector classes are untested.** Zero JWTs, AWS keys, SSH keys, connection strings,
  `Bearer` headers, `password=` assignments, UUIDs, or tracebacks appear in this file. A green
  Phase-1 gate here says nothing about them.

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
| 3 | **Residual secret in the corpus** | Nothing, until it is in a model artifact or a log | G2, §6.3 verified scan, Phase-2 `residual_pii`, the adversarial exercise. **Assume this is nonzero and design storage accordingly** — this is a requirement for `system-architect`, not a hope |
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

---

## 8. Open questions

| # | Question | Owner | Why it blocks |
|---|---|---|---|
| **P1** | **Does 152-FZ permit pseudonymised ticket text to leave the perimeter for LLM processing?** | Legal / DPO (classifier spec Q7) | Decides self-hosted vs hosted for Phases 2 and 3. §4.2.6 defaults to self-hosted so work is not blocked, but the quality ceiling differs |
| **P2** | **Is there a self-hosted LLM available inside the perimeter, and of what tier?** | Infra / `system-architect` | If not, and P1 says no, Phases 2 and 3 cannot run at all and the pipeline is Phase 1 only — which §3's B1 baseline says may be sufficient anyway |
| **P3** | **Who authors the decision rules for `auth`/`access-control`, `monitoring`/`logging`, `console-ui`?** | Product owner | §4.3.3. These are taxonomy decisions, not ML decisions. Without them the judge is guessing and so are the annotators |
| **P4** | **Can ~46 person-hours be allocated for the queue and validation, *in addition to* the ~70 h gold budget?** | Support lead (classifier spec Q9) | §4.3.7. If the answer is no, Phase 3 is dropped and the gold set is protected. **This must not be resolved by taking hours from the gold set** |
| **P5** | **What is the NER latency on real tickets?** | Whoever runs the benchmark, week 1 | §4.1.5 gate. Decides whether name redaction is Tier A (runs at inference) or Tier B (training corpus only, with documented skew) |
| **P6** | **Is a residual secret in the training corpus an acceptable risk, and what is the containment plan?** | Security / DPO | §4.1.13 says the FN rate is nonzero and cannot be proven zero. The honest ask is for a containment requirement (encryption at rest, access control, retention), which is `system-architect`'s to design |
| **P7** | **Does `preprocess.json` have a size budget?** Classifier spec §9.1 says < 50 KB; [measured] the ≥20× blocklist alone is 149 entries ≈ 9 KB here, and a real corpus's could be far larger | `system-architect` | If the budget holds, ship the blocklist as a separate hash-referenced artifact. Either way it must be versioned with the model |
| **P8** | **Is the judge allowed to see `labels` and `flags`?** | Product owner + ML | They are excluded from the *model* input (classifier spec §2.7), but they are legitimate context for a *judge*. My recommendation is **no** — a judge with more context than the annotator produces verdicts the annotator cannot reproduce, and the queue reviewer then cannot adjudicate. Confirm |
| **P9** | **Retention: how long are Phase-2/Phase-3 raw LLM responses kept?** | `system-architect` + DPO | They contain quoted ticket text including any residual PII. Needed for audit and for re-running the ablation, dangerous to keep forever |
| **P10** | **On real data, what is the actual base reject rate?** | Measurement, after the Phase-3 pilot | §4.3.7's queue sizing assumes 8%. At 20% the queue is ~1,600 rows and ~67 h, which changes P4's answer entirely. **Run the 200-row pilot before committing to the queue budget** |

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
| `gold_ids.json` | the frozen gold-set ID list, content-hashed | small |
| `judge_verdicts.parquet` | every verdict, evidence span, basis, run id, prompt version, model, self-consistency votes | ~1 MB [estimate] |
| `review_queue.parquet` | routed rows with the firing rule, judge evidence, and reviewer outcome | small |
| `canary_set.json` + `canary_report.json` | versioned canaries and the recall/precision measurement | small |
| `pipeline_report.json` | every metric in §6.1 | small |
| `snapshot_manifest.json` | content hashes of everything above, library versions, `preprocess_version`, `judge_prompt_version`, `judge_model` | small |

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

1. Phase ordering is a correctness property, not a performance choice: deterministic redaction
   **before** any LLM call; gold sample drawn **before** Phase 3; the gold-set assertion aborts
   the run (§4.3.8).
2. Phase 1 must be re-runnable standalone and must produce a complete, valid corpus on its own.
   Phases 2 and 3 are strictly additive.
3. Raw LLM responses contain quoted ticket text — storage, access control and retention are
   yours (P9).
4. The human review queue needs: the ticket, the judge's verdict and evidence, the firing routing
   rule, a decision control, **and 20 blind interleaved items per batch** for the anchoring check
   (risk 6).
5. `preprocess.json` (plus the blocklist) ships **with the model artifact**, and the serving
   process asserts version equality at startup.
6. If NER is Tier A, per-replica memory rises by ~205 MB over the classifier spec §9.4 figure.
7. Every phase writes a new artifact; nothing is mutated in place. The pipeline must be
   re-runnable from any phase given the prior phase's output.

### 9.5 Sequencing

1. **Week 1** — Phase 1 end to end on the fixture. Canary set v1. Human sample n=200. Gates
   G1–G8. **This alone produces a usable corpus**, and per §3 it may be all that is needed.
2. **Week 1** in parallel — the NER latency benchmark (P5) and the B0-vs-B1 ablation.
3. **Week 2** — Phase 2 on a 1,000-row sample as a discovery exercise. Human-review the mined
   rule diff. Merge approved rules into Phase 1. Run the §4.2.4 decision rule before committing
   to full coverage.
4. **Week 2** — the Phase-3 pilot: 200 verdicts, human-adjudicated, calibrating the reject rate
   (P10), the per-service cap (G11), and the queue size before anyone commits person-hours.
5. **Week 3** — full Phase 3, the queue, the ablation ladder, the snapshot freeze.

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
gold sample is drawn and are fenced from it (§4.3.8).

On this repository's fixture the runbook's steps 1–3 cannot be run at all, because there is no
`ticket_messages` table (§7.1). That is a property of the fixture, not a gap in the runbook.

### 9.7 Scratch work backing this spec

Throwaway measurement scripts, not part of any deliverable and not to be imported by anything:
`/tmp/claude-0/-home-user-bert-poc/351181d2-9b69-560c-8223-87b870e37c96/scratchpad/m1.py`
through `m11.py`, plus the downloaded `xlmr_tokenizer.json` used for token counts. Libraries used
for measurement only: `pandas` 3.0.5, `tokenizers` 0.23.1, `datasketch` 2.0.0, `py3langid` 0.3.0.
No files were added to the repository outside `docs/specs/`.
