# Spec: CSV → dataset pipeline

Status: specification — reviewed before code is written
Input: [`data/raw/tickets_export.csv`](../../data/raw/tickets_export.csv) (5,013 rows)
Companions: [`ticket-services-classifier.md`](./ticket-services-classifier.md) (the model this feeds),
[`dataset-construction-runbook.md`](./dataset-construction-runbook.md) (the production-database version of this problem),
[`llm-fallback-policy.md`](./llm-fallback-policy.md) (LLM usage at inference time — this document covers LLM usage at *build* time)

**Who this is for.** Engineers who are strong at software and new to ML. Every ML concept is
explained where the decision that needs it arrives, rather than assumed. If you already know a
concept, the explanation is a short box you can skip; the decision follows it.

**House convention, inherited from the other specs in this repo:** claims are tagged
**[measured]** (computed from the actual CSV, command given), **[estimate]** (arithmetic from
stated assumptions), or untagged (a design decision, argued in place).

---

## Assumptions

Surfaced up front so a reviewer can kill a wrong one before it costs anything.

1. **Python 3.11+.** The model side is Hugging Face; the pipeline must share preprocessing code
   with the model service (§2.9 of the classifier spec), so it cannot be a different language.
2. **The input is the CSV as it stands.** No `ticket_messages`, no database, no provenance
   reconstruction. Where the runbook's Case A/B/C/D analysis would apply, this pipeline does the
   honest reduced version and says so.
3. **The `services` column is the label source, and it is noisy.** It is a column someone typed,
   not adjudicated ground truth.
4. **LLM calls go through OpenRouter.** One key, model-agnostic, swappable.
5. **No 152-FZ constraint on this work.** The corpus is synthetic and already public in this git
   repository, so sending it to a hosted API is a cost decision, not a legal one. This assumption
   does **not** transfer to production data — see §5.5.
6. **Output format is Parquet.** Not CSV. Reasons in §2.
7. **This pipeline produces training data and an annotation queue. It does not produce a gold
   evaluation set** — that requires humans, and §6.4 explains why no amount of engineering
   substitutes for them.

---

## 0. Objective

### 0.1 What we are building

A deterministic, re-runnable pipeline that turns one raw CSV export into a versioned,
training-ready dataset — plus the two artifacts that make it trustworthy: a **manifest** that
pins exactly what was produced, and a **datasheet** that states what the data can and cannot
support.

```
tickets_export.csv                 →   data/processed/<version>/
  5,013 rows, dirty, PII-bearing        train.parquet / val.parquet / test.parquet
                                        quarantine.parquet     (rows we refused, with reasons)
                                        review/labels.jsonl    (human label-review queue)
                                        review/redaction.jsonl (human redaction-escalation queue)
                                        annotation_queue.jsonl (blind gold-set sampling frame)
                                        manifest.json          (hashes, config, counts, versions)
                                        datasheet.md           (what this dataset does not support)
```

### 0.2 What success looks like

- A teammate clones the repo, runs one command, and gets byte-identical output to yours.
- No sensitive value reaches the training set or a log, and we can *show* the recall number
  behind that claim rather than assert it.
- Human review time is spent on the few hundred rows most likely to be wrong, not on all 4,622.
- Nobody reads a metric off this dataset and believes it describes production, because the
  datasheet makes that impossible to do by accident.

### 0.3 Non-goals

Model training, tokenisation, ONNX export, serving, threshold *fitting* (we specify how
thresholds will be chosen; choosing them needs a trained model), and the production database
ingestion path.

### 0.4 The one idea to take from this document

> **A dataset is not a file. It is a file plus a claim about what the file is evidence for.**

The pipeline's job is to make that claim precise and defensible. Most of what follows is
mechanics in service of that one sentence. And for this particular CSV, the claim is
uncomfortable — see §1.8.

---

## 1. Concepts

The ML principles this pipeline is built on, each one grounded in a number measured from your
actual data. Reproduce any of them with `python -m ticketds.cli report --stage <name>`.

### 1.1 What a supervised dataset actually is

Supervised learning needs pairs: an **input** `x` and a **target** `y`. The model is a function
with adjustable internal numbers (**parameters**); training nudges those numbers until the
function's output on `x` is close to `y` across many examples.

Here:

- `x` = the ticket text — `"[priority: high] {title}\n{description}"`, after redaction (§4).
- `y` = the set of services the ticket concerns, drawn from a fixed taxonomy of 20.

Everything else in the CSV (`id`, `organization_id`, `created_at`, `flags`, `labels`) is
**metadata**: not fed to the model, but essential for splitting, auditing, and slicing results.

> **Why metadata that never reaches the model still matters.** `created_at` is what makes an
> honest split possible (§1.3). `organization_id` is what tells you two tickets aren't
> independent. Throwing away columns because "the model doesn't use them" is one of the more
> common ways to build a dataset you can't evaluate.

### 1.2 Multi-label is not multi-class, and the difference changes everything

**Multi-class:** each example gets exactly one of N labels. Digit recognition — an image is a 7
*or* a 3, never both.

**Multi-label:** each example gets a *set*, possibly empty, possibly several.

Yours is multi-label. **[measured]**

| Services per ticket | Rows |
|---|---|
| 0 (untriaged) | 391 |
| 1 | 1,569 |
| 2 | 1,988 |
| 3 | 1,034 |
| 4 | 31 |

Mean cardinality over labelled rows: **1.90**.

Three consequences that the rest of this spec keeps referring back to:

1. **`y` is a 20-dimensional vector of 0/1, not an integer.** The model has 20 independent
   outputs — think 20 yes/no questions asked in parallel, not one 20-way choice. Concretely, a
   ticket labelled `billing,subscriptions` becomes
   `[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]`.
2. **"Accuracy" is nearly meaningless.** Predict "no" for all 20 services on every ticket and
   you are right on 18.1 of 20 questions per row — **90.5% accuracy** **[estimate]**, from mean
   cardinality 1.90/20 — while predicting nothing at all. This is the **class imbalance** trap:
   with rare positives, the do-nothing baseline scores high on the wrong metric. §1.5 gives the
   metrics that don't lie here.
3. **Splitting is genuinely harder.** Keeping one label's proportions steady across train/val/test
   is easy. Keeping twenty proportions steady *simultaneously*, when they co-occur in structured
   ways, is a real algorithm — §6.2.

### 1.3 Leakage — measured on your data

> **Leakage** is when information that will not exist at prediction time is available during
> training. The model exploits it, scores brilliantly in your evaluation, and disappoints in
> production. It is the single most common cause of ML results that fail to reproduce
> ([Kapoor & Narayanan, *Patterns* 2023](https://arxiv.org/abs/2207.07048)).

The textbook `train_test_split(X, y, shuffle=True)` assumes rows are **independent**. Yours are
not, for three separate reasons — and each is measurable.

**(a) Near-duplicates.** Your corpus contains 8 incident bursts; the largest is 109 tickets on
2025-11-18, filed by ~90 organisations about one API-gateway outage. Different words, same event.
Plus 99 auto-generated monitoring alerts sharing a template, and 26 copy-paste refile clusters.

**[measured]** — character-5-gram Jaccard similarity over all 5,013 rows, candidate pairs via a
bottom-*k* MinHash sketch, exact Jaccard on candidates:

| | Pairs ≥ 0.6 | Pairs ≥ 0.8 | Rows involved (≥0.6) |
|---|---|---|---|
| Raw text | 583 | 429 | 348 |
| After redaction | 599 | 464 | 371 |

And the leakage those pairs cause, at a 70/15/15 split:

| Split strategy | Test rows with a ≥0.6 near-duplicate sitting in train |
|---|---|
| **Random** (`shuffle=True`, seed 0) | **47 / 752 = 6.2%** |
| **Temporal** (sort by `created_at`) | 28 / 752 = 3.7% |

6.2% of your test set would be graded on tickets the model had effectively already seen. Not
catastrophic — but it is free score you did not earn, and on a real corpus with heavier
copy-paste it is much worse.

> **Why the redacted row shows *more* duplicates, and why that is good news.** Two tickets
> describing the same outage differ in the request id, the timestamp, the hostname. Replace those
> with `<UUID>`, `<TS>`, `<HOST>` and the two texts collapse onto each other. Redaction makes
> near-duplicates *visible* to the detector. This is why **dedup runs after redaction** (§3, S3)
> — the gain here is modest (+16 pairs at ≥0.6) but it is free and it points the right way.

**(b) Organisation.** One customer's tickets share vocabulary, product surface, and writing
style. **[measured]** all 184 organisations present in a temporal test split also appear in
train. Perfect group separation is impossible here without discarding most of the data, so §6.3
specifies the compromise and what it costs.

**(c) Time.** Services get added, product surfaces change, phrasing drifts. A random split asks
the model to interpolate among tickets it has neighbours for on both sides in time. Production
asks it to extrapolate forward. Different task, easier task.

**Decision: temporal split, with a gap, with near-duplicate removal across boundaries** (§6).
And per the classifier spec §2.6, we **report both** random and temporal numbers once — the gap
between them is your drift magnitude, and it is a number the product owner should see.

### 1.4 Noisy labels

`services` was typed by support agents (and, in production, partly written by the outgoing
CatBoost model). It is wrong sometimes. **[measured]** the mechanical evidence for that is
already visible in the column:

| Defect | Rows |
|---|---|
| Space after the separator (`"logging, subscriptions"`) | 39 |
| Mixed case (`"Billing"`) | 38 |
| A service repeated inside one row (`"logging,logging"`) | 40 |
| Empty `services` (untriaged) | 391 |

Those are the *typos you can see*. The semantic errors — a ticket about a broken console page
tagged `console-ui` when it should also carry the component that actually failed — are invisible
to any string check. Estimating *that* rate is the entire purpose of §5.2.

> **Why noise is survivable but must still be measured.** Deep models tolerate a surprising
> amount of random label noise; the errors partially cancel. What they do not tolerate is
> **systematic** noise — if agents consistently tag `auth` where `access-control` is correct, the
> model learns that convention faithfully and reproduces it. And every point of label noise in
> the *evaluation* set is a hard ceiling on the score you can measure, because the model is
> graded against a wrong answer.

The empty-services rows deserve their own decision. **391 rows have no label.** Two readings:

- *"No service applies"* — a genuine negative, useful training signal for abstention.
- *"Nobody got round to triaging it"* — an unknown, and training on it as all-zeros teaches the
  model to stay silent on tickets that do have services.

You cannot distinguish these from the CSV. **Decision: quarantine all 391, never train on them,
and route a sample into the annotation queue.** Treating unknowns as negatives is the mistake the
runbook §2.5 calls a positive-unlabelled problem; the cheap correct move is to not guess.

### 1.5 Metrics, and why the choice is a data decision

For one service, with the model saying yes/no:

- **Precision** = of the tickets where we predicted `billing`, what fraction really were? Low
  precision = false alarms.
- **Recall** = of the tickets that really were `billing`, what fraction did we catch? Low recall
  = misses.
- **F1** = their harmonic mean — one number, punishes a model that wins one by sacrificing the
  other.

There are two ways to average across 20 services, and picking the wrong one hides your problem:

- **Micro-F1** pools every ticket-service decision into one pile. Dominated by the frequent
  services. `billing` (853 positives) swamps `terraform-provider` (15).
- **Macro-F1** computes F1 per service, then averages the 20 numbers equally. A service you never
  once predict correctly contributes a hard zero.

**Report both, always, with per-service `n`.** Your product constraint — a wrong service costs
more than a missing one — means precision matters more than recall, which the classifier spec
turns into per-service thresholds (§6.1 there).

**[measured]** the reason this is a *dataset* concern, not just a reporting one:

| Service | Total positives | In a temporal test split (15%) |
|---|---|---|
| `terraform-provider` | 15 | **1** |
| `message-queue` | 31 | **1** |
| `cdn` | 38 | 5 |
| `dns` | 41 | 5 |
| `managed-redis` | 98 | 18 |

With one positive example in test, `terraform-provider` F1 is either 0.0 or 1.0 depending on a
single ticket. That is not a measurement; it is a coin flip with a decimal point. **The pipeline
must therefore emit per-service test counts and mark services below a floor as unmeasurable**
(§6.5). Reporting a macro-F1 that silently averages in four coin flips is how a project ships a
number nobody can defend.

### 1.6 Thresholds are a dataset decision

A trained classifier does not output "yes" — it outputs a score between 0 and 1 per service. Turning
20 scores into a predicted set requires 20 cutoffs, and **you choose them**.

The rule that keeps this honest: **thresholds are fitted on validation, never on test.** If you
tune cutoffs on test, your reported number is the best of many attempts on the same data — an
optimistic estimate of a process you cannot repeat in production.

This is why the pipeline produces **three** splits, not two:

| Split | Purpose | Touched how often |
|---|---|---|
| **train** | fit model parameters | continuously |
| **validation** | choose thresholds, hyperparameters, checkpoints | many times |
| **test** | one honest final estimate | as rarely as possible |

> Each time you look at test and change something, you leak a little information into your
> design. Test is a budget you spend, not a dashboard you watch.

### 1.7 Freeze the data, iterate the model

The **data-centric** principle: when data and model both change between two experiments, you
cannot attribute the difference to either. So the dataset gets a **version and a content hash**,
and every training run records which one it consumed.

Concretely (§7): `manifest.json` carries the SHA-256 of the input CSV, the resolved config, every
tool version, the row counts at every stage, and the hash of each output file. Re-running with the
same inputs reproduces the same hashes, or the pipeline is broken.

> **This is why LLM stages cache to disk (§5.4).** An LLM is non-deterministic. Left uncached it
> would make your dataset unreproducible — the one property this section exists to protect.

### 1.8 The uncomfortable part: what this fixture can and cannot prove

`data/raw/README.md` is explicit: labels were assigned by the generator that wrote the text.
There is no blind annotation, no inter-annotator agreement, no adjudication.

**So any F1 you compute on this CSV describes the generator, not the world.**

That does not make the work pointless — it makes it *pipeline* work rather than *modelling* work.
What this fixture genuinely exercises:

| Provable here | Not provable here |
|---|---|
| The pipeline runs, is deterministic, is reproducible | That the model is any good |
| Redaction precision, and recall against a span set we label ourselves | Redaction recall against real customer text (§4.6) |
| Splitting is leak-free by our own checks | That the split reflects production drift |
| Label-review triage flags rows a human agrees are wrong | The true label error rate of production |
| Class balance, cardinality, length, timing distributions | That those match production |

**Requirement: `datasheet.md` states this in the first paragraph, generated by the pipeline, not
written by hand.** A teammate who picks up `test.parquet` six months from now must hit this
warning before they hit a number.

---

## 2. Tech stack

What the industry actually reaches for, why, and what we rejected. Versions are floors, pinned
exactly in `requirements.txt` (§7.3).

| Concern | Choice | Why | Rejected |
|---|---|---|---|
| Dataframes | **Polars ≥ 1.0** | Fast, strict typing, lazy evaluation, and — decisively — it does not silently coerce. pandas turning an all-digit id column into `float64` and rendering `4111111111111111` as `4.111111111111111e15` is a real corruption mode for this data. | pandas (implicit dtype coercion, index semantics nobody needs here); DuckDB (excellent for the SQL-shaped audit queries, extra dependency for what Polars already does) |
| On-disk format | **Parquet** | Columnar, compressed, *typed*. A CSV round-trip loses the distinction between empty string and null — and 391 of your rows depend on exactly that. | CSV (untyped, the disease we are curing), JSONL (fine for queues, wasteful for 5k×20 columns) |
| Schema validation | **Pandera** (Polars backend) | Schema as code, checked at every stage boundary, failures land as data rather than exceptions | Great Expectations (heavier: its own project structure, docs site, and store config for what is one schema file here); hand-rolled asserts (nobody maintains them) |
| Config | **YAML + Pydantic v2** | Human-diffable file, typed and validated at load, resolved config serialised into the manifest | argparse-only (unversionable), TOML (fine, less familiar to the DS side) |
| Near-duplicates | **`datasketch`** MinHash + LSH | Standard, scales past this fixture, tunable threshold | Exact hashing only (**[measured]** catches just 6 rows here); embedding similarity (needs the model we haven't trained — a circular dependency) |
| Multi-label splitting | **`iterative-stratification`** | Implements Sechidis et al. 2011 iterative stratification; the only maintained option with a scikit-learn API | `scikit-multilearn` (effectively unmaintained); `sklearn.train_test_split(stratify=…)` (single-label only — cannot express 20 simultaneous constraints) |
| PII detection | **Custom rules + Presidio for the global entities** | See §4.2 — **verified: Presidio ships no Russian recognizers at all** | Presidio alone (fails on ИНН/СНИЛС/ОГРН/КПП/БИК/р-с); LLM alone (§5.2: cannot be the gate) |
| LLM access | **OpenRouter**, via the OpenAI-compatible HTTP surface | One key, model-agnostic, swap models without touching code | Direct vendor SDKs (locks the pipeline to one provider) |
| Tests | **pytest** + **hypothesis** | Property tests are the right tool for redaction invariants (§9) | unittest |
| Orchestration | **A plain Python CLI** | 9 stages, one machine, minutes of runtime. | Airflow/Dagster/Prefect (a scheduler for a job with no schedule); DVC (revisit when data volume outgrows git — noted in §11) |

> **On `iterative-stratification`:** version 0.1.9, and its metadata still advertises Python
> 3.4–3.9. It works on 3.11 — the algorithm is ~200 lines of NumPy — but it is quiet upstream.
> Pin it, and keep §6.2's fallback (a simple grouped temporal split) as the exit if it breaks.
> Flagging maintenance risk on a dependency is part of choosing it.

---

## 3. Pipeline architecture

Nine stages. Each reads the previous stage's output, writes its own, and appends to the manifest.
Any stage can run alone against a cached predecessor — which is what makes iterating on redaction
rules bearable.

```
S0 ingest ──→ S1 normalize-labels ──→ S2 redact ──→ S3 dedup ──→ S4 enrich
                                                                     │
                     ┌───────────────────────────────────────────────┘
                     ▼
              S5 llm-label-review ──┐
              S6 llm-redaction-audit┤──→ S7 split ──→ S8 freeze
                                    │
                                (queues for humans)
```

**Design rules that hold for every stage:**

1. **Never delete a row — quarantine it.** Rejected rows go to `quarantine.parquet` with
   `quarantine_reason` and the stage that rejected them. Silent row loss is how a pipeline
   develops a 12% shortfall nobody notices.
2. **Every stage is a pure function of (input frame, config).** No wall-clock, no RNG without a
   seed from config, no network except S5/S6 — which cache to disk so re-runs are offline.
3. **Row counts in = row counts out + quarantined.** Asserted, and written to the manifest.

### S0 — `ingest`

**Purpose:** get the CSV into a typed frame, and refuse anything malformed.

The CSV is genuinely nasty: **[measured]** 1,771 descriptions contain embedded newlines, 805
contain escaped quotes, 4,307 contain commas inside prose. Do not hand-roll a parser; do not
`split(',')`. Use a real RFC 4180 reader with `quote_char='"'`.

**Mojibake repair — and why it belongs in S0 rather than being tidied up later.**

**[measured]** the file contains **9 rows of mojibake**, and they are *not* what you would guess.
The file is valid UTF-8 throughout: reading with `encoding_errors='replace'` and flagging U+FFFD
finds **zero rows**. What actually happened is that Windows-1251 bytes were decoded as Latin-1
*before* the file was written, so the corruption is faithfully encoded as legal characters:

```
stored:     "Ïàíîâ Ñåðãåé Âèêòîðîâè÷, äåéñòâóåò íà îñíîâàíèè óñòàâà"   (TICKET-16991)
recovered:  "Панов Сергей Викторович, действует на основании устава"
            via  text.encode('latin-1').decode('cp1251')
```

Detect with a run of ≥4 Latin-1 supplement characters (`[À-ÿ]{4,}`), repair by round-tripping,
and keep both the original and the repaired text.

**This is a redaction bypass, not a cosmetic defect.** That recovered string is a person's full
name. As stored, no Russian NER model tags it, no keyword rule fires on `паспорт` or `р/с`,
because the keywords are mangled too. **[measured]** the 9 rows include a person's name, a
forwarded email header, a `.env` paste described as `боевой` (production), and a document-requisites
block. Every one of them would sail through §4's detectors untouched.

**Therefore: repair runs in S0, before redaction in S2.** Rows that fail to round-trip cleanly are
quarantined rather than passed through — an un-repairable encoding is a row we cannot claim to
have redacted.

- **In:** `data/raw/tickets_export.csv`
- **Out:** `stage/s0_ingest.parquet` — all 10 source columns as `Utf8`, plus `row_index`
- **Quarantine:** column count mismatch, missing `id`, duplicate `id`, unparseable `created_at`
- **Acceptance:** 5,013 rows in, 5,013 accounted for; `id` unique; `updated_at >= created_at`

### S1 — `normalize-labels`

**Purpose:** turn the free-text `services` column into a validated set, and separate "no label"
from "empty label".

```
"logging, Subscriptions,logging"  →  {"logging", "subscriptions"}
""                                →  null   (NOT the empty set — see §1.4)
```

Procedure: split on `,` → strip → lowercase → drop empties → deduplicate → sort →
**validate every member against the 20-service taxonomy**, which lives in `configs/taxonomy.yaml`
as data, not as a constant in code. An unknown service is a quarantine, never a silent drop: a
service being renamed upstream is exactly the event this check exists to catch.

**[measured]** expected effect: 39 whitespace fixes, 38 case fixes, 40 intra-row dedups, 391 rows
to `label_status = "unlabelled"`, 0 unknown services.

- **Out:** adds `services_norm` (`List[Utf8]`, sorted), `n_services`, `label_status ∈
  {labelled, unlabelled}`, and `label_dirty_flags` recording which normalisations fired
- **Acceptance:** every `services_norm` member ∈ taxonomy; sorted; no duplicates within a row;
  labelled + unlabelled = 5,013

> **Why record which normalisations fired.** `label_dirty_flags` is a free, cheap prior on
> sloppiness. A row whose label needed three mechanical fixes is a better-than-random bet for
> *semantic* error too. S5 uses it as a triage signal, which costs nothing to collect now and is
> unrecoverable later.

### S2 — `redact`

Specified in full in §4. Contract only here:

- **In:** `title`, `description`
- **Out:** `title_redacted`, `description_redacted`, plus a sidecar
  `stage/s2_spans.parquet` with one row per detected span
  (`row_index`, `field`, `start`, `end`, `entity_type`, `placeholder`, `detector`, `confidence`)
- **Acceptance:** idempotent (`redact(redact(x)) == redact(x)`); every span replaced by a
  placeholder from the closed vocabulary in Appendix A; the span sidecar reconstructs the
  redacted text from the original exactly

> **Why a span sidecar rather than just the redacted string.** You need to audit *what* was
> removed without re-running detection, measure precision per entity type, and let S6 reason
> about coverage. Throwing away the spans makes every later question require a full re-run.

### S3 — `dedup`

**Purpose:** identify near-duplicate clusters so §6 can keep them on one side of a split
boundary.

Runs **after** S2 — evidenced in §1.3.

1. **Exact:** SHA-256 of `title_redacted + "\n" + description_redacted`. **[measured]** 6 rows.
2. **Near:** MinHash (128 permutations) over character 5-grams of the redacted text, LSH banding
   for candidates, exact Jaccard to confirm. Two thresholds, both recorded:
   - `≥ 0.80` → **[measured]** 464 pairs — same-text clusters; drop all but the earliest from
     train, and never allow across a split boundary
   - `≥ 0.60` → **[measured]** 599 pairs, 371 rows — flag as a cluster, keep, but never split a
     cluster across the train/test boundary
3. Assign `dup_cluster_id` via connected components over the ≥0.60 graph.

- **Out:** adds `text_sha256`, `dup_cluster_id`, `dup_max_jaccard`, `is_exact_dup`
- **Acceptance:** clustering is deterministic given the seed; cluster count and size histogram in
  the manifest

> **What MinHash structurally cannot catch, and why we say so out loud.** MinHash measures
> *lexical* overlap. Two tickets describing the same Postgres connection-pool exhaustion in
> completely different words score 0.2–0.5 and will not cluster. `data/raw/README.md` says such
> clusters were planted deliberately. Semantic dedup needs embeddings, which needs a model, which
> we do not have yet. **Decision: accept the gap, document it in the datasheet, revisit after the
> first model exists.** Naming a known blind spot beats pretending the number is complete.

### S4 — `enrich`

**Purpose:** compute everything the splitter and the slice-based evaluation need.

| Field | How | Note |
|---|---|---|
| `lang` | `py3langid` on redacted text, `{ru, en, other}` + confidence | see the warning below |
| `char_len`, `est_tokens` | length of the assembled model input | **[measured]** description p50 222, p90 688, p99 1,040, max 1,691 chars — feeds the `max_len` choice in classifier spec §4.6 |
| `burst_id` | `created_at` bucketed to 6h × dominant service; buckets above the 99th-percentile count | **[measured]** 8 bursts exist; largest 109 tickets on 2025-11-18 |
| `model_input` | `"[priority: {p}] {title}\n{description}"`, redacted | must be byte-identical to what the model service builds at inference |
| `has_payload` | any span from S2, or a pasted-block heuristic | ~30% of rows per the README |

> **Language detection will disagree with reality, and the disagreement is the finding.**
> **[measured]** a naive Cyrillic-character-ratio rule (>0.8 → ru, <0.2 → en, else mixed) labels
> **1,123 rows as "mixed"**, where the generator recorded only **237** as genuinely code-switched.
> A Russian ticket quoting an English stack trace is *Russian prose*, but character ratios cannot
> tell. This matters because classifier spec §6.3 makes language a first-class evaluation slice,
> and a slice built on a 4.7× over-counting rule measures nothing. **Decision:** use a real langid
> model over the *prose only* — strip pasted payload blocks (S2 already located them) before
> detecting. Record both the naive and the payload-stripped verdict, and put the disagreement rate
> in the datasheet.

- **Acceptance:** `model_input` is non-empty for every non-quarantined row; the assembly function
  is imported from the shared preprocessing module, not reimplemented

### S5, S6 — LLM stages

Specified in §5.

### S7 — `split`

Specified in §6.

### S8 — `freeze`

**Purpose:** write outputs, and make the dataset self-describing.

Writes the three splits, `quarantine.parquet`, the review queues, and:

- **`manifest.json`** — input CSV SHA-256; resolved config; git commit; `platform.python_version`
  and every pinned package version; per-stage row counts in/out/quarantined; SHA-256 of each
  output file; LLM cache hit/miss counts and model ids; wall-clock per stage
- **`datasheet.md`** — generated, following the *Datasheets for Datasets*
  ([Gebru et al., 2018](https://arxiv.org/abs/1803.09010)) structure: motivation, composition,
  collection, preprocessing, uses, distribution, maintenance. Opens with §1.8's warning. Contains
  the class-balance table, the cardinality histogram, the RU:EN ratio, the token-length
  distribution, the per-service test counts with unmeasurable services flagged, and the redaction
  recall estimate with its confidence interval.

- **Acceptance:** re-running the whole pipeline with an unchanged config and a warm LLM cache
  reproduces every output hash. This is asserted by a test (§9), not by hope.

---

## 4. Redaction

The largest section, because it is the part with a hard failure mode: a missed secret in training
data is an incident, and a model can memorise and later emit what it was trained on.

### 4.1 Two jobs that share one mechanism

They are constantly conflated. Keep them separate, configure them separately, measure them
separately:

| | **Redaction** | **Normalisation** |
|---|---|---|
| Why | Safety and privacy | Token budget |
| Failure mode | A leak — an incident | A slightly longer sequence |
| Example | `ivan@example.ru` → `<EMAIL>` | `2025-11-18T04:12:33Z` → `<TS>` |
| Can we turn it off? | No | Yes, it is an ablation |

Classifier spec §4.6 estimates that normalising high-entropy zero-signal spans recovers 10–30% of
the token budget. That is a real win — a shorter sequence is a faster model — but it is an
*optimisation*, and optimisations get ablated. Safety does not. `configs/dataset.yaml` therefore
carries `redaction.enabled: true` (not overridable) and `normalisation.enabled: true` (an
experiment knob).

### 4.2 Detector layers

Four layers, cheapest first, each with a different competence:

| Layer | Catches | Precision | Recall |
|---|---|---|---|
| **1. Structural rules** | Emails, URLs, IPs, UUIDs, JWTs, AWS keys, `sk_`/`whsec_` prefixes, PEM headers, connection strings, `Authorization:` headers | Very high | High for known shapes, zero for unknown ones |
| **2. Checksum-validated rules** | Cards (Luhn), ИНН/СНИЛС/ОГРН (check digits), IBAN (mod-97) | Very high | **Deliberately not used as a filter — see §4.4** |
| **3. Context rules** | Keyword-anchored: `паспорт`, `СНИЛС`, `р/с`, `БИК`, `дата рожд.`, `IBAN` followed by a plausible value | Medium | Catches formats the structural layer misses |
| **4. NER** | Person names, organisation names, addresses — open classes with no format at all | Medium | The only layer that can catch these |

**On layer 4.** [Microsoft Presidio](https://microsoft.github.io/presidio/) is the industry
default for this. Its global recognizers (`EMAIL_ADDRESS`, `CREDIT_CARD`, `IBAN_CODE`,
`IP_ADDRESS`, `PERSON`, `LOCATION`, `PHONE_NUMBER`, `URL`, `UUID`, `CRYPTO`, `MAC_ADDRESS`,
`DATE_TIME`, `NRP`, `MEDICAL_LICENSE`) are worth having.

**Verified against Presidio's `supported_entities.md`: it ships country-specific recognizers for
the USA, UK, Spain, Italy, Poland, Singapore, Australia, India, Finland, Korea, Nigeria, the
Philippines, Canada, Sweden, South Africa, Thailand, Turkey and Germany — and none for Russia.**
There is no ИНН, СНИЛС, ОГРН, КПП, БИК, or Russian-passport recognizer. Every Russian entity in
your corpus needs a custom recognizer that you write and you test. Do not assume the library
covers you because it is the standard choice; it covers the entities its contributors needed.

For Russian `PERSON`/`LOCATION`, Presidio needs a Russian spaCy pipeline
(`ru_core_news_lg`) wired through its NLP engine, or a transformer NER model. **This is the
lowest-confidence, highest-cost layer, and §5.2 exists mostly because of it.**

### 4.3 What the rules actually catch here

**[measured]** — a first-pass rule set over all 5,013 rows, to calibrate expectations *before*
writing production code:

| Entity | Instances | Rows |
|---|---|---|
| Email | 566 | 468 |
| URL | 568 | 542 |
| Phone (RU `+7`, US `+1 555`) | 286 | 267 |
| IPv4 | 1,218 | 366 |
| `Authorization:` / bearer | 194 | 190 |
| AWS access key id | 125 | 124 |
| DB connection string with inline password | 98 | 98 |
| ИНН | 89 | 86 |
| КПП | 52 | 51 |
| Stripe-style `sk_`/`whsec_` | 51 | 51 |
| Card-like digit run | 39 | 39 |
| р/с, к/с | 31 | 16 |
| ОГРН | 19 | 19 |
| БИК | 15 | 15 |
| СНИЛС | 13 | 13 |
| JWT | 9 | 9 |
| PEM private key header | 4 | 4 |
| Partial self-redaction (`***`, `REDACTED`) | 335 | 168 |
| **Rows with ≥1 sensitive hit** | | **1,226 (24.5%)** |

Now compare against what `data/raw/README.md` says is actually in the file:

| | README says | Rules found | Gap |
|---|---|---|---|
| Rows carrying a pasted payload | 1,500 (29.9%) | 1,226 | **274 rows** |
| Card-number strings | 89 | 39 | **50** |
| Phone numbers | 287 | 286 | 1 |
| Emails | 566 | 566 | 0 |

**Three findings, and they are the point of this section:**

1. **The card gap is a format gap.** The 50 missing are masked or partial forms —
   `4111 11** **** 1111`, `карта ****1111`. A digit-run regex cannot match a string containing
   `*`. Any masked-card pattern must be an explicit rule, and it will look wrong until you
   remember that a partial PAN is still cardholder data.
2. **The 274-row gap is mostly not PII** — pasted k8s manifests, nginx configs, stack traces. They
   carry internal infrastructure identifiers, which is a *different* sensitivity class, and the
   reason `<HOST>` and `<PATH>` exist in Appendix A.
3. **A plausible IBAN regex found zero of the IBANs that are actually there.** `[A-Z]{2}\d{2}[A-Z0-9]{11,30}`
   matched **0**; the same pattern allowing spaces between groups matched **16**
   (`FR00 3000 3000 0000 0000 0000`). Nobody types an IBAN unspaced. The same
   trap hit passports: a `\d{2}\s?\d{2}\s?\d{6}` pattern matched 0, while the keyword `паспорт`
   appears 14 times with the values nearby in a shape the pattern did not anticipate.

> **The transferable lesson.** A regex that returns zero matches feels like "this entity isn't in
> the data." It usually means your pattern is wrong. **Every detector that finds zero must fail
> the build until a human either fixes it or records why zero is genuinely expected.** This is a
> test (§9), not a convention.

### 4.4 The Luhn trap

Card numbers have a check digit; the Luhn algorithm validates it. The obvious engineering move is
to validate before redacting, killing false positives on order numbers and ОГРНs.

**On this corpus that move is catastrophic, and the failure is invisible.**

`data/raw/README.md`: card values are the published test PANs, and a Luhn sweep guarantees no
*other* 13–19 digit run in the file passes. Meanwhile ИНН, ОГРН and СНИЛС were built with
**deliberately invalid check digits** so they cannot match real people or companies.

**[measured]** of 39 card-like digit runs, **6 pass Luhn**.

So:

- Filter cards on Luhn → you redact 6 of 39 and score 100% precision on a fixture built to reward
  exactly that.
- Filter ИНН/СНИЛС/ОГРН on check digits → **recall goes to zero**. Every one of them is
  arithmetically invalid by construction.

**Decision: checksums adjust confidence, they never gate redaction.**

```
checksum passes  →  entity_type=CARD, confidence=high   → redact
checksum fails   →  entity_type=CARD, confidence=low    → redact anyway, flag for S6
```

Which is also correct on real data, for a plainer reason: **a customer who mistypes one digit of
their card has still pasted their card number into your ticket.** Recall-first is right on the
fixture and right in production. Getting there via the fixture is luck; getting there via the
argument is engineering.

### 4.5 Placeholder design

Full vocabulary in Appendix A. Three decisions worth arguing:

**(a) Semantic placeholders, not a single `<REDACTED>`.** The entity *type* is signal.
"Customer pasted a payment decline dump" is evidence for `billing`; "customer pasted a k8s
manifest" is evidence for `compute`. Collapsing both to `<REDACTED>` throws away a feature that
correlates with the label you are trying to predict.

**(b) Coarse in text, fine in the sidecar.** Twelve placeholders reach the text. The detector's
finer type (`INN` vs `KPP` vs `OGRN`, all → `<ORG_ID>`) is kept in `s2_spans.parquet`. The model
does not benefit from distinguishing ИНН from КПП — but an auditor asking "did we ever miss an
ОГРН" absolutely does. Text carries what the model needs; the sidecar carries what the auditor
needs.

**(c) Placeholders are part of the model's input contract.** Per classifier spec §2.9, **the same
redaction must run at inference or you get train/serve skew** — a model trained on `<EMAIL>` and
served raw addresses at runtime sees a distribution it never learned. So:

- The redactor is a **library** (`ticketds.redact`), not a pipeline stage. The pipeline calls it;
  the model service imports the same module at the same version.
- Its rules live in `configs/redaction.yaml`, and that file's hash goes in `preprocess.json`
  (classifier spec §9.1) and into the manifest.
- Changing a redaction rule **changes the dataset version**. Not negotiable — it changes what the
  model sees.

> This is the one place where the "one capability, one spec" boundary bends: the redactor outlives
> this pipeline. It is specified here because here is where it is built and measured, but it is
> owned jointly with the model service.

### 4.6 Measuring redaction — the part that is actually hard

You cannot compute recall without knowing the right answer. Nothing in the CSV tells you where
the sensitive spans are. So we make a small ground truth:

**The span gold set.** 200 rows, hand-annotated for every sensitive span, stored at
`tests/fixtures/span_gold.jsonl`. Sampling is deliberately **not** random — random sampling of
5,013 rows where ~25% carry anything would spend most of the budget on rows with nothing in them:

| Stratum | Rows | Why |
|---|---|---|
| Rows with ≥1 detected span | 80 | measures precision, catches over-redaction |
| Rows with a payload but **no** detected span | 60 | **the highest-value stratum — this is where misses live** |
| Random rows with no payload | 40 | catches false positives on ordinary prose |
| Rows with partial self-redaction | 20 | **[measured]** 168 such rows; the format most likely to break a detector |

**[estimate]** ~200 rows × ~3 min = **10 person-hours**, once. Every miss it finds becomes a
regression test.

**Metrics, reported per entity type:**

- **Span recall** — fraction of gold spans overlapping a detected span. **The gate.**
- **Span precision** — fraction of detected spans that are genuinely sensitive.
- **Over-redaction rate** — fraction of *characters* replaced. **[measured]** baseline needed; a
  sharp rise means a rule went greedy and is eating signal.

**Gates:**

| Metric | Threshold | On failure |
|---|---|---|
| Span recall, `SECRET`/`CARD`/`BANK_ID`/`GOV_ID` | **1.00** on the gold set | Build fails |
| Span recall, all other types | ≥ 0.95 | Build fails |
| Span precision, overall | ≥ 0.90 | Warn; investigate over-redaction |
| Detectors returning zero matches | 0 | Build fails (§4.3) |

> **And the honest caveat, which belongs in the datasheet.** A grep for `EXAMPLE` or `DO-NOT-USE`
> hits **[measured] 311 rows** of this corpus. The fixture's own README says a redaction pipeline
> tested only on this file will look better than it performs on production text. Our gold set is
> drawn from the same synthetic distribution, so it inherits that optimism. **These numbers are a
> floor on quality, not an estimate of production recall.** State it; do not let a green check
> mark imply otherwise.

---

## 5. LLM stages

Both stages run through OpenRouter. They exist for **opposite** reasons, and the most valuable
thing in this section is why they must not be built the same way.

| | **S5 — label review** | **S6 — redaction audit** |
|---|---|---|
| Problem type | **Sampling** — find the wrong rows cheaply | **Recall** — find what the rules missed |
| If the LLM is wrong | One noisy label among thousands. Training barely notices (§1.4) | A missed secret reaches training data. Incident |
| Is the LLM the decision-maker? | It ranks; a human decides on the top of the queue | **Never.** It widens recall; a deterministic rule redacts |
| Success metric | Human-hours saved per label corrected | Misses found per 100 rows audited |
| Failure tolerance | High | Zero on `SECRET`/`CARD`/`BANK_ID`/`GOV_ID` |

> **The failure this table prevents.** Benchmark an LLM judge on label agreement, get 94%, and
> quietly start trusting it for PII too. 94% over 5,013 rows is ~300 misses. An LLM that is
> excellent at one of these jobs tells you nothing about its fitness for the other.

### 5.1 S5 — label review

**Goal:** replace "a human reads 4,622 rows" with "a human reads the ~400 most likely to be
wrong," and *quantify* what that trade cost.

**Two passes, because they answer different questions.**

**Pass A — measurement (500 rows).** Random sample of labelled rows. The LLM proposes a service
set **blind** — it never sees the stored `services`. Compare to stored. This gives an *unbiased*
estimate of the label disagreement rate, with a bootstrap confidence interval.

> **Why blind, and why random.** Show the model the existing label and you measure whether it can
> be talked into agreeing — the same **anchoring** effect that makes classifier spec §2.2 forbid
> showing annotators the stored value. And the sample must be random: sampling rows that look
> suspicious measures your suspicion, not the corpus.

**Pass B — triage (all 5,013 rows).** Same blind prompt, full sweep. Produces a
`disagreement_score` per row, combining:

- set difference between proposed and stored labels (weighted: a *missing* service is a milder
  signal than a *contradicting* one)
- the LLM's own confidence
- `label_dirty_flags` from S1 — mechanical sloppiness as a free prior
- `n_services == 0` (all 391 unlabelled rows enter the queue by construction)

Rows are ranked; the top **K = 400** (config) are written to `review/labels.jsonl` with the
ticket, both label sets, and the model's stated evidence.

**Prompt design:**

- **Constrained adjudication**, per `llm-fallback-policy.md` §4.1: the output space is the closed
  20-service taxonomy, not free text.
- **Structured output** with a JSON schema — `service` constrained by `enum`, plus `accept`
  (boolean) and `evidence` (a quoted span from the ticket). The shape is guaranteed, not hoped for.
- **Requiring `evidence` is a quality lever, not decoration.** A model forced to quote the span
  that justifies `billing` cannot hand-wave, and a human reviewing the queue reads the quote
  rather than re-reading the ticket. It is also the cheapest hallucination check available: if
  the quoted text is not in the ticket, discard the verdict automatically.
- The taxonomy definitions and few-shot examples go in a **stable prefix** so prompt caching
  engages (§5.4). Per-ticket content goes last.
- The prompt is versioned at `prompts/label_review.v1.md`; its hash goes in the manifest.
  **A prompt change is a dataset change.**

**Acceptance gates:**

| Metric | Gate | Meaning |
|---|---|---|
| Queue precision | ≥ 0.30 | ≥30% of the 400 flagged rows get corrected by the human. Below that the ranking is noise and the stage is wasting review time |
| Pass-A agreement | reported, not gated | The label-noise estimate. It goes in the datasheet |
| Evidence-span validity | ≥ 0.95 | Quoted evidence actually appears in the ticket |

**[estimate] the saving, stated as arithmetic so it can be checked:** reviewing all 4,622
labelled rows at ~60 s each is **77 person-hours**. Reviewing 400 ranked rows plus the 500-row
measurement sample is **~15 person-hours** — an **81% reduction**, at the cost of missing errors
that ranked below 400. Pass A is what lets you put a number on what you missed instead of
pretending you missed nothing.

> **What this stage must never do: overwrite a label automatically.** An LLM correcting labels
> unsupervised means training a model on another model's opinions — distillation you did not
> choose, with no measurement of the teacher. It also destroys the independence of any evaluation
> set built from those labels. The LLM ranks. A human decides. This is the guardrail
> `llm-fallback-policy.md` §1 calls "keeping it from re-contaminating the dataset."

### 5.2 S6 — redaction audit

**Goal:** estimate what layers 1–4 (§4.2) missed, and turn every miss into a permanent rule.

**Sampling — deliberately not random**, for the same reason as the span gold set:

| Stratum | Rows | Why |
|---|---|---|
| Payload present, **no** span detected | ~274 | **[estimate]**, from the 1,500 − 1,226 gap in §4.3. The exact population where a miss hides; recomputed for real once S4 sets `has_payload` |
| Detected spans, low confidence (checksum failed, §4.4) | ~200 | precision check on the recall-first decision |
| Partial self-redaction | 168 | **[measured]** the format most likely to defeat a rule |
| Random control | 150 | catches misses in ordinary prose |
| | **~790** | |

**The prompt asks one question**, and its narrowness is what makes it work: *"Here is a support
ticket in which an automated system has already replaced sensitive values with placeholders. List
any remaining span that is a personal, financial, corporate, or authentication identifier."*
Structured output: a list of `{start, end, text, entity_type, why}`.

**What happens to the output — the whole design in four steps:**

1. Every reported span is **verified programmatically** — does the offset actually contain the
   quoted text? Hallucinated spans are dropped automatically.
2. Surviving spans go to `review/redaction.jsonl` for a **human**.
3. Confirmed misses become **new rules plus regression test cases** in
   `tests/fixtures/redaction_regressions.jsonl`.
4. The dataset version is **rebuilt** with the fixed rules.

**The LLM never edits text.** It nominates; rules redact. The reason is reproducibility, not
distrust: a non-deterministic component in the redaction path makes §1.7's guarantee impossible.
A rule that fires on every future run is worth more than a correct one-off replacement.

**Gate:** if the audit finds a miss in `SECRET`, `CARD`, `BANK_ID`, or `GOV_ID`, the dataset
version is **not publishable** until a rule covers it and the regression test passes.

### 5.3 The data-egress question

Even with 152-FZ off the table (Assumption 5), the sequencing matters and generalises:

- **S5 sends redacted text.** It only needs the topic. Sending raw text would be an unforced leak.
- **S6 must see the original**, because its job is finding what redaction missed. You cannot audit
  a redactor using only its output.

That is a real, unavoidable tension. Our resolution: **S6 sends only its ~790-row sample, only the
raw text, logged explicitly, and gated behind `llm.allow_raw_text: true`** — a flag that must be
set deliberately and that the manifest records for every run.

On this synthetic corpus this is theatre. **On production data it is the control that makes the
stage legal or not**, and building the switch now costs nothing while retrofitting it later means
re-auditing every historical run. Design the boundary while it is free.

### 5.4 Determinism, caching, and cost

**Cache to disk, keyed by `sha256(model_id + prompt_version + rendered_prompt + params)`.**
`data/cache/llm/{key}.json`. Consequences:

- Re-running the pipeline is **free and offline** — the property §1.7 requires.
- Changing the prompt or the model invalidates exactly the affected calls, and nothing else.
- The manifest records model id, prompt hash, and cache hit/miss counts, so "which model built
  this dataset" is always answerable.

Set `temperature = 0`. This reduces variation; it does not eliminate it. **Determinism comes from
the cache, not from the sampling parameters** — do not let a low temperature convince you the
stage is reproducible on its own.

**Prompt caching.** Put the taxonomy and few-shot block in a stable prefix. One trap, already
documented in `llm-fallback-policy.md` §4.3: **the minimum cacheable prefix is model-dependent and
not monotonic** — 512 tokens on Claude Opus 5, 1,024 on Sonnet 5, but **4,096 on Haiku 4.5**. Our
prefix is ~2,000 tokens, so it caches on Opus 5 and Sonnet 5 and **silently does not cache on
Haiku 4.5** — you just pay full price on every call with no error to tell you. Verify by reading
`cache_read_input_tokens` off the response; if it is zero across repeated calls, caching is not
working.

**[estimate] cost — S5 full sweep, 5,013 calls**, assuming a 2,000-token cached prefix, ~350
fresh input tokens and ~200 output tokens per ticket, at first-party Anthropic list prices:

| Model | Input / Output per MTok | Prefix caches? | Estimated total |
|---|---|---|---|
| **Claude Opus 5** (`claude-opus-5`) | $5.00 / $25.00 | yes | **~$39** |
| Claude Sonnet 5 (`claude-sonnet-5`) | $3.00 / $15.00 | yes | ~$23 |
| Claude Sonnet 5 — intro pricing to 2026-08-31 | $2.00 / $10.00 | yes | ~$16 |
| Claude Haiku 4.5 (`claude-haiku-4-5`) | $1.00 / $5.00 | **no — 4,096-token minimum** | ~$17 |

S6 adds **~$6** at Opus 5 (≈790 calls, larger inputs, smaller outputs).

Two things worth noticing in that table. **Haiku 4.5 is not meaningfully cheaper than Sonnet 5 at
intro pricing**, because losing prompt caching costs more than its lower per-token rate saves — a
result you would never guess from the price list alone. And **the whole spread is $16 to $45 for a
one-time build.** Optimising this is not worth an engineer-hour.

**Decision: default `claude-opus-5`**, configurable via `llm.model`. Cost is not the binding
constraint; the queue precision gate in §5.1 is, and a stronger model clears it more easily. Route
through OpenRouter, which passes provider pricing through and adds its own fee on credit
purchases — check your current account rates and budget ~10% headroom over the table above.

**Operational requirements:** exponential backoff with jitter on 429/5xx; a hard `llm.max_calls`
ceiling so a bug cannot spend unbounded money; a `--dry-run` that renders prompts and prints the
projected token count and cost without calling anything.

---

## 6. Splits

### 6.1 Strategy

**Temporal, gapped, cluster-aware, group-reported.** Per classifier spec §2.6, and now with
measured numbers.

```
sort by created_at
train = oldest 70%   →  [7-day gap]  →  val = next 15%  →  [7-day gap]  →  test = newest 15%
```

**[measured]** over the 4,622 labelled rows: boundaries land at **2026-04-06** and **2026-06-04**,
producing **train 3,237 / val 621 / test 627**, with **137 rows (3.0%) discarded into the gaps**.

> **What the gap buys.** An incident burst produces many near-identical tickets within hours. A
> boundary falling inside a burst puts siblings on both sides. Seven days of silence costs 3% of
> the data and removes the single most likely leak. Discarded gap rows are quarantined with
> `quarantine_reason = "split_gap"`, not deleted — §3 rule 1.

**Cluster rule:** any `dup_cluster_id` from S3 spanning a boundary is assigned wholesale to the
**earlier** split, and its later members are dropped from the later split. Rationale: keeping a
duplicate in train is harmless; keeping one in test inflates the score.

**Also produce a random split**, purely as the drift diagnostic classifier spec §2.6 requires.
Never train on it. **[measured]** the leakage difference is already known — 6.2% versus 3.7% of
test rows having a near-duplicate in train (§1.3).

### 6.2 Multi-label stratification, and why we mostly cannot use it

The ideal is that every service appears in train/val/test at its corpus-wide rate. For a single
label, `stratify=y` does it. For 20 simultaneous labels it is a genuine optimisation problem —
`iterative-stratification` implements the Sechidis et al. (2011) greedy algorithm: repeatedly
place the example carrying the *rarest* still-under-allocated label.

**But stratification and temporal ordering are in direct conflict.** Stratification needs freedom
to place any row anywhere; a temporal split fixes every placement by date. You cannot have both.

**Decision: temporal wins, because leakage is a correctness bug and imbalance is a reporting
problem.** A leaked test set gives a wrong number that looks right. An imbalanced test set gives a
*noisy* number that we can label as noisy — which §6.4 does.

`iterative-stratification` is still used, in one place: sampling the **annotation queue** (§6.3),
where we *do* have full freedom over which rows to send to humans, and where getting rare services
represented is the whole point.

### 6.3 Groups, and the compromise

**[measured]** all 184 organisations in the temporal test split also appear in train. `org_0001`
alone has 331 tickets. Enforcing group separation would either move ~30% of the corpus wholesale
or shrink the test set below usefulness.

**Decision: do not enforce group separation. Measure and report it instead.** The manifest and
datasheet carry the organisation overlap rate and the share of test rows whose organisation has
>50 training tickets. That is an honest "this number is optimistic by an unmeasured amount,"
which beats both a silent leak and a mutilated dataset.

> **Why this compromise is defensible here specifically.** The prediction target is *which
> service* a ticket concerns, not *which customer* wrote it. Organisation identity is a weaker
> shortcut than it would be for, say, churn prediction — where per-customer leakage is fatal.
> Different task, different verdict on the same trade. Copying the rule without the reasoning is
> how these decisions get made wrongly elsewhere.

### 6.4 The gold set this pipeline cannot build

Classifier spec §2.2 requires the evaluation set to be **blind-annotated by ≥2 support agents**,
with Krippendorff's α ≥ 0.67 and adjudication. **[measured]** this CSV has none of that — labels
came from the generator that wrote the text.

**So the pipeline emits a sampling frame, not a gold set:** `annotation_queue.jsonl`, stratified
over (month × language × cardinality) exactly as §2.2 specifies, blind (no `services` field
present in the record at all — omitted from the schema, not blanked, so it cannot leak through a
UI bug), with the taxonomy definitions attached.

**Until humans annotate it, `test.parquet` is a smoke-test set, and the datasheet says so in those
words.**

> **The trap this closes.** The most likely way this project goes wrong is not a bug. It is
> someone training a model, computing macro-F1 = 0.86 on `test.parquet`, putting it in a slide,
> and nobody noticing that the labels were written by the same generator that wrote the text. The
> pipeline cannot prevent that socially. It can make the warning impossible to miss.

### 6.5 Unmeasurable services

**[measured]** per-service positives in the gapped temporal split:

| Service | Train | Val | Test |
|---|---|---|---|
| `terraform-provider` | 13 | 1 | 1 |
| `message-queue` | 25 | 5 | **0** |
| `cdn` | 26 | 6 | 4 |
| `dns` | 31 | 5 | 5 |
| `managed-redis` | 67 | 12 | 17 |

`message-queue` has **zero** positives in test. Its F1 is not low — it is **undefined**. Recall
has a zero denominator. Averaging it into a macro-F1 requires inventing a value, and whichever
value you pick is a fiction that moves the headline number.

**Requirement:**

- The pipeline computes per-service test counts and writes them to the manifest.
- Services with **< 20 test positives** are marked `measurable: false`.
- `datasheet.md` lists them by name.
- Downstream evaluation **must** report macro-F1 twice: over all 20 services, and over measurable
  services only, with `n` beside every per-service number.

This is classifier spec §2.8's "services that cannot be learned or evaluated reliably — list them
explicitly rather than pretending the metric covers them," made mechanical so it cannot be
forgotten under deadline.

---

## 7. Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Full build (LLM stages read from cache when warm)
python -m ticketds.cli run --config configs/dataset.v1.yaml

# One stage, against the cached predecessor — the iteration loop for redaction work
python -m ticketds.cli stage redact --config configs/dataset.v1.yaml

# What will the LLM stages cost and send? No calls made.
python -m ticketds.cli run --config configs/dataset.v1.yaml --dry-run

# Reproduce any [measured] claim in this document
python -m ticketds.cli report --stage dedup --config configs/dataset.v1.yaml

# Redaction quality against the span gold set (§4.6)
python -m ticketds.cli eval-redaction --gold tests/fixtures/span_gold.jsonl

# Verify a build reproduces byte-identically
python -m ticketds.cli verify --version v1

# Quality gates
pytest -q                                    # unit + property + golden
pytest -q -m integration                     # full pipeline on a 200-row slice
ruff check . && ruff format --check .
mypy src/ticketds
```

`run` is idempotent: unchanged config plus unchanged input plus warm cache means every stage is a
cache hit and every output hash is unchanged.

## 8. Project structure

```
configs/
  dataset.v1.yaml          Pipeline config (Appendix B). Versioned; a change = a new dataset version
  taxonomy.yaml            The 20 services, definitions, 3 positive / 2 negative examples each
  redaction.yaml           Entity rules + placeholder vocabulary (Appendix A)
prompts/
  label_review.v1.md       S5 prompt. Hash goes in the manifest
  redaction_audit.v1.md    S6 prompt
src/ticketds/
  cli.py                   Typer CLI: run / stage / report / eval-redaction / verify
  config.py                Pydantic models for every config file
  schema.py                Pandera schemas, one per stage boundary
  pipeline.py              Stage registry, sequencing, manifest assembly
  stages/                  s0_ingest.py … s8_freeze.py — one module per stage
  redact/                  THE SHARED LIBRARY — also imported by the model service (§4.5)
    __init__.py            redact(text, cfg) -> RedactionResult
    rules.py               Structural, checksum, context rules
    ner.py                 Presidio engine + custom Russian recognizers
    placeholders.py        The closed vocabulary
  llm/
    client.py              OpenRouter client: retry, backoff, structured output, call ceiling
    cache.py               Content-addressed disk cache (§5.4)
  metrics.py               Class balance, cardinality, leakage, per-service counts
  datasheet.py             Generates datasheet.md
data/
  raw/tickets_export.csv   Input (in git)
  stage/                   Intermediate parquet (gitignored)
  cache/llm/               LLM response cache (gitignored; shareable via a tarball)
  processed/v1/            Output (gitignored; manifest.json IS committed)
tests/
  unit/  property/  integration/
  fixtures/
    span_gold.jsonl              200 hand-annotated rows (§4.6)
    redaction_regressions.jsonl  Every confirmed miss, forever (§5.2)
    slice_200.csv                Deterministic 200-row slice for fast integration runs
```

> **Why `manifest.json` is committed and the parquet is not.** The data is regenerable; the claim
> about it is the reviewable artifact. A diff on `manifest.json` in a pull request shows exactly
> what a config change did to the dataset — a row count moved, a hash changed — which is the
> review you actually want.

## 9. Code style

One snippet is worth three paragraphs. Every stage looks like this:

```python
# src/ticketds/stages/s1_normalize_labels.py
"""S1 — normalise the free-text `services` column into a validated label set."""

from __future__ import annotations

import polars as pl

from ticketds.config import Config
from ticketds.schema import S0_SCHEMA, S1_SCHEMA
from ticketds.stages.base import StageResult, quarantine


def run(df: pl.DataFrame, cfg: Config) -> StageResult:
    """Split, clean, validate against the taxonomy.

    Rows whose services are not all in the taxonomy are quarantined, never dropped:
    a service renamed upstream is exactly what this check exists to catch.
    """
    S0_SCHEMA.validate(df)
    taxonomy = set(cfg.taxonomy.services)

    parsed = df.with_columns(
        pl.col("services")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars().str.to_lowercase())
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique()
        .list.sort()
        .alias("services_norm")
    )

    unknown = parsed.filter(
        pl.col("services_norm").list.eval(~pl.element().is_in(list(taxonomy))).list.any()
    )
    kept = parsed.join(unknown.select("row_index"), on="row_index", how="anti")

    kept = kept.with_columns(
        pl.col("services_norm").list.len().alias("n_services"),
        pl.when(pl.col("services_norm").list.len() > 0)
        .then(pl.lit("labelled"))
        .otherwise(pl.lit("unlabelled"))
        .alias("label_status"),
    )

    S1_SCHEMA.validate(kept)
    return StageResult(
        frame=kept,
        quarantined=quarantine(unknown, reason="unknown_service", stage="s1"),
        metrics={"labelled": ..., "unlabelled": ..., "unknown_service": unknown.height},
    )
```

**Conventions:**

- `run(df, cfg) -> StageResult` for every stage. No exceptions, no side channels.
- Schema validated on entry **and** exit. A stage that corrupts its output fails at its own
  boundary, not three stages later.
- Rejected rows are returned as data with a reason, never dropped, never raised.
- Every stage returns `metrics` — that dict is what the manifest and datasheet are built from.
- Type hints everywhere; `mypy --strict` on `src/ticketds`.
- Docstrings explain **why**, not what. The code says what.
- `ruff` for lint and format, default settings, line length 100.
- Regex rules live in YAML with a `name`, `pattern`, `entity_type`, `confidence`, and **at least
  one `example` string that the test suite asserts it matches** (§10).

## 10. Testing strategy

`pytest`, plus `hypothesis` for the properties that matter.

| Level | Location | What it covers |
|---|---|---|
| **Unit** | `tests/unit/` | Label normalisation edge cases; each redaction rule against its declared examples; taxonomy validation; config parsing |
| **Property** | `tests/property/` | Redaction invariants — see below |
| **Golden** | `tests/unit/test_redaction_golden.py` | `span_gold.jsonl` (§4.6) and `redaction_regressions.jsonl` (§5.2). Every confirmed miss stays fixed forever |
| **Integration** | `tests/integration/` | Full pipeline over `slice_200.csv` with the LLM client stubbed; asserts manifest reproducibility |

**Properties worth stating explicitly**, because each maps to a real failure:

```python
@given(text=st.text())
def test_redaction_is_idempotent(text):
    once = redact(text, CFG).text
    assert redact(once, CFG).text == once          # placeholders must not re-trigger rules

@given(text=st.text())
def test_placeholders_are_closed_vocabulary(text):
    assert set(PLACEHOLDER_RE.findall(redact(text, CFG).text)) <= KNOWN_PLACEHOLDERS

@given(text=st.text())
def test_spans_reconstruct_the_output(text):
    r = redact(text, CFG)
    assert apply_spans(text, r.spans) == r.text    # the sidecar is not a separate truth
```

**Build-failing gates** (§4.6, §5.2, and the CI job in §11):

- Any detector matching **zero** spans across the corpus (§4.3 — silent regex failure)
- Span recall below 1.00 on `SECRET` / `CARD` / `BANK_ID` / `GOV_ID`
- Span recall below 0.95 on any other entity type
- Any row whose text still matches the mojibake signature after S0 repair
- Any row present in two splits
- Any `dup_cluster_id` spanning a split boundary
- Any row with `label_status == "unlabelled"` in train, val, or test
- A repeated build producing a different manifest hash

> **Why "a detector found zero" is a test failure and not a log line.** §4.3 measured it happening
> for real: a defensible IBAN regex matched 0 of 16, and a defensible passport regex matched 0 of
> ~14. Both looked like "this entity isn't in the corpus." A zero-match detector is the quietest
> possible bug in a safety-critical path, so it has to be loud somewhere.

## 11. Boundaries

**Always:**

- Quarantine with a reason; never silently drop a row
- Validate the schema at both ends of every stage
- Record every decision in the manifest — versions, hashes, counts, model ids, prompt hashes
- Re-run the full test suite before committing
- Treat a change to `redaction.yaml`, `taxonomy.yaml`, or any prompt as a **new dataset version**
- Send **redacted** text to the LLM unless `llm.allow_raw_text` is explicitly set (§5.3)

**Ask first:**

- Adding a dependency
- Changing the placeholder vocabulary (it is part of the model's input contract — §4.5)
- Changing split ratios, the gap length, or the `measurable` threshold
- Raising `llm.max_calls` or switching `llm.model`
- Introducing DVC or an orchestrator (§2 says not yet; that decision has an owner)

**Never:**

- Let an LLM write a label or edit text into the dataset without human confirmation (§5.1, §5.2)
- Use test data to select thresholds, checkpoints, or hyperparameters (§1.6)
- Gate redaction on a checksum (§4.4)
- Report a metric for a service marked `measurable: false` without its `n` beside it
- Commit raw or partially-redacted ticket text outside `data/raw/`
- Disable a redaction rule to make a test pass

## 12. Success criteria

Each is mechanically checkable — that is the point.

1. `python -m ticketds.cli run` completes on all 5,013 rows; input rows = output rows +
   quarantined rows, asserted.
2. Two consecutive builds produce **identical hashes** for all outputs (`verify` passes).
3. All 5,013 rows are accounted for: labelled / unlabelled / quarantined, with reasons.
4. Span recall on the gold set is **1.00** for `SECRET`, `CARD`, `BANK_ID`, `GOV_ID`, and ≥0.95
   elsewhere, with precision ≥0.90.
5. No detector returns zero matches, and no row survives S0 still matching the mojibake signature.
6. Splits are leak-free by the §10 checks; the manifest reports the random-vs-temporal leakage
   delta.
7. `review/labels.jsonl` holds 400 ranked rows; queue precision on the first human pass ≥0.30.
8. S6 audits ~790 rows; every confirmed miss has a rule and a regression test.
9. `datasheet.md` opens with §1.8's warning, and names every `measurable: false` service.
10. A teammate who has not read this document can run the pipeline from `README` alone and get
    hash-identical output.

## 13. Open questions

Blocking ones first. Each needs an owner before the affected stage is built.

| # | Question | Why it matters | Owner |
|---|---|---|---|
| **1** | Are the 391 empty-`services` rows "no service applies" or "not triaged"? | Decides whether they are training negatives or unknowns (§1.4). Currently quarantined — the safe default, possibly wasteful | Product |
| **2** | Who annotates the 200-row span gold set, and when? | **Blocks the §4.6 recall gate, which blocks every success criterion about redaction.** ~10 person-hours | Eng |
| **3** | Does the review queue depth of 400 match available human capacity? | Sets the S5 saving and the ≥0.30 precision gate (§5.1) | Eng lead |
| **4** | Is `<PERSON>` NER worth its cost on this corpus? | Presidio + `ru_core_news_lg` is the heaviest, least precise layer (§4.2). Names are also the entity a model is least likely to need. Possible answer: ship without it, let S6 tell us how much we lost | Eng |
| **5** | Is 20 test positives the right `measurable` floor? | Arbitrary today. A power calculation for the target precision CI would replace the guess | Eng |
| **6** | Do we keep the repaired mojibake rows, or quarantine them? | Repair is proven to work on all 9 **[measured]**, and they hide real PII (S0). Keeping them is the recommendation; the risk is that a real export has un-repairable variants | Eng |
| **7** | When does the fixture get replaced by a real export? | Everything in §1.8 changes the day it does. Until then, no number here means anything about production | Product |

---

## Appendix A — Placeholder vocabulary

Twelve placeholders reach the text. The detector's finer `entity_type` is preserved in
`s2_spans.parquet` (§4.5b).

### A.1 Redaction — safety, never disabled

| Placeholder | Fine-grained `entity_type` in the sidecar | Detector layer |
|---|---|---|
| `<EMAIL>` | `EMAIL` | structural |
| `<PHONE>` | `PHONE_RU`, `PHONE_INTL` | structural |
| `<URL>` | `URL`, `URL_WITH_TOKEN`, `WEBHOOK` | structural |
| `<IP>` | `IPV4`, `IPV6`, `CIDR` | structural |
| `<HOST>` | `HOSTNAME`, `K8S_POD`, `BUCKET`, `QUEUE` | structural + context |
| `<PERSON>` | `PERSON_NAME` | NER |
| `<ADDR>` | `ADDRESS`, `DOB` | NER + context |
| `<GOV_ID>` | `SNILS`, `PASSPORT_RU`, `DRIVING_LICENCE`, `TAX_ID_PERSONAL` | context + checksum-as-confidence |
| `<CARD>` | `PAN`, `PAN_MASKED`, `EXPIRY`, `CVV` | structural + Luhn-as-confidence |
| `<BANK_ID>` | `IBAN`, `SWIFT_BIC`, `BIK`, `ACCOUNT_RU`, `RRN`, `AUTH_CODE`, `PAYMENT_ID` | context + structural |
| `<ORG>` | `LEGAL_NAME` | NER |
| `<ORG_ID>` | `INN`, `KPP`, `OGRN`, `OGRNIP`, `VAT`, `DUNS`, `CONTRACT_NO` | context |
| `<SECRET>` | `API_KEY`, `BEARER`, `JWT`, `AWS_AKID`, `AWS_SECRET`, `OAUTH_SECRET`, `WEBHOOK_SECRET`, `PRIVATE_KEY`, `SSH_KEY`, `CONNSTR_PASSWORD`, `PASSWORD` | structural |

### A.2 Normalisation — token budget, ablatable

| Placeholder | Replaces | Rationale |
|---|---|---|
| `<TS>` | ISO-8601 timestamps, log-line datetimes | High entropy, zero label signal |
| `<UUID>` | UUIDs, request ids, trace ids | Same |
| `<HEX>` | Hex runs ≥16 chars (hashes, object ids) | Same |
| `<NUM>` | Bare integer runs ≥5 digits not matched by a redaction rule | Same. Ordered **last**, so it can never pre-empt a safety rule |

**Rules:** placeholders are uppercase, angle-bracketed, and a closed set (asserted, §10). Rule
order is redaction → normalisation, and within redaction most-specific first — a card number must
be consumed by `<CARD>` before `<NUM>` can see it. Consecutive identical placeholders collapse to
one (`<IP> <IP> <IP>` → `<IP>`), configurable, since a 40-line access-log paste otherwise becomes
40 tokens of noise.

## Appendix B — Config schema

```yaml
# configs/dataset.v1.yaml
version: v1
seed: 20260820                       # every stochastic step derives from this

input:
  path: data/raw/tickets_export.csv
  encoding: utf-8
  encoding_errors: replace           # flag U+FFFD rows, keep them (S0)

taxonomy: configs/taxonomy.yaml
redaction:
  enabled: true                      # NOT overridable — safety, not an experiment (§4.1)
  rules: configs/redaction.yaml
  collapse_repeats: true
normalisation:
  enabled: true                      # ablation knob (§4.1)
  placeholders: [TS, UUID, HEX, NUM]

dedup:
  minhash_permutations: 128
  shingle_size: 5
  drop_threshold: 0.80               # drop later copy from the later split
  cluster_threshold: 0.60            # keep, but never split a cluster across a boundary

split:
  strategy: temporal                 # temporal | random (random is diagnostic only, §6.1)
  ratios: {train: 0.70, val: 0.15, test: 0.15}
  gap_days: 7
  measurable_min_test_positives: 20  # below this → measurable: false (§6.5)

llm:
  provider: openrouter
  model: anthropic/claude-opus-5
  temperature: 0.0
  max_calls: 7000                    # hard ceiling; a bug cannot spend unbounded money
  allow_raw_text: false              # S6 requires this explicitly (§5.3)
  cache_dir: data/cache/llm
  label_review:
    prompt: prompts/label_review.v1.md
    measurement_sample: 500
    queue_depth: 400
  redaction_audit:
    prompt: prompts/redaction_audit.v1.md
    strata: {no_span_with_payload: 274, low_confidence: 200, self_redacted: 168, control: 150}

output:
  dir: data/processed/v1
  formats: [parquet]
  write_datasheet: true
```

## Appendix C — Output schema

`train.parquet` / `val.parquet` / `test.parquet`:

| Column | Type | Note |
|---|---|---|
| `id` | `Utf8` | Source ticket id |
| `model_input` | `Utf8` | **The `x`.** `"[priority: {p}] {title}\n{description}"`, redacted |
| `services_norm` | `List[Utf8]` | **The `y`.** Sorted, validated, deduplicated |
| `y` | `List[UInt8]` | 20-wide binary vector, taxonomy order (§1.2) |
| `n_services` | `UInt8` | Cardinality |
| `split` | `Utf8` | `train` / `val` / `test` |
| `organization_id` | `Utf8` | Group reporting (§6.3). **Not a feature** |
| `created_at` | `Datetime` | Temporal slices |
| `lang` | `Utf8` | `ru` / `en` / `other`, payload-stripped (S4) |
| `priority` | `Utf8` | Part of `model_input`; kept separately for slicing |
| `char_len`, `est_tokens` | `UInt32` | Length slices, `max_len` choice |
| `dup_cluster_id` | `UInt32?` | Null if not in a cluster |
| `has_payload` | `Boolean` | Pasted-block slice |
| `n_redacted_spans` | `UInt16` | Redaction-intensity slice |
| `label_dirty_flags` | `List[Utf8]` | Which S1 normalisations fired |
| `llm_disagreement_score` | `Float32?` | From S5; null if not scored |
| `human_reviewed` | `Boolean` | True once a human has confirmed the label |

`quarantine.parquet` carries every source column plus `quarantine_reason` and `quarantine_stage`.

---

## Appendix D — Reproducing every [measured] claim

Every number in this document came from the committed CSV. The scratch scripts that produced them
become `ticketds.cli report` subcommands during implementation, so the claims stay checkable as
the data changes:

| Claim | §  | Command |
|---|---|---|
| Cardinality, class balance, label hygiene | 1.2, 1.4 | `report --stage normalize-labels` |
| Near-duplicate pairs, raw vs redacted | 1.3, S3 | `report --stage dedup` |
| Random vs temporal leakage (6.2% / 3.7%) | 1.3 | `report --stage split --compare-strategies` |
| Organisation overlap (184/184) | 6.3 | `report --stage split` |
| Entity hit counts, README gaps, Luhn (6/39) | 4.3, 4.4 | `report --stage redact` |
| Language heuristic disagreement (1,123 vs 237) | S4 | `report --stage enrich` |
| Mojibake rows (9) and their repair | S0 | `report --stage ingest` |
| Split sizes, gap loss, per-service test counts | 6.1, 6.5 | `report --stage split` |
