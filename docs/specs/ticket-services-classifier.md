# Ticket → Services Classifier: replacing the CatBoost model with a fine-tuned multilingual encoder

Status: specification (not implemented)
Author: ml-researcher
Date: 2026-08-15
Audience: implementing developer; `system-architect` owns §9 integration design.

---

## 0. Repo state and scope

**No existing assets in this repo.** `/home/user/bert-poc` contains only `README.md` (stub)
and `.claude/agents/*.md`. There are no datasets, no notebooks, no model artifacts, no
training code, no CatBoost model, no evaluation harness. Everything below is specified from
scratch. The existing CatBoost model and the ticket database live outside this repo and must
be obtained (see §8 open questions).

All measurements labelled **[measured]** were run by me on the machine this spec was written
on and are reproducible from the scratch script noted in §9.6. Everything labelled
**[estimate]** is arithmetic with the working shown; treat it as a planning number, not a
result.

---

## 1. Problem statement

### 1.1 Task in ML terms

Multi-label text classification with a hard cardinality constraint.

- **Input** (available at ticket-creation time): `title` (short string), `description`
  (free text), `priority` (categorical). Both text fields may be Russian, English, or
  code-switched within one ticket. `labels` and `flags` are *candidate* inputs whose
  availability at prediction time is unresolved — see §2.7 and §8 Q2.
- **Output**: a subset `S ⊆ Services`, `|Services| ≈ 20`, with the empirical constraint
  `1 ≤ |S| ≤ 4`. The model emits a score per service; the decode rule (§4.6) turns scores
  into a set, and is allowed to emit the empty set (abstain) even though ground truth never
  is empty.
- **Decision informed**: the predicted set is written to `tickets.services` and rendered as
  an internal `ticket_messages` row visible to human agents. So the model output is a
  *suggestion to a human*, not a terminal automated action.

### 1.2 What "good enough" means

The product constraint given is explicit: **a wrong service shown to an agent is worse than
a missing one.** This makes the task precision-first with recall as the thing we maximise
subject to a precision floor. Two consequences that shape everything downstream:

1. The primary metric is **micro-averaged recall at a fixed micro-precision floor**
   (`R@P≥0.90`), not F1 and not accuracy. F1 trades precision away for recall at a 1:1 rate,
   which contradicts the stated preference.
2. **Abstention must be a first-class output.** If no service clears its threshold, emitting
   nothing is the correct behaviour. Whether the product tolerates an empty internal message
   is §8 Q6.

Minimum bar to ship: strictly better than the current CatBoost model *re-measured on the same
clean test set* (its currently-reported numbers are not comparable — §3.3), at equal or lower
operational cost, on CPU-only inference. Full ship criterion in §6.6.

### 1.3 Explicit non-goals

- Not predicting `labels` (free-form, open vocabulary) or `flags`.
- Not routing/assignment, not priority prediction, not SLA prediction.
- Not choosing where inference runs — §9 states requirements only.

---

## 2. Data

### 2.1 The label-provenance trap (read this before writing any query)

`tickets.services` is a **mixture of model output and human correction**. Training on the
column as-is trains the new model to imitate the old CatBoost model, which caps quality at
CatBoost's ceiling and, worse, makes offline metrics look *good* precisely where the new
model reproduces old mistakes. This is the classic direct-feedback-loop failure described in
[Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf):
a model influences the data that trains its successor. Sculley et al.'s recommended
mitigation — hold out a random slice of traffic on which the model does not act, and use it
as an unbiased label source — is adopted in §6.7.

**Non-negotiable rule: no label whose provenance cannot be established as human may appear in
the validation or test set.** Model-provenance labels may optionally be used as auxiliary
*training* signal (§2.5), never for selection or reporting.

#### 2.1.1 The lucky break: `ticket_messages` is already an audit log

The described lifecycle writes the predicted list into a `ticket_messages` row with
`type = 'internal'` at workflow completion. That row is a **timestamped snapshot of the model
prediction for every ticket** — it exists whether or not the platform has a proper audit
table. This gives a provenance reconstruction that needs no schema change:

For each ticket, define
- `P` = set parsed from the earliest `type='internal'` prediction message for that ticket
- `C` = current value of `tickets.services`

Then:

| Case | Condition | Interpretation | Use |
|---|---|---|---|
| **A — corrected** | `C ≠ P` | A human demonstrably edited the field | **Gold-eligible.** `C` is a human label. |
| **B — unchanged** | `C = P` | Ambiguous: agent agreed, *or* nobody ever looked | **Not gold.** Weak/unlabelled. |
| **C — no prediction row** | no internal message | Pre-model ticket, or manual creation path | **Gold-eligible if** the field was ever set by a human; verify. |
| **D — unparseable** | message format drifted | Parser gap | Quarantine, count, report. |

Case A is the highest-value training and evaluation data available without new engineering.
Case C (tickets predating the CatBoost rollout) is the *cleanest* source if it exists and is
recent enough to be in-distribution — check the rollout date first.

**Parsing risk:** the internal message contains the service list "as text". Its format may
have changed over time. Before trusting this pipeline, sample 200 messages across the whole
period, parse them, and report parse-success rate per month. Fail the approach if
success < 95% in any month you intend to use.

#### 2.1.2 Case A is biased, and that bias must be corrected

Case A is not a random sample of tickets: it over-represents tickets the model got wrong, and
under-represents tickets that are easy. Training on Case A alone yields a model tuned to
hard cases; *evaluating* on Case A alone under-states real precision. Therefore:

- Case A tickets may be **up-weighted training data**, but
- the **test set must be a stratified random sample of all tickets** in the period, relabelled
  by humans (§2.2), not a filter on Case A.

#### 2.1.3 What to do about Case B (silent agreement)

Case B is the majority of the corpus and is genuinely unknown. Three options, in order of
preference:

1. **Engagement-conditioned weak labels.** Restrict Case B to tickets where an agent
   demonstrably engaged with the ticket *after* the prediction was posted (≥1 agent-authored
   message, ticket reached a resolved/closed state, assignee set). Call this **B-strong**.
   Treat B-strong as noisy-positive supervision with a reduced sample weight (start at 0.3;
   tune as a hyperparameter, §5.3). **Validate this assumption empirically**: blind-relabel a
   random 200 B-strong tickets and measure how often the stored set matches the human set. If
   agreement < 80%, drop B-strong from training entirely and say so.
2. **Unlabelled data for domain-adaptive pretraining** (§4.7) — the text is useful even when
   the label is not.
3. Discard.

#### 2.1.4 If the platform records nothing about who wrote `services`

Say it plainly to stakeholders: **if neither an audit trail nor a parseable prediction
message exists, there is no way to separate model labels from human labels retroactively, and
no amount of modelling fixes it.** In that case:

- **Required engineering (blocking, owned by backend/`system-architect`):** append-only
  capture of every write to `tickets.services` with `(ticket_id, old_value, new_value,
  actor_type ∈ {model, agent, api, migration}, actor_id, timestamp, model_version)`. This is
  a schema/infra decision and is **not mine to design** — I am stating the data requirement.
- **Waiting cost [estimate]:** at a ticket volume of *V*/month and a human-correction rate of
  *r*, you accumulate `V·r` Case-A tickets per month. At V=20,000 and r=15% that is 3,000/mo,
  so ~4 months to reach a 12,000-ticket corrected set. Both V and r are unknown (§8 Q3) —
  measure them in week 1, because they set the project timeline.
- **Fallback that does not require waiting (recommended to run in parallel regardless):**
  human relabelling of a stratified random sample, blind to the stored value (§2.2). This is
  the only path that yields a trustworthy test set on week-one timelines, and it is needed
  even in the happy path.

### 2.2 The gold set (human relabelling)

This is the backbone of the whole project. Specification:

- **Sampling**: stratified random over (month × language × predicted-cardinality). Do **not**
  stratify on the stored service values — that bakes the old model's distribution into the
  test set. Include a stratum of tickets where the model abstained/failed, if such exist.
- **Blindness**: annotators must **not** see the stored `services` value or the internal
  prediction message. Showing it causes anchoring and inflates measured agreement with the
  old model; this is the same automation-bias mechanism that makes model-in-the-loop
  annotation produce labels that agree with the model rather than with the truth. Present
  title + description + priority only, in the same form the model sees.
- **Annotators**: experienced support agents, ≥2 of them, with a written guideline containing
  one-paragraph definitions and 3 positive / 2 negative examples per service. The guideline is
  a deliverable, versioned in the repo alongside the data snapshot.
- **Volume [estimate]**:
  - test: **2,000 tickets** — sized so that a per-service slice with 5% prevalence has ~100
    positives, enough for a precision estimate with ±~8pp bootstrap CI. Rare services will
    still be under-powered; §6.3 says how to report that honestly.
  - validation: **1,000 tickets** (thresholds + hyperparameter selection).
  - overlap for agreement: **300 tickets** double-annotated.
  - **Cost [estimate]:** 3,000 single-annotations + 300 duplicates ≈ 3,300 items × ~60 s ≈
    **55 person-hours**, plus guideline authoring and adjudication ≈ 70 h total.
- **Agreement**: report **Krippendorff's α with the MASI distance**, the standard choice for
  set-valued annotation ([Passonneau 2006, "Measuring agreement on set-valued items (MASI)"](https://www.researchgate.net/publication/228528415_Measuring_agreement_on_set-valued_items_MASI_for_semantic_and_pragmatic_annotation);
  see also [Springer 2023, "Annotation Quality Measurement in Multi-Label Annotations"](https://link.springer.com/chapter/10.1007/978-3-031-44696-2_3)).
  Also report per-service Cohen's κ (binary, one-vs-rest) — a low global α is usually caused
  by 2–3 confusable services, and the per-service view finds them.
  **Gate: α ≥ 0.67.** Below that, the taxonomy is the problem, not the model: stop, merge or
  redefine the offending services with the product owner, and re-annotate. Krippendorff's
  conventional thresholds (α ≥ 0.80 good, ≥ 0.667 tentative) are the reference here.
  **Human α is also the model's ceiling** — report model metrics against it.
- **Adjudication**: disagreements resolved by a third senior agent; the adjudicated set is
  the test set.

### 2.3 Training corpus

Priority order, each with a provenance flag carried through to the training record:

1. Case A (corrected) — weight 1.0
2. Case C (pre-model human-authored), if in-distribution — weight 1.0
3. Gold-set training remainder, if any beyond val/test — weight 1.0
4. B-strong (§2.1.3) — weight ~0.3, only if the 200-ticket audit passes
5. Case B generally — **excluded** from supervised training; usable as unlabelled text (§4.7)

Store provenance as a first-class column. Every metrics report must be sliceable by it.

### 2.4 Volume required (and the cold-start path)

Unknown until §8 Q3 is answered. Planning brackets [estimate]:

| Usable human-labelled tickets | Recommended approach | Expectation |
|---|---|---|
| < 1,000 | Do **not** fine-tune. Frozen multilingual embeddings + one-vs-rest logistic regression, or SetFit. Spend the effort on annotation instead. | Likely at or below CatBoost |
| 1,000 – 5,000 | SetFit or frozen-embedding head; fine-tuning viable but unstable — use the small-data recipe (§5.4) | Parity to modest gain |
| 5,000 – 20,000 | **Fine-tuned encoder — the recommended path** | Expect clear gain |
| > 20,000 | Fine-tuned encoder, larger backbone worth testing | Best case |

The low-data anchor is [SetFit (Tunstall et al., 2022)](https://github.com/huggingface/setfit),
which reports that with **8 labelled examples per class** on the Customer Reviews sentiment
task it is competitive with RoBERTa-Large fine-tuned on the full 3k training set. That is a
sentiment task, not a 20-way multi-label ticket task, so do not port the number — port the
*shape* of the finding: contrastive fine-tuning of a sentence encoder is the right tool below
a few thousand labels.

**How many labels to beat CatBoost — honest answer: we must measure it.** The deliverable is
a **learning curve**, not a guess: train the recommended model on 10/25/50/100% of the
labelled data, 3 seeds each, plot val `R@P90` vs training size with error bars, and read off
(a) where it crosses the CatBoost baseline and (b) the slope at 100%, which tells the product
owner whether buying more annotation is worth it. This curve is a required artifact (§9.1);
it is the single most decision-relevant output of the project.

### 2.5 Optional: using model-provenance labels without distilling the old model

If Case A + C is thin, model labels can still help, but only in ways that cannot leak into
evaluation:

- **Pretraining then fine-tuning:** train one epoch on model-labelled data, then fine-tune on
  human-labelled data only. The second stage dominates.
- Never mix them in one batch at equal weight, never select hyperparameters on model labels,
  never report a metric computed against model labels.
- Report both variants (with and without stage 1) on the same gold test set. If stage 1 does
  not help, drop it — it carries permanent contamination risk for marginal gain.

### 2.6 Splits

**Temporal split, grouped and deduplicated.** Concretely:

- Sort by `created_at`. Train = oldest 70%, validation = next 15%, test = most recent 15%,
  with a **gap of ≥1 week** between train and val and between val and test to avoid burst
  correlation (an incident produces many near-identical tickets within hours).
- The 2,000-ticket gold test set is drawn from the **most recent** period. Additionally keep
  a small **older gold slice (~300 tickets)** to measure drift sensitivity.
- **Group by reporter/organisation** where that field exists: a single customer's tickets are
  near-duplicates of each other and split them across train/test and you leak.
- **Near-duplicate removal across the split boundary**: MinHash/LSH on character 5-grams,
  Jaccard ≥ 0.8 → drop the later copy from test. Report how many were dropped; a large number
  is itself a finding about the corpus.

Why temporal and not random: services get added and renamed, product surface changes, and
ticket phrasing drifts, so a random split measures a task the model will never face.
Empirically, random splits overestimate performance on temporally-ordered operational data;
[Lyu et al., "An Empirical Study of the Impact of Data Splitting Decisions on the Performance
of AIOps Solutions" (TOSEM 2021)](https://dl.acm.org/doi/fullHtml/10.1145/3447876) shows
time-based splitting materially reduces leakage versus random splitting in an operational
ticket/log setting, and [Kapoor & Narayanan, "Leakage and the reproducibility crisis in
ML-based science" (Patterns, 2023)](https://arxiv.org/abs/2207.07048) catalogues how
pervasive this class of error is. **Required diagnostic:** report both random-split and
temporal-split validation numbers once. The gap is your drift magnitude, and it is a number
the product owner should see.

### 2.7 Features other than title/description

- **`priority`** — available at creation. Include as a text prefix (e.g. `[priority: high]`)
  rather than a separate numeric head; it costs 3–4 tokens and keeps the model single-input.
  Ablate it: if it adds < 0.5pp on val it should be dropped for simplicity.
- **`labels` / `flags`** — **unresolved (§8 Q2)** and dangerous. If they are populated
  *after* creation (e.g. by agents during triage) then training on them is target leakage:
  the model learns from information that does not exist at inference and offline metrics will
  be fantasy. Spec for both cases:
  - **If not reliably present at creation:** exclude entirely. This is the default until
    proven otherwise. Prove it by measuring populated-rate *as of creation time* from the
    audit trail, not from the current row.
  - **If present at creation for ≥95% of tickets and their creation-time value is
    recoverable:** include as an additional text segment, and train with **feature dropout**
    (randomly blank them 20–30% of the time) so the model degrades gracefully when they are
    missing. Report metrics both with and without the field at inference.
  - Free-form `labels` are open-vocabulary: feed as text, do not one-hot.

### 2.8 Class balance

Unknown; must be measured in week 1. Expect a long tail across ~20 services with a head of
3–5 services covering the majority of positives. Required report before modelling: per-service
positive counts in the human-labelled set, cardinality histogram (`|S| ∈ {1,2,3,4}`), and
co-occurrence matrix. Services with < 50 human-labelled positives cannot be learned or
evaluated reliably — list them explicitly and handle them via §7 (rules or abstain), not by
pretending the metric covers them.

### 2.9 Bias, PII, licensing

- Ticket text is **personal data**. Russian personal-data localisation (Federal Law 152-FZ,
  with tightened Art. 18 requirements in force from 1 July 2025 —
  [overview](https://www.lidings.com/media/legalupdates/localization_pd_update/),
  [Microsoft's summary of the localisation rule](https://learn.microsoft.com/en-us/compliance/regulatory/offering-russia-data-localization))
  requires that recording, storage and extraction of Russian citizens' personal data occur in
  databases located in Russia. Practically this **rules out sending raw ticket text to a
  foreign hosted LLM API** for training or inference, and constrains where training snapshots
  may live. Confirm applicability with legal (§8 Q7) — I am flagging the constraint, not
  giving legal advice.
- **Pseudonymise before the corpus leaves the production database**: regex + NER redaction of
  emails, phone numbers, card-like digit runs, full names, URLs with tokens, into stable
  placeholders (`<EMAIL>`, `<PHONE>`, …). Do this *before* tokenisation so training and
  inference preprocessing are identical — the same redaction must run at inference or you
  introduce train/serve skew.
- **Language imbalance** is a known bias axis: Russian is expected to dominate, so aggregate
  metrics will be Russian metrics. English is a first-class slice (§6.3).
- Model licences: all recommended checkpoints are MIT or Apache-2.0 (§4.2). No dataset
  licensing issue — the data is first-party.

---

## 3. Baselines

All baselines are evaluated on the same gold test set with the same decode rule and the same
threshold-selection procedure. Cheap ones come first because a negative result here is worth
more than a month of transformer work.

1. **Trivial — prior.** Always predict the single most frequent service; and a variant that
   predicts the top-2 most frequent. Gives the floor. If a learned model does not beat this
   on macro metrics, something is broken.
2. **Trivial — keyword rules.** Per service, take the top-30 TF-IDF terms (Russian and
   English, stemmed) from the human-labelled training set; predict a service if ≥2 of its
   terms appear. Cheap, interpretable, and in narrow-domain ticket taxonomies it is
   embarrassingly strong. It is also the fallback if the ML path fails.
3. **TF-IDF (word 1–2 grams + char 3–5 grams) + one-vs-rest linear SVM / logistic
   regression.** Character n-grams handle Russian morphology without a lemmatiser. This is
   the real baseline to beat, not the trivial ones.
4. **Current CatBoost model, re-measured.** See §3.3.
5. **Frozen multilingual embeddings + one-vs-rest logistic regression** (§4.1 row 2).
6. **Zero-shot LLM prompting** (§4.1 row 4) — as a baseline only, with the residency caveat.

### 3.3 Re-measuring CatBoost is mandatory and non-trivial

The production CatBoost model's historical metrics were almost certainly computed against
labels contaminated by its own predictions (§2.1), so they are not comparable and are
probably optimistic. Required:

- Obtain the exact production artifact and its feature-extraction code (§8 Q4).
- Score it on the **gold test set**, using the *same* decode and threshold procedure as the
  new model, with thresholds re-tuned on the gold validation set (giving the incumbent its
  best shot is the point).
- If any of its features are not reconstructible as-of-creation-time for those tickets, say so
  and report the comparison as approximate.
- Publish this as the single baseline number the project is judged against.

Also spec the **improved-CatBoost arm** (§4.1 row 1): the same gradient-boosting model with
better text handling via CatBoost's native text features (BoW / NaiveBayes / BM25 calcers —
[CatBoost text features docs](https://catboost.ai/docs/en/features/text-features),
[feature calcers reference](https://catboost.ai/docs/en/references/text-processing__feature_calcers))
and optionally frozen sentence-embedding features (CatBoost supports embedding features
directly). This arm is cheap, and if it closes most of the gap it changes the recommendation.

---

## 4. Model approach

### 4.1 Strategy comparison

Quality columns are **[estimate]s relative to the TF-IDF+linear baseline** unless a citation
is attached; they exist to rank the options, not to promise numbers. CPU-inference figures in
the last column are **[measured]** where marked (§4.5).

| # | Strategy | Expected quality | CPU inference feasibility | Training cost | Data requirement | Maintenance |
|---|---|---|---|---|---|---|
| 1 | **CatBoost + better text/embedding features** | Baseline to +small. Bag-of-words features cannot resolve paraphrase or cross-lingual synonymy; RU morphology hurts token matching | Excellent — ms-scale, tiny artifact | Minutes on CPU | Low | Low, but feature engineering is manual and per-language |
| 2 | **Frozen multilingual sentence embeddings + one-vs-rest logistic regression** | Moderate gain. ruMTEB *Classification* (which is exactly this protocol: frozen embeddings + logistic regression) scores mE5-small 56.44, mE5-base 58.26, mE5-large 61.01, ru-en-RoSBERTa 62.74, rubert-tiny2 52.17 ([Snegirev et al., ruMTEB, NAACL 2025](https://aclanthology.org/2025.naacl-long.12.pdf)) | Same encoder cost as row 3 — the head is free | Minutes (head only); embeddings computed once | Works from ~500 labels | Low. Re-fit head when services change — very attractive property |
| 3 | **Fine-tuned multilingual encoder** ← **recommended** | Highest of the CPU-feasible options. Fine-tuning consistently beats frozen features when ≥ a few thousand labels exist | **Feasible [measured]:** 43 ms p50 @ 256 tok, 4 threads, INT8 base-size encoder | 15–70 min GPU (§5.5) | ~5k+ labels for a clear win | Medium: retrain per taxonomy change; artifact + threshold versioning |
| 4 | **LLM zero/few-shot prompting** | Unknown, plausibly decent on head services, weak on org-specific taxonomy distinctions. Cannot be tuned to a precision floor without calibration | **Self-hosted: not CPU-feasible** at this latency/throughput. Hosted API: blocked by 152-FZ residency (§2.9) | None | None (its selling point) | Low technically, high governance. Prompt drift + vendor model deprecation |
| 5 | **LoRA-tuned small generative LLM** (e.g. Qwen3-0.6B/1.7B, Apache-2.0, 119 languages — [Qwen3 technical report](https://arxiv.org/pdf/2505.09388)) | Probably ≈ row 3, not better, on a closed 20-class task | Poor on CPU: autoregressive decode of a service list, 0.6B–1.7B params ≫ 118–278 MB encoder. Constrained decoding helps but does not fix throughput | Higher than row 3 (LoRA reduces trainable params by orders of magnitude — [Hu et al., LoRA, 2021](https://arxiv.org/abs/2106.09685) — but forward/backward cost still scales with the base model) | Similar to row 3 | Higher: generation needs output validation, prompt/template versioning |
| 6 | **kNN retrieval over embeddings** | Moderate. Strong at cold-start and adding a service (add examples, no retrain); weaker on the head where a discriminative boundary matters | Encoder cost + index lookup; index of 50k×384 fp32 ≈ 77 MB [estimate] | None beyond embedding the corpus | Works from tens of examples per service | Low, but the index becomes state to manage and version |

### 4.2 Recommendation

**Fine-tune a multilingual encoder with a 20-way sigmoid (binary-relevance) head, export to
INT8 ONNX, decode with per-service thresholds plus the cardinality cap.**

Why, specifically:

- The task is closed-set, high-volume, latency-tolerant-but-CPU-bound, and precision-critical.
  A discriminative encoder is exactly the right shape: it produces a *calibratable score per
  service*, which is what makes a precision floor enforceable. Generative approaches (rows
  4–5) produce text, and text does not have a threshold.
- The RU/EN mixed corpus is the reason for multilingual rather than monolingual: a shared
  encoder lets English tickets benefit from Russian training data and vice versa, which
  matters because the English slice will be the small one.
- CPU-only inference is satisfied with margin: **[measured]** 24.3 ms p50 at 128 tokens and
  42.9 ms p50 at 256 tokens for a base-size INT8 encoder on 4 vCPU (§4.5).
- Rows 1, 2 and 6 are not discarded — row 2 is a required baseline and the cold-start
  fallback, row 6 is the new-service mechanism (§4.8), row 1 is the incumbent's best defence.

**Default checkpoint: `FacebookAI/xlm-roberta-base`**, with **`microsoft/mdeberta-v3-base`**
and **`intfloat/multilingual-e5-base`** as the two challengers in the same sweep, and
**`intfloat/multilingual-e5-small`** as the CPU/RAM-frugal tier. Pick on validation
`R@P90`, not on my prior. My prior, for what it is worth: mDeBERTa wins quality if training
is stable, and its XNLI Russian zero-shot transfer of **80.8 vs XLM-R-base's 78.1** (avg
**79.8 vs 76.2**, [mDeBERTa-v3-base model card](https://huggingface.co/microsoft/mdeberta-v3-base))
is the best public evidence for a RU-relevant quality edge at identical size — but DeBERTa's
disentangled attention has a history of fp16 instability and slower/awkward ONNX export, so
XLM-R is the lower-risk default.

### 4.3 Checkpoint candidates (all figures verified from each model's `config.json`)

Parameter counts are **[computed]** by me from the config values (embedding matrix +
positional + per-layer attention/FFN/LayerNorm); they match the published counts where the
card states one.

| Checkpoint | L / H / vocab | Params (emb / total) | Emb share | fp32 size | Licence | Notes |
|---|---|---|---|---|---|---|
| [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2) | 3 / 312 / 83,828 | 26.8M / **29.1M** | 92% | 116 MB | MIT | Russian-first, tiny, 2048 max positions. ruMTEB Classification 52.17 — visibly weaker. Good latency floor, poor English |
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | 12 / 384 / 250,037 | 96.2M / **117.5M** | 82% | 470 MB | MIT | Best small option. ruMTEB Classification 56.44. Needs `query: ` prefix convention (footgun) |
| [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) | 12 / 768 / 250,002 | 192.4M / **277.5M** | 69% | 1,110 MB | MIT | **Default.** Canonical MLM checkpoint, 100 languages, 2.5 TB CommonCrawl |
| [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base) | 12 / 768 / 251,000 | 193.2M / **278.2M** | 69% | 1,113 MB | MIT | Best expected quality (XNLI ru 80.8). fp16/export risk |
| [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) | 12 / 768 / 250,002 | 192.4M / **277.5M** | 69% | 1,110 MB | MIT | XLM-R-base further contrastively trained; strong frozen-embedding baseline (ruMTEB 58.26) |
| [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) | 12 / 768 / **501,153** | 385.3M / **470.3M** | 82% | 1,881 MB | Apache-2.0 | Half a million vocab entries for a 2-language task. Not recommended without trimming |
| [`ai-forever/ru-en-RoSBERTa`](https://huggingface.co/ai-forever/ru-en-RoSBERTa) | 24 / 1024 / 98,505 | 101.4M / **403.7M** | 25% | 1,615 MB | MIT | Best ruMTEB Classification (62.74) and *exactly* RU+EN. But 302M non-embedding params ⇒ ~3.5× the compute of a base model. Only if the base tier misses the precision floor |

### 4.4 The multilingual vocabulary problem, and whether trimming is worth it

**[computed]** The embedding matrix is **69–92%** of parameters in every multilingual
candidate. For a corpus that is only Russian and English, most of those rows are dead weight.

Trimming the vocabulary to the tokens actually observed (plus specials) gives, for a 40k
target:

| Model | Params 250k vocab → 40k | fp32 artifact | INT8 artifact [estimate] |
|---|---|---|---|
| mE5-small (H=384) | 117.5M → **36.9M** | 470 → **147 MB** | ~118 → **~37 MB** |
| XLM-R base / mE5-base / mDeBERTa (H=768) | 277.5M → **116.2M** | 1,110 → **465 MB** | ~279 → **~116 MB** |
| LaBSE (501k → 40k) | 470.3M → **116.2M** | 1,881 → **465 MB** | ~470 → **~116 MB** |

**Is it worth it? Answer: it buys memory and artifact size, essentially not latency.** The
embedding layer is a gather (one row copy per token), not a matmul; compute is dominated by
the 12 transformer layers. So trimming is worth doing **if and only if** RSS or artifact size
is a binding constraint — and **[measured]** RSS for the INT8 base encoder is already only
503 MB (§4.5), so it probably is not binding. Recommendation:

- **Phase 1: do not trim.** Ship INT8 without it. One less thing to get wrong.
- **Phase 2: trim only if** the architect's memory budget per replica is < 512 MB, or many
  replicas share a host.
- If trimmed: derive the keep-set from ≥12 months of ticket text, keep all special tokens,
  keep the union of tokens from both languages, and **keep a generous margin** (40k, not 20k)
  because new product names appear over time. The literature supports this being safe:
  [Ushio et al., "An Efficient Multilingual Language Model Compression through Vocabulary
  Trimming" (EMNLP Findings 2023)](https://arxiv.org/abs/2305.15020) finds VT can retain
  original performance, with roughly 50% of the original vocabulary generally sufficient, and
  restricting to the top-40k most frequent tokens per target language introducing no
  measurable loss.
- **Mandatory guard if trimmed:** monitor the `<unk>` rate on live traffic weekly. A rising
  unk rate is the leading indicator of vocabulary drift and is one of the drift alarms in §7.

### 4.5 Measured CPU inference [measured]

Environment: Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 + **AVX-512-VNNI** present, 15 GB RAM,
`onnxruntime` 1.28.0 CPU EP, `intra_op_num_threads` as stated, `inter_op=1`, graph
optimisation ALL, batch 1 unless stated, 30 iterations after 5 warm-ups, synthetic token IDs.
Encoder forward pass only — the 20-way classification head is < 0.1% of this and the
tokenizer adds ~1–3 ms [estimate].

Latency, ms (p50 / p95):

| Model | Precision | 4 threads @128 | 4 threads @256 | 2 threads @256 | 1 thread @256 |
|---|---|---|---|---|---|
| mE5-small (117M) | INT8 | **12.5 / 13.5** | **25.6 / 31.6** | 33.3 / 34.4 | 54.2 / 55.4 |
| mE5-small | fp32 | 20.9 / 25.5 | 35.7 / 39.7 | 66.4 / 73.3 | 119.0 / 122.8 |
| mE5-base (278M) | INT8 | **24.3 / 27.6** | **42.9 / 89.2**¹ | 71.8 / 74.4 | 127.8 / 132.6 |
| mE5-base | fp32 | 56.2 / 72.7 | 108.8 / 120.4 | 106.4 / — | — |

¹ that p95 is a noisy outlier on a shared vCPU; `min` was 40.9 ms. Re-measure on the real
production CPU before committing to a p95 SLO.

Throughput (batch 16, 4 threads) and footprint:

| Model | Precision | tickets/s @128 | tickets/s @256 | Artifact on disk | Peak process RSS |
|---|---|---|---|---|---|
| mE5-small | INT8 | **143.1** | **61.4** | **118 MB** | **258 MB** |
| mE5-small | fp32 | 66.7 | 30.1 | 470 MB | 867 MB |
| mE5-base | INT8 | 64.1 | 26.1 | **279 MB** | **503 MB** |
| mE5-base | fp32 | — | — | 1,110 MB | — |

Readings that matter:

- **INT8 dynamic quantisation gives ~1.4–2.5× on latency and ~4× on artifact size** on this
  VNNI-capable CPU. Base INT8 (42.9 ms) is *faster* than small fp32 (35.7 ms is comparable) —
  i.e. quantising buys you a whole model size tier.
- **VNNI is load-bearing.** On CPUs without AVX-512-VNNI, INT8 can be *slower* than fp32
  (see e.g. [onnxruntime issue #12854](https://github.com/microsoft/onnxruntime/issues/12854)).
  §9.4 makes the production CPU feature set a hard requirement to confirm.
- Sequence length is the biggest lever available: 256→128 tokens roughly halves latency.
  Budget it deliberately (§4.6).
- OpenVINO is a possible alternative backend but published comparisons find no consistent
  advantage over ONNX Runtime on CPU transformer workloads
  ([ML6, "OpenVINO vs ONNX for Transformers in production"](https://blog.ml6.eu/openvino-vs-onnx-for-transformers-in-production-3e10c01520c8)).
  **Recommendation: ONNX Runtime; do not spend time on OpenVINO unless ORT misses the SLO.**

### 4.6 Architecture, input construction, decode

- **Input text**: `"[priority: {p}] {title}\n{description}"`, after PII redaction. If the
  checkpoint is an E5 model, prepend its required `query: ` prefix — the E5 cards state every
  input must start with `query: ` or `passage: ` even for non-English text
  ([mE5-small card](https://huggingface.co/intfloat/multilingual-e5-small)); getting this
  wrong silently costs quality.
- **Max sequence length**: **measure the token-length distribution first**, then set
  `max_len` to the p90 of it, capped at 256. Truncation strategy: **head+tail** (first 192 +
  last 64 tokens) rather than head-only — ticket descriptions often end with the actual ask
  after a pasted log. Ablate head-only vs head+tail; it is cheap and sometimes worth a point.
- **Pooling**: mean pooling over the last hidden state with attention masking. Ablate `[CLS]`.
  For E5 checkpoints mean pooling is what they were trained with.
- **Head**: single linear layer `H → 20` with sigmoid (binary relevance). Dropout 0.1 before
  it. Optionally a 2-layer MLP — ablate, expect no gain.
- **Auxiliary cardinality head**: linear `H → 4`, softmax over `k ∈ {1,2,3,4}`, cross-entropy,
  loss weight λ ∈ [0.1, 0.3]. See §4.6.1.
- **Frozen layers**: freeze the **embedding matrix** always. It is 69–92% of parameters, it
  removes ~3 GB of optimizer state for a base model [estimate: 192.4M × 12 bytes for grad +
  Adam m,v], and it removes the main catastrophic-drift surface. In the < 5k-label regime,
  additionally freeze the bottom 4 transformer layers (§5.4).
- **Tokenizer / domain tokens**: SentencePiece handles unknown product names by
  over-segmenting rather than emitting `<unk>`, so adding tokens is usually unnecessary.
  **Do this instead:** normalise the high-entropy, zero-signal spans that eat the token
  budget — URLs, UUIDs, order/ticket IDs, stack traces, base64 blobs, timestamps — into
  placeholders. **[estimate]** on ticket-like corpora this typically recovers 10–30% of the
  token budget; measure it on your corpus, since it directly translates into either shorter
  sequences (cheaper) or more real content per ticket (better). Only add explicit special
  tokens (`<EMAIL>`, `<PHONE>`, `<URL>`, `<CODE>`) and resize embeddings if the redaction
  placeholders are themselves over-segmented. If you add tokens **and** trim vocabulary,
  trim last.

#### 4.6.1 Exploiting the 1 ≤ |S| ≤ 4 constraint

Three mechanisms, to be evaluated in this order:

1. **Hard cap (do this unconditionally).** After per-service thresholding, keep at most the
   top-4 by score. Free, cannot hurt, guarantees the output contract.
2. **Per-service thresholds τ_s tuned on validation** (§6.1). This is the main precision
   lever. Per-class thresholds are well established to beat a global 0.5 in multi-label
   settings; see [Pillai et al., "Threshold optimisation for multi-label classifiers",
   Pattern Recognition 2013](https://dl.acm.org/doi/abs/10.1016/j.patcog.2013.01.012).
3. **Explicit cardinality head + intersect decode.** Predict k̂, take the top-k̂ services by
   score, and intersect with the thresholded set:
   `S = {top-k̂ scores} ∩ {s : score_s ≥ τ_s}`, then cap at 4.
   Intersection (not union) is the precision-preserving choice: it can only shrink the
   thresholded set. Evaluate against a threshold-only decode — **it must earn its place**,
   because it adds a second failure mode.

**Do not** force `|S| ≥ 1`. Ground truth is never empty, but the *system* is allowed to
abstain, and forcing an output on low-confidence tickets is precisely the behaviour the
precision-first requirement forbids. If the product insists on always showing something
(§8 Q6), evaluate it as a separate, explicitly-labelled operating point and report the
precision cost.

**Modelling label dependence** (services co-occur non-randomly) via classifier chains
([Read et al., Machine Learning 2011](https://link.springer.com/article/10.1007/s10994-011-5256-5))
is *not* recommended for phase 1: it adds ordering sensitivity and inference complexity, and
a shared encoder already captures much co-occurrence structure through the shared
representation. Revisit only if the co-occurrence matrix (§2.8) shows strong structure *and*
error analysis shows the model violating it.

### 4.7 Optional: domain-adaptive pretraining

If ≥100k unlabelled tickets exist, continued MLM pretraining on ticket text before fine-tuning
is a cheap, low-risk gain and uses Case-B data without any label contamination. **[estimate]**
1 epoch over 100k tickets × 256 tokens on the base model ≈ 6 × 85.1M × 25.6M tokens ≈
13 PFLOP ≈ 13 min on a T4 at 25% MFU. Treat as an ablation, not a requirement.

### 4.8 Handling new services added over time

The taxonomy will change. Design for it now, because the cost of not doing so is a full
retrain per service addition.

- **Structural choice:** keep the head as a plain `H → N` linear layer, and keep the
  label-index mapping in a **versioned `labels.json` artifact** shipped alongside the model.
  Adding a service appends a row; **never** reuse a retired service's index. Model version
  and label-set version must be checked for compatibility by the consumer.
- **Day 0 of a new service (no data):** serve it via the keyword rule (§3 baseline 2) or kNN
  over embeddings (§4.1 row 6) — both work from a handful of examples and need no retrain —
  and mark those suggestions with lower confidence. This is why row 6 stays in the design.
- **Day 30 (tens to low hundreds of labels):** **partial refit** — freeze the encoder, re-fit
  only the head (a one-vs-rest logistic regression over cached embeddings) so all services
  including the new one get a boundary. Minutes on CPU, no GPU needed.
- **Full retrain** when the new service exceeds ~200 labelled positives, or quarterly,
  whichever comes first.
- **Evaluation implication:** a new service starts with zero test coverage. Do not report a
  global metric that silently averages it in. The per-service table (§6.3) must show
  `n_positives` in the test set next to every number, and services below n=30 must be marked
  "insufficient data" rather than given a precision figure.
- **Renames and merges** are taxonomy changes, not new services: they invalidate historical
  labels. Require a mapping table from the product owner and apply it to the whole snapshot.

---

## 5. Training plan

### 5.1 Objective

Primary: **binary cross-entropy with logits** over 20 sigmoid outputs (binary relevance), with
per-service `pos_weight` for the long tail. Plus optional auxiliary cardinality cross-entropy
(§4.6.1), total loss `L = L_BCE + λ·L_card`.

Ablation: **Asymmetric Loss (ASL)**
([Ridnik et al., ICCV 2021](https://arxiv.org/abs/2009.14119)), which down-weights easy
negatives and hard-thresholds very-easy negatives — designed exactly for the
positive/negative imbalance of multi-label problems with few positives per sample (here 1–4
positives out of 20). Its ability to discard probably-mislabelled samples is a bonus given
§2's label noise. Search `γ⁻ ∈ {2, 3, 4}`, `γ⁺ = 0`, `m (clip) ∈ {0.05, 0.1}` as reported in
the paper. Choose BCE vs ASL on validation, not on principle.

Sample weights carry provenance (§2.3).

### 5.2 Hyperparameters

Ranges, with the reasoning that fine-tuning a 12-layer encoder on 5k–50k examples is a
small-data regime where instability is the dominant risk:

| Hyperparameter | Range / value | Source |
|---|---|---|
| Learning rate | `{1e-5, 2e-5, 3e-5, 5e-5}` | BERT/RoBERTa standard grid; [Mosbach et al., ICLR 2021](https://arxiv.org/abs/2006.04884) recommends small LR (e.g. 2e-5) for stability |
| Epochs | 10–20 with early stopping | Mosbach et al.: instability on small datasets comes from too few *iterations*, not too little data; train longer at a small LR |
| Optimizer | AdamW, **bias correction on** | Mosbach et al. identify missing bias correction (BERTAdam) as a root cause of the vanishing-gradient instability |
| Warmup | 6–10% of total steps, linear decay after | RoBERTa convention |
| Batch size | 16 or 32 | Fits easily; see §5.5 |
| Weight decay | 0.01 | Standard |
| Grad clip | 1.0 | Standard |
| Max seq len | `{128, 192, 256}` | Directly trades quality vs the §4.5 latency numbers — treat as a *product* hyperparameter |
| `pos_weight` | `{1, sqrt(neg/pos), neg/pos}` per service | Standard imbalance grid |
| λ (cardinality aux) | `{0, 0.1, 0.3}` | λ=0 is the ablation |
| B-strong sample weight | `{0, 0.3, 0.6}` | §2.1.3 |
| Precision | bf16/fp16 for XLM-R and E5; **fp32 or bf16 for mDeBERTa** | DeBERTa-v3 fp16 instability is a known practical issue |

**Search strategy:** random search, **24 trials**, selecting on validation **macro average
precision** (threshold-free, so it does not entangle model selection with threshold
selection), then tune thresholds separately on the same validation set (§6.1). Random beats
grid at this dimensionality; 24 trials on the base model at 50k tickets is ~27 GPU-hours on a
T4 [estimate from §5.5] — schedule accordingly, or cut to 12 trials on a 25% data subsample
for the first pass.

**Early stopping**: on validation macro-AP, patience 3 evaluations, evaluating every 0.5
epoch. Keep the best checkpoint by validation macro-AP, not the last.

### 5.3 What is held fixed for comparability

Across every arm (CatBoost, frozen-embedding, each checkpoint, each loss): identical data
snapshot ID, identical splits, identical PII redaction, identical text construction,
identical threshold-selection procedure, identical decode rule, identical metric code. Any
arm that changes one of these is reported separately and labelled.

### 5.4 Small-data recipe (< 5,000 labels)

- Freeze embeddings + bottom 4 layers; train top 8 + head.
- LR 2e-5, 20 epochs, batch 16, warmup 10%.
- 10 seeds instead of 5 — variance is large here and single runs are meaningless.
- Compare head-to-head against SetFit and against frozen-embeddings+logreg. **Expect the
  frozen baselines to be competitive; if they win, ship them.** That is a legitimate and
  cheap outcome, and it is the honest reading of the low-data literature (§2.4).

### 5.5 Hardware and wall-clock

**Compute arithmetic [estimate]**, using the standard `C ≈ 6·N·T` FLOPs-per-token rule for
forward+backward from [Kaplan et al., "Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361),
with `N` = **non-embedding** parameters (embeddings are frozen, §4.6), `T` = tokens, seq len
256, 10 epochs, and an assumed **25% model-FLOPs-utilisation** (a conservative figure for
small-batch encoder training; MFU as an accounting convention is from
[Chowdhery et al., PaLM](https://arxiv.org/abs/2204.02311)). Peak FP16 throughput:
T4 = 65 TFLOPS ([NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)),
A100 = 312 TFLOPS. Attention FLOPs are excluded by the 6N rule and add < 10% at T=256.

| Tickets | Model (non-emb params) | Total FLOPs | T4 | A10G | A100-40G |
|---|---|---|---|---|---|
| 5,000 | small (21.3M) | 1.6 PFLOP | 1.7 min | 0.9 min | 0.3 min |
| 5,000 | base (85.1M) | 6.5 PFLOP | 6.7 min | 3.5 min | 1.4 min |
| 20,000 | small | 6.5 PFLOP | 6.7 min | 3.5 min | 1.4 min |
| 20,000 | base | 26.1 PFLOP | 26.8 min | 13.9 min | 5.6 min |
| 50,000 | small | 16.4 PFLOP | 16.8 min | 8.7 min | 3.5 min |
| 50,000 | base | 65.4 PFLOP | **67 min** | 35 min | 14 min |
| 200,000 | base | 261 PFLOP | 268 min | 139 min | 56 min |

**Apply a 1.5–2× fudge** for data loading, evaluation passes, and imperfect MFU. So: a single
base-model run on 50k tickets is a **~1.5–2 hour T4 job**; the full 24-trial search is
**~1.5–2 GPU-days on a T4**, or ~half a day on an A100. A T4 or any ≥16 GB GPU is sufficient.

**VRAM [estimate]:** base model, embeddings frozen, batch 32 × 256 tokens, mixed precision:
weights 278M × 2 B ≈ 0.6 GB; gradients + Adam states on the 85M trainable ≈ 85M × 12 B ≈
1.0 GB; activations ≈ 2–4 GB. **Total ≈ 4–6 GB.** Fits a 16 GB T4 comfortably; an 8 GB card
works at batch 16.

### 5.6 Seeds

**5 seeds** (10 in the small-data regime) for every arm that gets reported. Fixed seed list,
fixed data order. Report **mean ± standard deviation**. A single-seed number is not a result
and must not appear in any comparison table.

---

## 6. Evaluation

### 6.1 Threshold selection (validation only)

1. Fit the model; produce scores on validation.
2. For each service `s`, sweep τ_s over the score grid and choose the **largest recall subject
   to per-service precision ≥ P_floor**, with `P_floor = 0.90` as the starting point (a
   product decision — §8 Q6 — and it may differ per service tier).
3. If no threshold achieves `P_floor` for a service, set τ_s = 1.0 (**the model never predicts
   that service**) and record it. Silently predicting a service you cannot predict precisely
   is the failure this design exists to prevent.
4. Freeze τ. Freeze the cardinality decode variant. Only then touch the test set.

Thresholds are an **artifact** (`thresholds.json`), versioned with the model, and re-tuned
whenever the model or the label set changes.

### 6.2 Metrics

- **Primary: micro-averaged recall at micro-precision ≥ 0.90 (`R@P90`)** on the gold test set,
  with thresholds fixed from validation. One number, precision-first, directly interpretable
  ("of the services that should have been suggested, we suggested X%, while 90% of what we
  showed was right").
- **Model-selection metric (threshold-free): macro average precision (macro-AP / mAP).** Used
  during the hyperparameter search so selection is not entangled with thresholding.
- Secondary, all reported: micro/macro precision, recall, F1; exact-match (subset accuracy);
  Jaccard/example-F1; **coverage** (fraction of tickets with ≥1 prediction — the price of the
  precision floor, and the number the product owner will actually care about); mean predicted
  cardinality vs true; per-service PR curves.
- **Calibration**: reliability diagram + expected calibration error on the sigmoid scores. If
  the scores are shown to agents (even as ordering), miscalibration matters. Temperature
  scaling on validation is the cheap fix if ECE is poor.
- **Human ceiling**: report Krippendorff α (§2.2) and, on the 300 double-annotated tickets,
  annotator-vs-adjudicated precision. If the model's precision approaches the humans', the
  metric has saturated and further modelling is wasted effort.

### 6.3 Slices (all reported for every candidate, all with `n`)

| Slice | Why it matters |
|---|---|
| **Per service (20 rows)** | The aggregate hides a service that is never predicted or always wrong. Rows with n < 30 marked "insufficient data", not given a number |
| **Language: RU / EN / mixed** | Russian will dominate the aggregate; the English slice is the one that silently breaks. Assign language with a script-ratio heuristic plus a langid model, and treat >15% minority-script content as "mixed" |
| **Cardinality of true label set (1 / 2 / 3 / 4)** | Multi-service tickets are the hard case and the one where the cap and cardinality head act |
| **Description length buckets** (short / medium / long, and empty description) | Short tickets are the weak point and are common |
| **Priority** | Checks that a correlated feature has not become a shortcut |
| **Time period (month)** | Drift detection; also validates the temporal split choice |
| **Label provenance (Case A / C / gold)** | Detects "we are only good where CatBoost was wrong" |

### 6.4 Statistical reporting

- Seed spread: mean ± sd over 5 seeds for every headline number.
- Sampling uncertainty: **bootstrap over test tickets, 10,000 resamples, 95% CI** on the
  primary metric, for both the candidate and the baseline; report the CI on the *difference*
  (paired bootstrap on the same tickets).
- **Test set is scored once per candidate.** A candidate is a (checkpoint, loss, decode)
  configuration finalised on validation. If you look at the test set and then change
  something, you have created a new candidate and must record it as such in the experiment
  log. The experiment log is a deliverable.

### 6.5 Error analysis plan

Mandatory before any ship decision, on validation (never test):

1. Confusion structure: for each service, which other services are predicted in its place —
   as a 20×20 matrix over false positives. Adjacent/overlapping service definitions show up
   here as symmetric hot spots and are usually a taxonomy problem, fixable by merging the
   services or sharpening the guideline.
2. Manual read of **100 highest-confidence false positives**. These are the ones agents will
   see and lose trust over. Categorise: annotation error / genuinely ambiguous / model error /
   leakage artifact.
3. Manual read of **50 false negatives at high true-label confidence**.
4. High-loss training examples as a label-noise audit (confident-learning style): if the top
   high-loss training examples are mostly mislabelled, the ceiling is label quality, and the
   next sprint is annotation, not modelling.
5. RU vs EN error comparison on matched-difficulty tickets.

### 6.6 Ship / no-ship criterion

Ship only if **all** hold:

1. **Primary**: candidate `R@P90` on the gold test set exceeds re-measured CatBoost's `R@P90`
   (§3.3) with the **paired-bootstrap 95% CI on the difference strictly above 0**, and the
   point improvement ≥ **5 percentage points absolute**. (If the improvement is positive but
   < 5pp, that is a "keep CatBoost, revisit with more labels" finding — write it up rather
   than shipping a lateral move that adds a transformer to the ops surface.)
2. **Precision floor holds on test, not just validation**: micro precision ≥ 0.88 (allowing
   2pp of val→test slippage from the 0.90 target; a larger drop means the validation set is
   too small or the split is leaking).
3. **No slice disaster**: no service with n ≥ 30 has test precision < 0.75 at the shipped
   thresholds; both RU and EN slices have precision ≥ 0.85.
4. **Coverage acceptable**: ≥ 1 prediction on ≥ *X*% of tickets, X set by the product owner
   before test evaluation (§8 Q6). Placeholder: 80%.
5. **Stability**: sd across 5 seeds on the primary metric ≤ 3pp. Higher variance than that
   means the winner was luck.
6. **Quantisation is free**: INT8 vs fp32 val `R@P90` delta ≤ 1pp. Otherwise ship fp32 or
   invest in quantisation-aware training.
7. **Ops budget met**: p95 latency and artifact size within §9.3–9.4 on the *production* CPU.
8. **Shadow gate passed**: §6.7.

### 6.7 Rollout: shadow mode, then canary

**Stage 0 — shadow (≥ 2 weeks, ≥ 2,000 tickets).** The model scores every new ticket; output
is logged only. Nothing is written to `tickets.services`, no internal message. CatBoost stays
live. *(How this is wired is the architect's call; the requirement is that the model output is
recorded with the ticket ID and timestamp and is not visible to agents.)*
**Gate:** shadow precision, computed against the human-final service set observed ≥ 7 days
later, is within **3pp** of the offline test precision. A larger gap means offline evaluation
is wrong (leakage, skew, or preprocessing mismatch) — stop and diagnose, do not proceed.

**Stage 1 — canary.** Enable for a small traffic share. Monitor: agent correction rate on
predicted sets (the live proxy for precision), coverage, latency, error rate. Compare against
CatBoost tickets in the same period.
**Gate:** agent correction rate not worse than CatBoost's, over ≥ 1,000 tickets.

**Stage 2 — full rollout**, CatBoost retained as a runtime fallback for one release cycle.

**Permanent unbiased-label holdout (required, all stages).** On **3–5%** of tickets, randomly
selected, **write no prediction at all** — no `services` value, no internal message. Those
tickets get pure human labels and become the only uncontaminated training and evaluation
stream for every future model version. This is Sculley et al.'s prescribed feedback-loop
mitigation ([NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)),
and skipping it recreates exactly the trap §2.1 exists to escape. **[estimate]** at 20k
tickets/month, 4% ⇒ 800 clean labels/month ⇒ ~10k/year at zero annotation cost.

---

## 7. Risks and failure modes

| # | Risk | How it manifests | Detection |
|---|---|---|---|
| 1 | **Old-model distillation** (§2.1) | Great offline numbers, no real-world improvement; new model repeats CatBoost's exact mistakes | Provenance slice in §6.3; agreement rate with CatBoost predictions on the gold set (suspiciously high = distillation); shadow gate §6.7 |
| 2 | **Feedback loop after launch** | Metrics improve every retrain while agent correction rate does not | The 3–5% permanent holdout (§6.7) is the ground truth; alarm if holdout precision and non-holdout precision diverge |
| 3 | **Label noise / taxonomy ambiguity** | Ceiling well below 100%, confusable service pairs | Krippendorff α (§2.2); 20×20 FP matrix (§6.5); high-loss audit |
| 4 | **Temporal drift / new services** | Precision decays month over month | Monthly slice (§6.3); score-distribution PSI monitoring; `<unk>` rate if vocabulary is trimmed; per-service prediction-rate drift |
| 5 | **Leakage via near-duplicates / per-customer templates** | Big random-vs-temporal gap; test precision far above shadow precision | Report both splits (§2.6); MinHash dedupe counts; shadow gate |
| 6 | **Target leakage from `labels`/`flags`** | Excellent offline, collapses in shadow | Populated-rate *as of creation* (§2.7); feature-ablation runs; shadow gate |
| 7 | **Degenerate solution** — predicts only the 2–3 head services | High micro metrics, terrible macro metrics, mean cardinality far below truth | Macro precision/recall, per-service prediction counts, cardinality histogram vs truth |
| 8 | **Annotation anchoring** | Gold set agrees with the old model rather than with truth | Blind annotation (§2.2) plus a deliberate A/B: 100 tickets annotated blind vs 100 with the prediction shown; if agreement differs materially, the anchoring effect is real and blindness is proven necessary |
| 9 | **Quantisation damage** | INT8 shifts scores enough to invalidate thresholds | Ship criterion 6; **re-tune thresholds on the INT8 model**, never carry fp32 thresholds over |
| 10 | **CPU without VNNI in production** | INT8 slower than fp32 (§4.5) | Confirm CPU flags before committing (§9.4); benchmark on the real host |
| 11 | **Language fairness** | English tickets get worse suggestions; small English customer base suffers invisibly | RU/EN slice with a hard gate in ship criterion 3 |
| 12 | **PII in artifacts/logs** | Ticket text in prediction logs, or memorised by the model | Redact before storage (§2.9); log hashes + scores, not raw text, wherever possible; retention policy is the architect's/legal's call |
| 13 | **Threshold overfitting** | Thresholds tuned on 1,000 val tickets do not hold on test, especially for rare services | Ship criterion 2 (val→test precision slippage ≤ 2pp); for services with < 50 val positives, prefer a shrunk/pooled threshold over a per-service one |

---

## 8. Open questions (blocking, with owner)

| # | Question | Owner | Why it blocks |
|---|---|---|---|
| **Q1** | **Is there any record of who/what last wrote `tickets.services`?** Audit table, event log, trigger history, or is the `type='internal'` prediction message parseable back to the rollout date? | Backend / DBA | Determines whether §2.1.1 works today or whether audit capture must be added and data accumulated (months). This is *the* schedule risk |
| **Q2** | Are `labels` and `flags` populated **at ticket creation**, and is their creation-time value recoverable historically? | Product owner + backend | Target-leakage risk (§2.7). Default is exclusion |
| **Q3** | Ticket volume/month, historical span, RU:EN ratio, human-correction rate on `services`, cardinality distribution | Data query, week 1 | Sets the dataset-size regime (§2.4) and therefore the entire approach |
| **Q4** | Current CatBoost artifact, its feature pipeline, its training data, its retraining cadence, and any metrics it currently reports | Whoever owns it | Baseline must be re-measured (§3.3); its features may be reusable |
| **Q5** | Is the service taxonomy stable? Who adds services, how often, are there pending renames/merges? | Product owner | Drives §4.8 and whether labels must be remapped |
| **Q6** | **Product decisions on the operating point**: acceptable precision floor (0.90?), acceptable coverage (80%?), is an empty/abstained suggestion acceptable, and is showing a low-confidence suggestion better than showing none? | Product owner | Threshold selection and the ship criterion depend on these; they must be fixed **before** test evaluation |
| **Q7** | Legal: does 152-FZ localisation apply, may ticket text be used for model training, may pseudonymised text leave the production perimeter to a training environment? | Legal / DPO | Determines whether the LLM-API baseline is even runnable and where training data may live |
| **Q8** | Production CPU model and per-replica core/RAM allocation; expected peak ticket-creation rate | `system-architect` / infra | §4.5 shows INT8 gains depend on VNNI; sizing depends on peak rate |
| **Q9** | Is there an existing annotation tool, and can 2–3 agents be allocated ~70 hours (§2.2)? | Support lead | The gold set is a hard prerequisite; without it there is no trustworthy evaluation and the project cannot conclude anything |
| **Q10** | Is retraining expected to be automated (cadence, trigger) or manual? | Product owner + architect | Affects artifact/versioning requirements in §9 |

Q1, Q3, Q6 and Q9 block the start. The rest block the ship decision.

---

## 9. Implementation notes and inference requirements

These are **constraints for `system-architect`**, who owns where inference runs, how it is
called, retry/failure semantics, schema changes, and the Temporal workflow design. I am
deliberately not specifying any of those.

### 9.1 Artifacts to produce

| Artifact | Contents | Approx size |
|---|---|---|
| `model.onnx` (INT8 dynamic) | Encoder + classification head + cardinality head, opset ≥ 17, dynamic axes for batch and sequence | **279 MB** base / **118 MB** small [measured]; ~116 MB / ~37 MB if vocabulary-trimmed [computed] |
| `tokenizer.json` | Fast tokenizer, exact match to training | 5–17 MB |
| `thresholds.json` | Per-service τ, decode variant, P_floor used, validation metrics at that point | < 10 KB |
| `labels.json` | Ordered service list + label-set version; indices are append-only (§4.8) | < 5 KB |
| `preprocess.json` | PII redaction rules, normalisation rules, text template, max_len, truncation strategy, E5 prefix flag | < 50 KB |
| `metrics.json` + report | All §6 metrics, all slices, seed spread, bootstrap CIs, learning curve | small |
| `model_card.md` | Training data snapshot ID, provenance composition, known failure modes, intended use | small |

The **learning curve** (§2.4) is a required deliverable, not an optional plot.

### 9.2 Input / output contract

Input (per ticket):
```
{ "ticket_id": str,
  "title": str,                 // may be empty
  "description": str,           // may be empty; not both empty
  "priority": str|null,
  "created_at": iso8601 }
```
Output:
```
{ "ticket_id": str,
  "services": [ {"name": str, "score": float} ],   // 0..4 items, score-descending
  "abstained": bool,                               // true iff services == []
  "predicted_cardinality": int|null,               // 1..4 if the aux head is shipped
  "model_version": str,
  "label_set_version": str,
  "threshold_version": str }
```
Requirements on the consumer:
- **Do not re-threshold.** Thresholds live with the model. Scores are provided for display and
  logging only.
- **Must handle `services: []`.** Abstention is a normal, expected outcome.
- **Must reject on version mismatch** between `label_set_version` and its own service enum
  rather than mapping by position.
- **All preprocessing (PII redaction, normalisation, template) runs inside the model service**,
  not in the caller — identical code paths for training and inference are the only defence
  against train/serve skew.

### 9.3 Latency and throughput budget

Based on **[measured]** §4.5, on 4 vCPU with AVX-512-VNNI, INT8, 256 tokens:

| Metric | Base model (recommended) | Small model (fallback) |
|---|---|---|
| Encoder p50, batch 1 | 42.9 ms | 25.6 ms |
| Encoder p95, batch 1 | ~50 ms (excluding a noisy outlier) | 31.6 ms |
| + tokenisation & pre/post | +3–5 ms [estimate] | +3–5 ms [estimate] |
| **Requested p95 budget for the model call** | **≤ 250 ms** (≈5× headroom for a shared/noisy host) | ≤ 150 ms |
| Throughput, batch 16, 4 threads | 26 tickets/s | 61 tickets/s |
| Cold start (session init + first inference) | ~1–3 s [estimate] | ~1–3 s [estimate] |

Notes for the architect:
- The prediction is asynchronous relative to the user (it runs in a workflow at ticket
  creation), so **this is a throughput problem, not a user-facing latency problem**. Latency
  headroom can be traded for cheaper hardware.
- Batching gives a large throughput win (26 vs ~23 tickets/s single-stream at 256 tokens for
  the base model, and 61 vs 39 for the small one) but I am not prescribing whether batching
  happens — that is a topology decision.
- Threads: set `intra_op_num_threads` explicitly to the allocated core count and
  `inter_op_num_threads = 1`. Leaving ORT to auto-detect on a container with a CPU limit
  causes oversubscription and unstable latency.

### 9.4 Hardware requirements

- **CPU: x86-64 with AVX-512-VNNI strongly preferred.** [measured] INT8 gives 1.4–2.5× on a
  VNNI CPU; without VNNI it can regress ([onnxruntime #12854](https://github.com/microsoft/onnxruntime/issues/12854)).
  **If production CPUs lack VNNI, tell me and the recommendation changes** (ship fp32 small,
  or trim vocabulary and accept lower throughput). AVX2-only hosts should be benchmarked
  before sizing.
- **RAM: ≥ 1 GB per replica** (measured peak RSS 503 MB base INT8 / 258 MB small INT8, plus
  runtime); ≥ 2 GB if batching.
- **Cores: 2–4 per replica.** [measured] 4→2 threads costs ~1.7× latency; 4→1 costs ~3×.
  Below 2 cores the base model is not attractive.
- **No GPU in production.** GPU is training-only: any ≥16 GB card (T4 sufficient); §5.5 for
  wall-clock.

### 9.5 Reproducibility requirements

- **Dataset snapshot**: immutable export with a content hash; every metrics report references
  it. Never train against a live query.
- **Pinned versions**: `torch`, `transformers`, `onnxruntime`, `optimum`, `tokenizers`,
  `scikit-learn`, `catboost` — exact versions in a lockfile, recorded in `model_card.md`.
  (Benchmarks here used `onnxruntime` 1.28.0.)
- **Seeds**: fixed list, recorded per run; data ordering seeded; cuDNN determinism enabled for
  the final runs (accepting the throughput cost).
- **Experiment log**: one row per run — config hash, data snapshot hash, seed, val metrics,
  and whether the test set was touched. Test-set evaluations are logged explicitly (§6.4).
- **Export determinism**: the ONNX export and the INT8 quantisation must be scripted and
  re-runnable; the quantised model's validation metrics are re-computed after export and
  stored, and thresholds are re-tuned **on the exported INT8 model**.

### 9.6 Scratch work backing this spec

Throwaway benchmark script and downloaded ONNX encoders (not part of any deliverable, not to
be imported by anything):
`/tmp/claude-0/-home-user-bert-poc/78b1e29a-9270-5987-8575-f9e218c9f04d/scratchpad/bench_cpu.py`
(results in `bench_results.json`, models under `m/`). Parameter-count and vocabulary-trimming
arithmetic was computed inline from each checkpoint's published `config.json`.

---

## 10. Suggested sequencing

1. **Week 1 — answer Q1, Q3, Q6, Q9.** Run the provenance reconstruction query (§2.1.1),
   report Case A/B/C/D counts, class balance, cardinality histogram, RU:EN ratio, token-length
   distribution. *This report alone determines whether the project is a 3-week or a 4-month
   project — deliver it before writing any training code.*
2. **Week 1–2 — annotation guideline + gold set** (§2.2), and the trivial + TF-IDF baselines
   in parallel. Re-measure CatBoost (§3.3).
3. **Week 3 — frozen-embedding baseline** (§4.1 row 2). Cheap, and it sets the bar the
   fine-tuned model must clear.
4. **Week 3–4 — fine-tuning sweep** (§5), learning curve, INT8 export, threshold tuning.
5. **Week 5 — evaluation, error analysis, single test-set scoring, ship decision** (§6).
6. **Week 6+ — shadow mode** (§6.7).
