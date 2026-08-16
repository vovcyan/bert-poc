# Dataset construction runbook

Status: executable procedure — this is the week-1 deliverable
Companion to: [`ticket-services-classifier.md`](./ticket-services-classifier.md) §2, and the [proposal](../proposals/ticket-services-classifier.md) §3

This document answers three questions: how to build a trustworthy dataset from a `services`
column that mixes CatBoost output with human corrections, how many records we need, and how to
select them.

**Assumed schema** (adjust names to match production):
`tickets(id, title, description, priority, services text[], labels, flags, created_at, updated_at, organization_id)`,
`ticket_messages(id, ticket_id, type, body, author_id, created_at)`.

---

## 0. The governing principle

**Training data and evaluation data have opposite selection rules, and mixing them up is the
single most common way this project fails.**

| | Training set | Evaluation set (val + test) |
|---|---|---|
| Goal | maximum usable signal | an unbiased estimate of real-world precision |
| Selection | **filtered** — take the tickets whose labels we can trust | **stratified random over all tickets** |
| Label source | reconstructed provenance (Case A / C), optionally weak labels | **blind human re-annotation only** |
| Bias tolerated | yes, correct it with weights | **no — bias here invalidates the whole project** |

If you take one thing from this document: **you cannot build the test set by filtering on
"tickets a human corrected."** Those tickets are, by construction, the ones the old model got
wrong. Measuring on them tells you how you do on hard cases only, and the number you report
will not resemble production.

---

## 1. Step 1 — audit the internal messages before trusting anything

The `type='internal'` prediction message is our provenance reconstruction. Two things can
break it, and both must be checked before any dataset is built.

### 1.1 `type='internal'` is probably not only predictions

Agents very likely post their own internal notes with the same type. If so, a naive
`WHERE type='internal'` treats a human note as a model prediction and mislabels the ticket.
**Anchor on the author, not just the type.**

```sql
-- Who writes internal messages? Expect a bot/system account for predictions.
SELECT author_id, count(*) AS n, min(created_at), max(created_at)
FROM ticket_messages
WHERE type = 'internal'
GROUP BY author_id
ORDER BY n DESC
LIMIT 20;
```

If predictions have no distinguishable author, fall back to a strict format anchor (the exact
prefix/template the workflow emits) and treat everything else as a human note. Record the rule
you used — it becomes part of the dataset snapshot definition.

### 1.2 Format drift over time

The list is stored "as text", and text formats drift silently across releases.

```sql
-- Eyeball the format across the whole period before writing a parser.
SELECT date_trunc('quarter', created_at) AS q, left(body, 200) AS sample
FROM ticket_messages
WHERE type = 'internal' AND author_id = :system_actor
ORDER BY q, random()
LIMIT 60;
```

Then write the parser in Python (not plpgsql — it will need iteration), run it over a sample of
200 messages per month, and report **parse-success rate per month**.

> **Gate: reject any month with < 95% parse success.** Do not silently drop failures — an
> unparseable month that you exclude is a hole in your temporal coverage, and you need to know
> about it.

### 1.3 Multiple predictions per ticket

Workflow retries or re-runs can produce more than one prediction message. Always take the
**earliest** one — that is the prediction the human first reacted to.

---

## 2. Step 2 — partition the corpus

Materialise the parse into a work table rather than re-parsing in every query.

```sql
CREATE TABLE work_provenance AS
WITH first_pred AS (
  SELECT DISTINCT ON (tm.ticket_id)
         tm.ticket_id,
         tm.created_at AS predicted_at,
         tm.body       AS pred_text
  FROM ticket_messages tm
  WHERE tm.type = 'internal'
    AND tm.author_id = :system_actor        -- §1.1
  ORDER BY tm.ticket_id, tm.created_at      -- earliest, §1.3
)
SELECT t.id AS ticket_id,
       t.created_at,
       t.services              AS current_services,
       fp.pred_text,
       fp.predicted_at,
       NULL::text[]            AS predicted_services,  -- filled by the Python parser
       NULL::text              AS provenance_case
FROM tickets t
LEFT JOIN first_pred fp ON fp.ticket_id = t.id;
```

Normalise before comparing — array order and duplicates are not meaningful here:

```sql
CREATE OR REPLACE FUNCTION norm_services(a text[]) RETURNS text[] AS $$
  SELECT array(SELECT DISTINCT lower(trim(x)) FROM unnest(coalesce(a, '{}')) AS x ORDER BY 1);
$$ LANGUAGE sql IMMUTABLE;
```

Then assign cases:

```sql
UPDATE work_provenance SET provenance_case =
  CASE
    WHEN pred_text IS NULL                                            THEN 'C'  -- no prediction
    WHEN predicted_services IS NULL                                   THEN 'D'  -- unparseable
    WHEN norm_services(predicted_services) <> norm_services(current_services) THEN 'A'  -- corrected
    ELSE 'B'                                                                    -- unchanged
  END;
```

**The report to produce (this is the week-1 deliverable):**

```sql
SELECT date_trunc('month', created_at) AS month,
       count(*) AS total,
       count(*) FILTER (WHERE provenance_case = 'A') AS corrected,
       count(*) FILTER (WHERE provenance_case = 'B') AS unchanged,
       count(*) FILTER (WHERE provenance_case = 'C') AS no_prediction,
       count(*) FILTER (WHERE provenance_case = 'D') AS unparseable,
       round(100.0 * count(*) FILTER (WHERE provenance_case = 'A')
             / nullif(count(*) FILTER (WHERE provenance_case IN ('A','B')), 0), 1) AS correction_rate_pct
FROM work_provenance
GROUP BY 1 ORDER BY 1;
```

`correction_rate_pct` is the number that sizes the entire project — see §5.

### 2.1 What each case is worth

| Case | Label trust | Use |
|---|---|---|
| **A — corrected** | high for what is *present*, uncertain for what is *absent* (§3) | **primary training data**, weight 1.0 |
| **B — unchanged** | unknown — agreement or nobody looked | not supervised training; usable as unlabelled text for domain-adaptive pretraining |
| **C — no prediction** | high and **unbiased** | **the most valuable data you have**, if in-distribution (§2.2) |
| **D — unparseable** | none | quarantine, count, report |

### 2.2 Case C is the prize — check for it first

Tickets created before the CatBoost rollout have human-authored services and **no model
contamination at all**. They are the only large clean source that requires no annotation
budget.

```sql
-- When did CatBoost go live? Look for the onset of prediction messages.
SELECT date_trunc('week', created_at) AS wk, count(*)
FROM ticket_messages
WHERE type='internal' AND author_id = :system_actor
GROUP BY 1 ORDER BY 1 LIMIT 20;
```

The catch is **in-distribution-ness**. If the rollout was 3 years ago, pre-rollout tickets may
predate product changes and taxonomy changes that make them misleading. Rule of thumb: use
Case C freely if it is within ~18 months of today; beyond that, use it but hold out a slice and
check that a model trained with it does not do *worse* on the recent gold test set.

Also verify Case C tickets actually have services set — some may be untriaged rather than
human-labelled:

```sql
SELECT count(*) FILTER (WHERE cardinality(coalesce(current_services,'{}')) = 0) AS empty,
       count(*) FILTER (WHERE cardinality(coalesce(current_services,'{}')) > 0) AS populated
FROM work_provenance WHERE provenance_case = 'C';
```

---

## 3. Step 3 — the subtlety in Case A: corrections may be incomplete

**Case A tells you what a human *left*, not necessarily the complete correct set.**

An agent who sees a wrong service in the list will remove it. Whether that same agent also adds
a service the model *missed* is a behavioural question, not a technical one — and if they
typically do not, then Case A labels have reliable positives and **unreliable negatives**. Train
on them with plain binary cross-entropy and you actively teach the model that genuinely-correct
services are negatives.

This is measurable, and measuring it is cheap:

> **Include ~200 Case A tickets and ~200 Case B tickets inside the blind gold sample (§4).**
> Then compare the blind human labels against the stored values on exactly those tickets. You
> get two numbers you cannot obtain any other way:
> - **Case A label precision and recall** — if recall is low (say < 0.85), corrections are
>   removal-only, and Case A must be trained as a positive-unlabelled problem (mask the
>   negatives, or use a loss tolerant of missing labels) rather than as complete supervision.
> - **Case B agreement rate** — if the stored set matches the blind human set ≥ 80% of the time,
>   B-strong (§3.1) is usable as weak supervision. If not, drop it and say so.

Do not skip this. It costs 400 of the ~3,300 annotations you were already paying for, and it
determines how you train on the majority of your data.

### 3.1 B-strong: rescuing some of Case B

Case B will be most of the corpus. Restrict it to tickets where an agent *demonstrably engaged
after the prediction was posted*:

```sql
-- B-strong candidates: prediction exists, unchanged, but a human clearly worked the ticket.
SELECT w.ticket_id
FROM work_provenance w
WHERE w.provenance_case = 'B'
  AND EXISTS (
    SELECT 1 FROM ticket_messages m
    WHERE m.ticket_id = w.ticket_id
      AND m.author_id <> :system_actor
      AND m.created_at > w.predicted_at
  );
```

Use it as noisy-positive supervision at **sample weight ~0.3**, and only if the 200-ticket audit
in §3 clears 80% agreement. Otherwise discard.

---

## 4. Step 4 — the gold set (this is the backbone)

Everything above reconstructs *training* labels. The evaluation set must be built differently.

### 4.1 Selection rules

1. **Stratified random over all tickets in the most recent window** — not filtered on Case A,
   not filtered on anything correlated with the old model's behaviour.
2. **Strata: month × language × predicted-cardinality × provenance case.** Note what is
   *missing*: do **not** stratify on the stored service values. Doing so bakes the old model's
   class distribution into your test set.
3. **Oversample the diagnostic strata** — the ~200 Case A and ~200 Case B tickets from §3 — and
   record them so they can be down-weighted back to the true distribution when computing
   headline metrics.
4. **Blind.** Annotators see title + description + priority, in the same form the model sees.
   They must **not** see the stored `services` value or the prediction message. Anchoring is
   real and it will make your gold set agree with CatBoost rather than with the truth.
5. **Deterministic and reproducible** — sample with a seeded hash, not `random()`, so the
   snapshot can be rebuilt exactly.

```sql
-- Reproducible stratified sample. Repeat per stratum with the appropriate quota.
SELECT ticket_id
FROM work_provenance
WHERE created_at >= now() - interval '12 months'
  AND ticket_id NOT IN (SELECT ticket_id FROM work_near_duplicates)   -- §4.3
ORDER BY md5(ticket_id::text || 'gold-v1')
LIMIT :quota;
```

### 4.2 Volumes

| Split | Tickets | Why this size |
|---|---|---|
| **Test** | **2,000** | ~2.2 services/ticket ⇒ ~4,400 positive labels. A service at 5% prevalence gets ~100 positives — enough for a precision estimate with roughly ±8pp bootstrap CI. Below that, services must be reported as "insufficient data", not given a number |
| **Validation** | **1,000–1,500** | threshold tuning is per-service; at 1,000 tickets a 5% service has only ~50 positives, which is thin. Prefer 1,500 if the annotation budget allows; otherwise use pooled/shrunk thresholds for rare services |
| **Double-annotated overlap** | **300** | inter-annotator agreement (Krippendorff α with MASI distance). **Gate: α ≥ 0.67** — below that the taxonomy is the problem, not the model |
| **Older drift slice** | **300** | sampled from 9–12 months back; measures how fast performance decays |
| **Total** | **~3,300 annotations** + 300 duplicates | ≈ 60 s each ⇒ ~55 person-hours, ~70 h including guideline authoring and adjudication |

Test and validation come from the **most recent** period, with test more recent than validation.

### 4.3 Deduplicate before sampling, not after

Support corpora are full of near-duplicates: templates, auto-generated tickets, one customer
filing forty similar reports. Sample 2,000 rows naively and you may have 300 effectively
independent examples, which quietly destroys your confidence intervals.

- MinHash/LSH on character 5-grams, Jaccard ≥ 0.8 → collapse to one representative.
- **Group by `organization_id`** when splitting: one customer's tickets must not straddle
  train and test, or you leak.
- Report how many were dropped. A large number is itself a finding about the corpus.

---

## 5. How many records do we need, and how many must exist in the DB?

Two different numbers, and the second is the one people forget.

### 5.1 How many labelled tickets the model needs

| Usable human-labelled tickets | Approach | Expectation |
|---|---|---|
| < 1,000 | **Do not fine-tune.** Frozen embeddings + logistic regression, or SetFit | at or below CatBoost |
| 1,000 – 5,000 | SetFit or frozen head; fine-tuning viable but unstable — freeze bottom layers, 10 seeds | parity to modest gain |
| **5,000 – 20,000** | **fine-tuned encoder — the target zone** | clear gain expected |
| > 20,000 | fine-tuned encoder, larger backbone worth testing | best case |

**The honest answer to "how many do we need to beat CatBoost" is that it must be measured**, and
the measurement is cheap: train on 10/25/50/100% of the data, 3 seeds each, plot validation
recall-at-90%-precision against training size. The **slope at 100%** tells you whether buying
more annotation is worth it. That learning curve is a required deliverable, not an optional
plot — it is the most decision-relevant output of the whole project.

### 5.2 How many raw DB records must exist to yield that

Case A accrues at `volume × correction_rate`. So the raw corpus you need to scan is set by the
**correction rate**, not by the model:

| Correction rate `r` | Tickets needed for 5,000 Case A | for 15,000 Case A |
|---|---|---|
| 5% | 100,000 | 300,000 |
| 10% | 50,000 | 150,000 |
| 15% | 33,000 | 100,000 |
| 25% | 20,000 | 60,000 |

Add Case C on top — it needs no annotation and no correction rate, so if a decent pre-rollout
corpus exists, this table stops mattering.

**If the numbers do not reach 5,000**, the options in order of preference are: (1) use Case C,
(2) extend the window further back and accept some drift, (3) fall back to frozen embeddings +
logistic regression, which works from ~500 labels, (4) buy more annotation, (5) wait — at
volume *V* and rate *r* you accrue `V·r` per month, so at 20k/month and 15% that is 3,000/month.

Option (5) is the one to avoid needing, which is why the audit tables in the proposal §5.4
should be built in week 1 regardless of the modelling outcome.

---

## 6. Step 5 — splits

**Temporal, grouped, with a gap.**

- Sort by `created_at`. Train = oldest 70%, validation = next 15%, test = most recent 15%.
- **Gap of ≥1 week** between train/val and val/test. An incident produces dozens of
  near-identical tickets within hours; without a gap they straddle the boundary.
- Group by `organization_id` (§4.3).
- Near-duplicate removal **across the split boundary** — drop the later copy from test.

Why temporal rather than random: services get added and renamed, the product surface changes,
and phrasing drifts. A random split measures a task the model will never face.

> **Required diagnostic:** report both random-split and temporal-split validation numbers, once.
> **The gap between them is your drift magnitude**, and it is a number the product owner should
> see — it predicts how fast the model will decay and therefore how often you must retrain.

---

## 7. Order of operations

1. Audit internal-message authorship and format drift (§1). **Gate: ≥95% parse rate per month.**
2. Build `work_provenance`, produce the monthly A/B/C/D report and the correction rate (§2).
   *This report determines whether the project is 6 weeks or 4 months.*
3. Check for Case C and its rollout date (§2.2).
4. Measure class balance, cardinality histogram, RU:EN ratio, token-length distribution.
5. Deduplicate, then draw the gold sample including the 200+200 diagnostic strata (§4).
6. Write the annotation guideline; annotate blind; adjudicate; compute α.
7. **Only now**: compute Case A label precision/recall and Case B agreement (§3), and decide
   how Case A and B-strong are used in training.
8. Freeze the dataset snapshot with a content hash. Never train against a live query.

Steps 1–4 are queries and cost a few days. Steps 5–6 are the annotation budget. **Step 7 is the
one that will change how you train**, and it is free once step 6 is done.
