# LLM fallback policy

Status: design decision — when an LLM is used, and when it is not
Companion to: [`ticket-services-classifier.md`](./ticket-services-classifier.md), [proposal §4](../proposals/ticket-services-classifier.md)

The proposal rejects an LLM **on the main prediction path**. This document specifies the
paths where it is the right tool anyway, the exact trigger conditions, and the guardrails
that keep it from re-contaminating the dataset.

---

## 1. The principle

**The LLM is a complement on paths the classifier structurally cannot serve — never a
replacement on the path it serves well.**

The argument against an LLM on the main path was never "LLMs are bad at classification". It
was that your product constraint (a wrong service costs more than a missing one) requires a
**calibratable score per service**, and generative text has no threshold. That argument holds
on the main path and does not hold on these four paths, for a specific reason each time:

| Path | Why the classifier can't serve it |
|---|---|
| A new service with no training data | It has no head row, no examples, and no threshold. The classifier cannot emit it at all. |
| Abstention | By construction, nothing cleared a threshold. There is no score left to use. |
| The near-threshold band | Scores exist but sit below the precision floor. Discarding them throws away real signal. |
| Out-of-distribution input | The scores exist but are not trustworthy, and the model has no way to know that. |

Each of these is a case where the classifier's output is **absent or known-unreliable**, so
"the LLM has no calibrated score" is no longer a comparison it loses — there is nothing to
compare against.

---

## 2. Trigger conditions

Each case is independently gated by a config flag, and each is measured separately (§6).

| # | Case | Trigger condition | Mode (§4) | Priority |
|---|---|---|---|---|
| **1** | **New / cold service** | service is in `labels.json` but has `< 200` training positives, **or** its tuned threshold `τ_s == 1.0` (§6.1 of the spec — the model was never able to predict it precisely) | constrained adjudication, scoped to that service only | **Highest — do this one first** |
| **2** | **Near-threshold band** | some service scores in `[τ_s − δ, τ_s)`, with `δ ≈ 0.10` tuned on validation | constrained adjudication over the shortlist | High |
| **3** | **Abstention** | decoded set is empty after thresholding and the top-4 cap | open prediction over the full taxonomy, marked low-confidence | Medium — product decision required |
| **4** | **Out-of-distribution input** | any of: langid says neither RU nor EN with confidence; token length > `3 × max_len` (heavy truncation); description empty and title below N tokens; embedding distance to the nearest training centroid above the validation 99th percentile | open prediction, marked low-confidence | Medium |
| **5** | **Classifier service unavailable** | classifier fails after retries | whatever mode the ticket would otherwise have used | Infra, not ML |

### 2.1 Case 1 is the one that pays for itself

The proposal's §4.8 currently covers new services with keyword rules and kNN retrieval. An
LLM does that job strictly better: **you can put the service's written definition straight
into the prompt** — the same one-paragraph definition and 3-positive/2-negative examples the
annotation guideline already contains (§2.2 of the spec). No training data, no embedding
index, no retrain, available the hour the service is created.

This is also the narrowest and safest case, because it is scoped to *one named service* at a
time rather than the whole taxonomy. Build this one first; treat the rest as optional.

### 2.2 Case 5 ordering

Classifier → CatBoost → LLM → no prediction.

CatBoost sits ahead of the LLM while it is still deployed: it is trained on your actual
taxonomy, costs microseconds, and is already there. Once CatBoost is decommissioned (one
release cycle after full rollout, per the proposal §5.5), the LLM becomes the only fallback
and moves up.

---

## 3. Why CPU feasibility flips on the fallback path

I rejected a self-hosted LLM on the main path on CPU-throughput grounds. **That objection
does not transfer, because fallback traffic is roughly two orders of magnitude smaller.**
The arithmetic is worth showing, because it changes the answer:

At 20,000 tickets/month with a 20% abstention rate, case 3 fires on ~4,000 tickets/month —
**0.0015 tickets/s**. A small quantized LLM on CPU needs perhaps 20–30 s per ticket
(prefill dominates: ~2,000 prompt tokens, then ~100 output tokens). That is

```
4,000 tickets × 30 s = ~33 CPU-hours/month
```

against 720 hours available on a single worker — **under 5% utilization**. Cases 1, 2 and 4
are rarer still. Burst tolerance is free for the same reason it is on the main path:
prediction is asynchronous behind Temporal, so a spike queues and drains.

So a self-hosted LLM is genuinely viable here even under a strict data-residency reading.
Sizing it properly needs a benchmark on the real host, as with the encoder — but the
feasibility question is settled by the volume, not by the model.

---

## 4. Two modes, and why the first is strongly preferred

### 4.1 Constrained adjudication (cases 1, 2)

Give the model the ticket **plus a shortlist of candidate services with their definitions**,
and ask for an accept/reject verdict on each candidate. The output space is closed, small,
and auditable.

Force it through **structured outputs** so the shape is guaranteed rather than hoped for:

```python
output_config={
    "format": {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "enum": SERVICE_NAMES},
                            "accept": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["service", "accept", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        },
    }
}
```

`enum` over the service names is supported by structured outputs, so **an invalid service
name is not a failure mode you have to handle** — it cannot be emitted. Requiring an
`evidence` span (a quote from the ticket) is a cheap, effective grounding constraint and
gives reviewers something to audit.

Two things this does **not** buy you, stated plainly:

- **It constrains shape, not confidence.** A self-reported confidence field would not be
  calibrated and should not be thresholded. The operating point comes from measuring
  accept-precision per service on the gold set (§6), not from anything the model says about
  itself.
- **You cannot get sampling variance from `temperature` any more.** It is rejected with a
  400 on current models. If you want N-of-M self-consistency, vary the *prompt* — shuffle
  the candidate order across the N calls — rather than resampling at a different
  temperature. Require 2-of-3 agreement to accept; this is the closest thing to a threshold
  available on this path, and it is measurable.

### 4.2 Open prediction (cases 3, 4)

Same structured-output schema, but the candidate list is the full taxonomy. Strictly
weaker: a larger output space, no shortlist to anchor on, and no classifier signal to
combine with. Reserve it for the cases where there genuinely is no shortlist, and always
mark the result low-confidence.

### 4.3 Prompt caching pays for itself here

The ~20 service definitions are a **stable prefix** and the ticket text is the varying
suffix — the textbook shape for prompt caching. Put the definitions and instructions first,
put the ticket last, and set the cache breakpoint on the last definition block. Cache reads
run at roughly a tenth of input price.

One gotcha worth measuring before you pick a model: the **minimum cacheable prefix is
model-dependent and not monotonic** — 512 tokens on Claude Opus 5, 1024 on Sonnet 5, but
**4096 on Haiku 4.5**. Twenty definitions without examples land around 1,500–2,000 tokens,
which caches on Opus 5 and Sonnet 5 and **silently will not cache on Haiku 4.5** — no error,
just `cache_creation_input_tokens: 0` and full price on every call. Count the real prefix
before committing.

---

## 5. The contamination rule — non-negotiable

**Every LLM-produced prediction is model provenance, exactly like CatBoost output, and is
excluded from supervised training on the same terms.**

This is the whole point of §3 of the proposal, and it is easy to lose. Concretely:

- `service_predictions.model_kind` gains an `llm` value alongside `catboost` and
  `transformer`. The training query filters on provenance, not on model name, so a new
  model kind is excluded by default rather than by remembering to add it.
- LLM output written into `tickets.services` produces a `ticket_services_audit` row with
  `actor_type = 'model'`. It is never Case A.
- **The LLM must never touch the gold set** — not as an annotator, not as a pre-annotator,
  and not to pre-filter which tickets get annotated. Pre-annotation reproduces the exact
  anchoring problem that makes blind annotation necessary; pre-filtering changes the
  sampling distribution and biases the test set. There is no safe version of this.
- The permanent 3–5% holdout stays holdout. It receives no prediction from any model,
  including this one.

If the LLM fallback ships and this rule is not enforced, the next model generation inherits
a corpus contaminated by *two* models instead of one, and the provenance partition stops
working.

---

## 6. Evaluation and the kill switch

**Each case gets its own precision measurement.** An aggregate "LLM fallback precision"
number is meaningless — the cases have different inputs, different modes, and different
baselines.

| Case | Measured against | Ships only if |
|---|---|---|
| 1 — new service | the gold set, restricted to that service | precision ≥ the same floor the classifier must clear (0.90) |
| 2 — near-threshold | gold tickets whose scores fell in the band | accepting the LLM's verdicts raises recall **without** dropping micro-precision below the floor |
| 3 — abstention | gold tickets where the classifier abstained | precision ≥ a **separately agreed, explicitly lower** floor — this is a distinct operating point and must be reported as one |
| 4 — OOD | gold tickets flagged OOD by the detector | precision ≥ the case-3 floor, **and** the OOD detector's own precision is reported |

Three standing rules:

1. **Each case has an independent enable flag.** They fail differently and should be
   switched off independently.
2. **Auto-disable on drift.** Compute rolling precision per case against the human-final
   service set observed ≥7 days later. If a case drops below its floor over a 200-ticket
   window, disable that case and alert. This is the same signal the main model uses.
3. **LLM-sourced suggestions are visibly marked** in the internal `ticket_messages` row.
   Agents should be able to tell a threshold-backed suggestion from a low-confidence one,
   and you need the marker to compute the per-case metrics at all.

---

## 7. Deployment and data residency

| Option | Fit | Blocker |
|---|---|---|
| **Self-hosted small LLM on CPU** | Viable at fallback volume (§3). No data leaves the perimeter. | Needs a benchmark on the real host; adds a second model artifact to version and monitor |
| **Self-hosted on a small GPU** | Comfortable, simpler to size | Contradicts "CPU only in production" — needs a deliberate exception, and at this volume it buys latency you don't need |
| **Hosted API (Claude)** | Best quality, zero ops, structured outputs and prompt caching out of the box | **Blocked pending legal (Q8/§2.9 of the spec).** 152-FZ localisation constrains sending Russian citizens' personal data abroad. PII redaction helps but does not by itself settle it |

**Cost, if legal clears the hosted path** — this is not a factor in the decision. At 4,000
fallback tickets/month with ~2,000 cached prompt tokens, ~500 ticket tokens and ~150 output
tokens per call:

| Model | Input $/MTok | Output $/MTok | Est. monthly (with caching) |
|---|---|---|---|
| Claude Opus 5 | $5.00 | $25.00 | ~$25–40 |
| Claude Sonnet 5 | $3.00 | $15.00 | ~$15–25 |
| Claude Haiku 4.5 | $1.00 | $5.00 | ~$10–15 *(but see the 4096-token cache minimum in §4.3 — without caching this is closer to $20)* |

**Recommend Claude Opus 5** and let the gold-set measurement decide whether a cheaper tier
holds the precision floor. At these volumes the spread between tiers is tens of dollars a
month, which is far below the cost of a precision regression on an agent-visible field —
picking the cheap tier to save $20/month is the wrong trade here. Run the sweep on the gold
set rather than assuming either direction.

**Latency budget: 30 s p95** for the fallback call — the path is asynchronous, and this is
generous on purpose so that hardware choice stays open.

---

## 8. Workflow integration

The fallback is a **second activity**, not a branch inside the classifier activity:

```
predictServices (classifier)
  → decide(result)            # pure function, no I/O — testable
  → predictServicesLlm(...)   # separate Temporal activity, own retry policy and timeout
  → persistPrediction(...)    # unchanged, but records source per service
```

Keeping it a separate activity means the Temporal history shows exactly which path ran, the
LLM's slower timeout doesn't distort the classifier's retry policy, and a fallback outage
degrades to classifier-only rather than failing the workflow.

`persistPrediction` records **the source per service**, not per ticket: a ticket can carry
two threshold-backed services and one LLM-adjudicated one, and the metrics in §6 need that
granularity.

---

## 9. When not to use it

- **Not on the main path.** The precision argument stands: no calibrated per-service score,
  no enforceable precision floor.
- **Not as a training-label source.** Distilling an LLM has the same feedback-loop failure
  as distilling CatBoost, and adds a third provenance class to untangle later (§5).
- **Not for gold-set annotation or pre-annotation.** See §5 — there is no safe version.
- **Not as a CatBoost replacement while CatBoost still exists.** It is slower, costlier, and
  not trained on your taxonomy.
- **Not before the classifier ships.** Every case here is defined relative to the
  classifier's thresholds and training coverage. Building the fallback first means building
  against conditions that don't exist yet.

The one exception to that last point: if the week-1 data report (§7 of the runbook) comes
back with **under ~1,000 usable labels**, LLM zero-shot becomes a legitimate *interim
production system* while annotation accumulates — with every prediction tagged `llm` in the
audit tables, and the 3–5% holdout enforced from day one rather than retrofitted.
