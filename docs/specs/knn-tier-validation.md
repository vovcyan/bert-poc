# Tier 4 (embedding kNN + UMAP) — adversarial validation

**Status:** evaluation of an existing specification, not a specification. Nothing here is
implemented. It is a recommendation to act on
[`dataset-pipeline-cleaning.md` §4.3.4](./dataset-pipeline-cleaning.md),
[`dataset-pipeline-architecture.md` §3.3/§4.3/§4.8](./dataset-pipeline-architecture.md) and
[`finetuning-dataset-pipeline.md` §3.1/§4.3](../proposals/finetuning-dataset-pipeline.md).

---

## 0. Verdict, first paragraph

**Cut tier 4. Both halves of it.** The per-row kNN flagging is a weaker, noisier, unanchored
restatement of tier 3: at a matched reviewer budget of 185 rows, spending the whole budget on
tier 3's ranking recovers **68.3% ± 2.2** of injected label errors, while spending 129 on tier 3
and 56 on tier 4 recovers **58.9% ± 1.5** — tier 4 makes the cascade *worse* at fixed cost, on
both noise processes I injected, at every budget from 25 to 500 rows [measured, 3 seeds]. Its
flag list is 29% one auto-alert template, it is structurally biased toward single-label and
rare-service rows, 17 of its top 56 rows are *less* neighbour-disagreeing than a random-neighbour
null model for their own label set, and dropping its unique flags from training costs the same
macro-AP as dropping the same number of rows at random (−0.42pp vs −0.28pp, 3 seeds) — the same
"hard-but-correct" character that already reframed tier 3, but without tier 3's yield. The UMAP
diagnostic fails on its own separate merits: on this corpus the between-service geometry a human
would read off the 2-D plot correlates with the true 384-d between-service geometry at Spearman
**0.011–0.239**, and moves between UMAP seeds (0.484–0.608). Two tables that cost nothing —
label co-occurrence and per-service-pair confusion from tier 2's out-of-fold probabilities —
answer the taxonomy question the UMAP was for, conditionally on base rate, and they recover the
fixture's documented ambiguous pairs.

The spec's own §9.7 caveat understated the problem. The reported **56** was not merely
"a TF-IDF proxy that will differ" — the real `multilingual-e5-small` run flags **36** rows at the
specified ≥0.95 cut, and only **34% of its top-56 list overlaps the proxy's top-56**. The number
in the spec was never a measurement of the thing being proposed, and the thing being proposed is
not the thing the proxy measured.

**What I do not claim:** that embeddings are useless here, or that kNN label agreement is a bad
idea in general. On a corpus with real annotator noise it may well earn its keep. The claim is
narrower and it is the one that matters for the decision: **on the only corpus we can measure,
tier 4 costs reviewer time and a fitted-artifact class and returns less per row than widening
tier 3, which is already built.**

Flip conditions are in §7.

---

## 1. The question

Tier 4 is specified as "corroborating only", flagging 56 rows, plus a once-only UMAP taxonomy
diagnostic. It brings a `sentence-transformers` dependency, a pinned multilingual checkpoint, a
new fitted-artifact class in a bit-deterministic pipeline, `umap-learn`, decision item D-11
(D-9 in the proposal), and stage `s45` on the DAG's critical path. Three questions:

1. What does it find that tiers 1–3 do not?
2. Are those rows actually wrong labels?
3. Does it need embeddings at all, or is TF-IDF cosine enough?

---

## 2. Setup (so this is reproducible)

All numbers below are on `data/raw/tickets_export.csv` (5,013 rows, **4,622 labelled**, 8,771
positives, 20 services — matches `data/raw/README.md`). Environment: Python 3.11.15, 4 vCPU / 15 GB,
no GPU; `pandas` 3.0.5, `scikit-learn` 1.9.0, `cleanlab` 2.9.0, `datasketch` 2.0.0,
`torch` 2.13.0+cpu, `sentence-transformers` 5.7.0, `umap-learn` 0.5.12, `numpy` 2.4.6.

- **Text**: NFC, URL/e-mail/phone/card → placeholder, whitespace collapse, `title + ". " + description`.
  This is an *approximation* of Phase 1 (no NER, no boilerplate stripping, no log reduction) — see §8.
- **Dedup clusters**: `datasketch` MinHashLSH, 128 perms, char-5-gram shingles, threshold 0.80,
  two signatures (raw join, Phase-1 text) unioned by union-find → 37 clusters of size > 1
  covering 185 rows, largest **99** (the auto-alert clique, as documented in the fixture README).
- **Tier 2**: word 1–2 gram (`min_df=2`, sublinear) ∪ `char_wb` 3–5 gram (`min_df=3`, sublinear,
  200k cap); OvR `LogisticRegression(class_weight='balanced', max_iter=2000)`; `GroupKFold(5)` on
  the dedup cluster id; out-of-fold probabilities.
- **Tier 3**: `cleanlab.multilabel_classification.filter.find_label_issues` mask ∪
  `rank.get_label_quality_scores(method='self_confidence', aggregator='exponential_moving_average', alpha=0.8) < 0.20`.
- **Tier 4**: `intfloat/multilingual-e5-small` (MIT, 12 layers, d=384, revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`), `"query: "` prefix per the model card,
  `max_seq_length=128`, mean pooling, L2-normalised; k = 10 cosine neighbours **excluding the row's
  own dedup cluster**; `knn_disagreement = 1 − mean_j Jaccard(S_i, S_j)`, exactly as §4.3.4 specifies.
  Also computed: the distance-weighted variant from architecture §4.3 (Spearman 0.9998 against the
  unweighted one — the weighting is a no-op, do not spend a config field on it), a `char_wb` TF-IDF
  cosine variant (the spec's proxy), and the full tier-2 feature-union cosine variant.

Scratch scripts: `k1_prep.py` … `k10_determinism.py` in
`/tmp/claude-0/-home-user-bert-poc/351181d2-9b69-560c-8223-87b870e37c96/scratchpad/`. Throwaway;
nothing was added to the repo outside `docs/specs/`.

---

## 3. What I measured

### 3.1 Tiers 2 and 3 reproduce — same region, so the comparison is fair [measured]

| Quantity | Cleaning spec §4.3.3 | This reproduction |
|---|---|---|
| Tier-2 runtime, 5 folds × 20 services × 4,622 rows | 113 s | **113.8 s** |
| macro-AP / micro-AP (GroupKFold) | 0.9384 / 0.9751 | 0.9279 / 0.9676 |
| micro-F1@0.5 / macro-F1@0.5 | 0.9397 / 0.8807 | 0.9254 / 0.8850 |
| `find_label_issues` mask | 95 | **96** |
| quality score < 0.20 | 126 | 70 |
| quality score < 0.30 / < 0.50 | 313 / 670 | 208 / 779 |
| **Union selector (tier-3 flags)** | **161** | **129** |

The mask lands on 96 vs 95 rows and the runtime is within 1%. The score-threshold counts differ
because my text preprocessing is an approximation of Phase 1 (§8), which shifts the quality-score
distribution; the union selector is 129 instead of 161. Same instrument, same region, ~20% fewer
rows at the same threshold. Every tier-3-vs-tier-4 comparison below uses **my** 129, so the
comparison is internally consistent and, if anything, biased *in tier 4's favour* (a smaller
tier-3 set leaves more room for tier 4 to add something).

### 3.2 Tier 4 with real embeddings is not the tier 4 that was specified [measured]

| Representation | median | p90 | p99 | ≥0.90 | **≥0.95** | =1.00 |
|---|---|---|---|---|---|---|
| Spec §4.3.4 (TF-IDF char proxy, not re-run) | 0.550 | 0.800 | 0.950 | 141 | **56** | 23 |
| My TF-IDF `char_wb` proxy | 0.550 | 0.792 | 0.950 | 143 | **52** | 20 |
| **`multilingual-e5-small` (the proposed instrument)** | 0.467 | 0.750 | 0.923 | 90 | **36** | 14 |
| TF-IDF full tier-2 feature union | 0.576 | 0.800 | 0.950 | 171 | 57 | 25 |

My proxy reproduces the spec's proxy to within a few rows, so the discrepancy is real and not a
setup difference: **the embedding version flags 36 rows at the specified threshold, not 56.**
Every threshold in §4.3.4 is a threshold on a distribution that does not exist.

### 3.3 Representation choice changes two thirds of the list [measured]

This was the requester's most-likely-outcome hypothesis ("TF-IDF gives substantially the same
rows"). **It does not**, and the actual result is worse for tier 4 than that hypothesis would have been.

| Pair | Spearman over all 4,622 rows | top-56 overlap | mean shared neighbours (k=10) |
|---|---|---|---|
| e5 vs TF-IDF `char_wb` | 0.730 | **19/56 (34%)** | 4.44 / 10 |
| e5 vs TF-IDF union | 0.719 | 23/56 (41%) | 4.30 / 10 |
| TF-IDF `char_wb` vs TF-IDF union | 0.934 | 41/56 (73%) | 8.16 / 10 |

Two TF-IDF variants agree with each other far more than either agrees with the encoder. If the
flag list were tracking real label problems, an arbitrary representation swap would not replace
two thirds of it. There is no external anchor that says which list is right — which is the core
problem, and §3.7 shows neither is.

### 3.4 The central overlap table [measured]

Tier-1 conflicts in my reproduction: 1 `partial` group (the 99-row alert clique, capped at 5) and
2 `nested` groups → **5 rows queued** (spec: ~14; the difference is my dedup grouping, and it does
not matter here because tier-1 and tier-4 flags are disjoint under every variant).

| Tier-4 variant | n flagged | also in tier 3 | also in tier-3 mask | also tier 1 | **tier-4-only** | tier-3-only |
|---|---|---|---|---|---|---|
| e5 ≥ 0.95 | 36 | 9 | 5 | 0 | **27** | 120 |
| e5 top-56 | 56 | 15 | 8 | 0 | **41** | 114 |
| TF-IDF `char_wb` top-56 | 56 | 19 | 9 | 0 | **37** | 110 |
| e5 distance-weighted ≥ 0.95 | 31 | 7 | 3 | 0 | 24 | 122 |

So tier 4 is **not** a near-subset of tier 3 — 41 of 56 are its own. Under the spec's own logic
that is the point at which it earns a look. §3.5–3.8 are that look, and they are all negative.
One number to hold onto: `spearman(knn_disagreement, −cleanlab_quality_score) = 0.542` — the
two rankings are the same ranking with noise on it.

### 3.5 The flag has no null model, and at the threshold it is mostly the null [measured]

The spec concedes "the measure has no null model". I built one: for each row, the expected
disagreement if its 10 neighbours were drawn uniformly at random from the corpus (300 draws/row).

- Corpus-wide **null** disagreement: median **0.907**, p10 0.876, min 0.834. **9.4% of rows have a
  null ≥ 0.95** — i.e. would clear the spec's flagging threshold with *random* neighbours.
- e5 top-56: mean measured disagreement 0.957 against mean own-null 0.945 → **mean excess +0.012**.
  **17 of the 56 flagged rows are below their own null** (their real neighbourhood agrees with them
  *more* than chance, and they are flagged anyway).
- Re-ranking by excess-over-null instead of raw disagreement changes 15 of 56 rows and does not fix
  the character of the list (§3.6): 15/56 are still the alert template.

**Structural biases the metric has, none of which are label-error signals:**

| Cut | corpus share | share of e5 top-56 |
|---|---|---|
| `\|S\|=1` rows | 34% | **54%** |
| `\|S\|=3` rows | 22% | **2%** |
| rows carrying a service with < 50 positives | 2.5% | **35.7%** (14×) |
| `[ALERT]` auto-template rows | 2.1% | **29%** (16/56; 13 of the 41 tier-4-only) |

Mean Jaccard punishes small label sets by construction (a singleton row scores ≤ 0.5 against a
2-label neighbour even when fully contained), and the rare-service concentration is exactly the
tail-erosion hazard that forces guard rule R2 on tier 3 — here it is 14× worse. The alert-template
concentration is a *designed-in* artifact: excluding the row's own dedup cluster (a rule that is
right for its own reason) strands the 99-row template clique with meaningless neighbours. The
TF-IDF variant does not have this failure (0/56) and the encoder variant does — which is another
way of saying the two instruments are not measuring the same thing.

### 3.6 Are the tier-4-only rows wrong labels? Mostly no [measured + judgement]

I read 32 tier-4-only rows by hand (20 sampled from the 41 e5-only rows, seed 0; 12 sampled from
the excess-over-null variant, seed 1). Dumps:
`scratchpad/t4_only_sample.txt`, `scratchpad/t4_excess_sample.txt`.

Of the 20-row e5-only sample:

| Judgement | n | Examples |
|---|---|---|
| Label correct, no action — row is just isolated | **15** | `TICKET-13300` (DNS resolution → `dns`), `TICKET-12278` ("how do we verify backup integrity" → `backups`), `TICKET-14560` (CDN request logs → `cdn,logging`), and **6 `[ALERT]` template rows** labelled `compute,monitoring`, which the fixture README documents as correct by construction |
| Debatable missing-label candidate | 4 | `TICKET-14061` (`api-gateway`; `object-storage` arguable), `TICKET-14335` (`compute`; `api-gateway` arguable), `TICKET-14838`, `TICKET-15494` |
| Plausibly wrong label | **1** | `TICKET-16709` "доступ в проекте data-lake" labelled `console-ui`, tier-2 `P(access-control)=0.72` |

The excess-over-null sample reads the same way: 15/56 alert-template rows, and the single
plausible candidate is `TICKET-16709` again. **Estimated precision of tier-4-only flags for
"a reviewer would change something": 5% strict, 25% counting every debatable missing-label case
[estimate from a 20-row sample; ±11pp binomial at 95% for the strict figure].** And the one row
that is plausibly wrong is already visible in tier-2 probabilities alone — it is the shape that
tier-5 strata S2/S3 select by construction, with no embedding involved.

This is the same finding that reframed the cascade, one tier over: the rows are hard, or short, or
rare, or template — not wrong.

### 3.7 The decisive measurement: detection of *injected* label errors at matched budget [measured, 3 seeds]

Hand inspection has no ground truth, so I made one. I corrupted 3% of labelled rows (138 rows) under
two noise processes, refit tier 2 on the corrupted labels (`GroupKFold`, same grouping), re-ran
cleanlab and re-ran the kNN agreement against the corrupted label sets — then asked which tier's
ranking recovers the injected rows.

- **`random`** — replace one carried label with a uniformly drawn other service. Roughly the
  class-conditional flip process confident learning assumes ([Northcutt, Jiang & Chuang, JAIR 2021](https://arxiv.org/abs/1911.00068)).
- **`ambiguous`** — swap inside the fixture's documented ambiguous pairs (`auth`↔`access-control`,
  `monitoring`↔`logging`, `notifications`↔`integrations`, `cdn`↔`networking`, plus `console-ui`
  drops). **This is the noise process cleaning spec §4.3.0(4) argues cleanlab is structurally bad at**,
  and therefore tier 4's best case.

Recall of injected errors, mean ± sd over 3 seeds:

| Budget (rows a reviewer sees) | ambiguous: tier 3 | ambiguous: tier-4 e5 | ambiguous: tier-4 TF-IDF | random: tier 3 | random: tier-4 e5 |
|---|---|---|---|---|---|
| 25 | 0.135 ± 0.011 | 0.082 ± 0.004 | 0.065 ± 0.015 | 0.174 ± 0.007 | 0.130 ± 0.008 |
| 56 | **0.309 ± 0.023** | 0.143 ± 0.004 | 0.138 ± 0.008 | **0.377 ± 0.019** | 0.276 ± 0.019 |
| 129 | **0.543 ± 0.019** | 0.220 ± 0.011 | 0.215 ± 0.004 | **0.720 ± 0.029** | 0.374 ± 0.034 |
| 185 | **0.683 ± 0.022** | 0.239 ± 0.007 | 0.241 ± 0.004 | **0.862 ± 0.022** | 0.408 ± 0.029 |
| 300 | 0.826 ± 0.007 | 0.331 ± 0.011 | 0.331 ± 0.018 | 0.952 ± 0.028 | 0.510 ± 0.017 |
| 500 | 0.930 ± 0.018 | 0.486 ± 0.022 | 0.461 ± 0.044 | 0.986 ± 0.025 | 0.664 ± 0.011 |

And the actual cascade question — **185 review slots, spend them how?**

| Allocation | ambiguous | random |
|---|---|---|
| tier 3 top-185 **alone** | **0.683 ± 0.022** | **0.862 ± 0.022** |
| tier 3 top-129 + tier-4 e5 top-56 | 0.589 ± 0.015 | 0.807 ± 0.029 |
| tier 3 top-129 + tier-4 TF-IDF top-56 | 0.587 ± 0.019 | 0.790 ± 0.022 |

**Adding tier 4 to a fixed budget costs 9.4pp (ambiguous) and 5.5pp (random) of recall**, with the
sign consistent across all 3 seeds and both representations. The same holds at ~300 rows
(tier 3 alone 0.826 vs tier-3∪tier-4 at ≤258 rows 0.606). Tier 4 is not worthless in isolation —
0.276 at 56 rows versus a 1.2% chance rate is real signal — it is **strictly dominated** signal,
and it is dominated on its supposedly-complementary noise process too (0.143 vs 0.309 at the same
56 rows). Cleanlab's theoretical weakness on ambiguity-driven noise is visible (0.683 vs 0.862 at
185), and tier 4 does not fill that gap: it degrades on the same noise, harder.

Caveat, stated plainly: injected noise is not real annotator noise, and 3% is a guess. But the
`ambiguous` process was designed to be tier 4's best case and tier 3's worst, and tier 4 still
loses by more than 2×.

### 3.8 Ablation: dropping tier-4-only flags ≈ dropping random rows [measured, 3 seeds]

The §4.3.3 ΔB protocol, extended: flags removed from **training** folds only, evaluation folds
untouched, folds re-drawn per seed by seeded cluster→fold assignment, same folds across conditions
within a seed. Same approximation caveat as the spec's ΔB (flags come from the global OOF matrix,
not nested CV).

| Condition | macro-AP (mean ± sd, 3 seeds) | Δ vs baseline | Δ per dropped row |
|---|---|---|---|
| baseline (4,622 rows) | 0.9292 ± 0.0026 | — | — |
| drop tier-3 flags (129) | 0.9182 ± 0.0012 | **−1.10pp** | −0.0085pp |
| drop tier-4 e5-only (41) | 0.9250 ± 0.0031 | −0.42pp | −0.0102pp |
| drop tier-4 TF-IDF-only (37) | 0.9264 ± 0.0030 | −0.28pp | −0.0076pp |
| **drop 41 random rows (control)** | 0.9264 ± 0.0026 | **−0.28pp** | −0.0068pp |

Tier-4-only flags are, per row, **not distinguishable from random rows** — if anything slightly
more costly to lose (worse than the control in all 3 seeds individually, by 0.19/0.16/0.09pp).
This reproduces the cascade's most useful negative result (dropping tier-3 flags hurts: my −1.10pp
vs the spec's −0.79pp) and extends it: tier 4's unique rows are ordinary training data.

### 3.9 The UMAP diagnostic, judged separately — also cut [measured]

The diagnostic's purpose is "do `auth` and `access-control` occupy the same region", read off a
2-D projection by a human. Three measurements:

1. **The plot's geometry is not the embedding's geometry.** Spearman between the 190 inter-service
   centroid distances in UMAP-2D (`n_neighbors=15, min_dist=0.1, metric=cosine`) and the same
   distances in the 384-d space: **0.011, 0.208, 0.239** for seeds 0/1/2. A human reading "these two
   services are close, those two are far" off the plot is reading something that barely tracks the
   space being projected.
2. **It moves between seeds.** Spearman of those 190 distances across seed pairs: **0.484, 0.568,
   0.608**. Architecture §4.5 already assigns UMAP a `purpose`-derived seed "because a diagnostic
   that moves between runs is worse than none" — that is a fix for reproducibility of *one* run,
   not for the fact that the reading itself is seed-dependent. (UMAP's own FAQ states that making
   inter-cluster distance meaningful is a *goal*; on this corpus, with these parameters, it is not
   achieved — see [UMAP FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html).)
3. **20 overlapping multi-label classes on one 2-D scatter.** Mean cardinality is 1.90, so most
   points carry ≥ 2 colours; the head classes each cover 13–18% of rows. There is no honest
   colouring of this plot.

**The free alternative answers the same question better.** From tier-2 out-of-fold probabilities
alone (already computed, $0, no new dependency), per-service-pair confusion — the rate at which the
model asserts `b` on rows carrying `a` and not `b`:

| pair | P(b\|a-only) > 0.5 | P(a\|b-only) > 0.5 | co-occurrence |
|---|---|---|---|
| `monitoring` ↔ `notifications` | 0.223 | 0.019 | 119 |
| `api-gateway` ↔ `cdn` | 0.000 | 0.184 | 0 |
| `cdn` ↔ `console-ui` | 0.143 | 0.000 | 3 |
| `compute` ↔ `networking` | 0.020 | 0.095 | 178 |
| `access-control` ↔ `console-ui` | 0.054 | 0.035 | 134 |
| `billing` ↔ `subscriptions` | 0.014 | 0.056 | 273 |
| `access-control` ↔ `auth` | 0.026 | 0.040 | 213 |

This recovers the fixture's deliberately-planted ambiguities, is **directional** (the
"in given label, not in suggested / suggested, not given" asymmetry cleanlab's
`common_multilabel_issues` already gives), is conditional on base rate, and is a table you can
diff between snapshots. The embedding-neighbour version of the same ranking (Spearman 0.575
against it) is dominated by class size — its top pairs are `billing`↔`subscriptions` and
`compute`↔`networking` simply because those classes are large.

If a neighbourhood view is still wanted, **neighbour purity does not need UMAP** — it is a table
over the 384-d vectors (`cdn` 0.139, `terraform-provider` 0.160, `dns` 0.166, … `billing` 0.690).
But note what it actually measures: purity tracks class frequency almost perfectly, so it restates
"rare classes are rare". Tier 2's per-service AP says the same thing with a decision attached.

---

## 4. The cost side, priced honestly [measured unless marked]

| Cost | Measured value | Note |
|---|---|---|
| `torch` CPU install | **754 MB** on disk (`site-packages/torch`) | plus `transformers`, `sentence-transformers`, `tokenizers`, `safetensors`, `huggingface_hub` |
| `umap-learn` | pulls `numba` 0.67 + `llvmlite` 0.49 + `pynndescent` 0.6 | a JIT toolchain in a data pipeline |
| Checkpoint on disk | **471 MB** (fp32 safetensors, HF cache) | architecture §4.8's "118 MB INT8" is the *quantised* artifact; the fp32 pull is what `sentence-transformers` fetches by default, and the 1.5 GB disk allowance in the proposal is the right number |
| Embedding wall-clock, 4 threads | 4,622 rows in **59.5 s** (77.7 rows/s) → **~10.6 min at 50k** [estimate by scaling] | fp32 torch; the spec's ~6 min assumes ONNX INT8 at 143 rows/s, which is a further build step not yet specified |
| `OMP_NUM_THREADS=1` tax | **3.03×** (78.8 → 26.0 rows/s, 1,000 rows, uncontended) → **~32 min at 50k** | the spec's ~3× estimate is correct |
| What the tax buys | **Nothing measurable on one host**: 1-thread vs 4-thread output was **bit-identical** over 1,000 rows (max abs diff 0.0), and all 1,000 k=10 neighbour sets were unchanged | so the documented determinism hazard is a *cross-host / cross-BLAS* risk, untested here, not a thread-count risk. Either way it is a hazard that disappears entirely if the stage does |
| Exact kNN | ~2 s at 5k; O(n²) — 960 GFLOP at 50k [estimate, arch §6.2] | fine at this scale |
| Artifacts | 7.1 MB embedding matrix at 4,622 rows (77 MB at 50k), plus `embeddings_sha256`, `revision` pinning, mirror, `models pull` command, egress-blocked CI assertion | a whole artifact class |
| Governance | D-11/D-9 (checkpoint + residency), D-12 (determinism vs wall-clock), `s45` on the critical path, `--skip-tier 4` escape hatch, licence review of 4 more packages | |
| **Reviewer time — the cost that actually hurts** | 56 rows × ~2 min ≈ **1.9 person-hours per snapshot** on a list whose strict precision I estimate at ~5%, of which ~29% is one alert template | and, per §3.7, those 56 slots are worth **more** spent on tier 3 |

Against that: the measured benefit is **negative at matched budget** and **zero-to-negative** in
the ablation. There is no reading of these numbers where the price is worth paying.

---

## 5. Verdict and what to remove

**Verdict: cut.** Not "downgrade to TF-IDF cosine" — the TF-IDF variant is also dominated by tier 3
at every budget (§3.7) and also fails the ablation control (§3.8). Downgrading would keep a
useless flag list and merely make it cheap. Not "diagnostic-only" either: §3.9 shows the
diagnostic is the weaker of two available instruments and the free one is already computed.

Concretely, to act on this:

**`docs/specs/dataset-pipeline-cleaning.md`**
- Delete §4.3.4 in its entirety. Replace with a two-paragraph §4.3.4 "Why there is no embedding
  tier", citing this document's §3.7 and §3.8 numbers, so the option is closed with evidence
  rather than silently dropped.
- §4.3.0 cascade table: remove the tier-4 row; renumber the LLM judge to tier 4 and human review to
  tier 5, or keep the numbering and mark tier 4 "cut — see `knn-tier-validation.md`". Prefer the
  latter; renumbering invalidates cross-references in three documents.
- §4.3.5: tier-5 strata are unaffected (S1–S4 never referenced kNN). No change.
- §4.3.10 routing and §4.3.11 queue sizing: remove `t4_neighbour_disagree` as a corroborating
  sort key. Tier-3 rank alone is the sort key. **The freed 56 slots should go to widening the
  tier-3 selector** — measured payoff, §3.7.
- Add to §6.1 the two free taxonomy reports (per-service-pair tier-2 confusion, both directions;
  label co-occurrence) as required per-snapshot outputs. They replace the UMAP deliverable and cost
  nothing.
- §9.7: the tier-4 caveat becomes moot; replace with a pointer here.

**`docs/specs/dataset-pipeline-architecture.md`**
- Delete stage **`s45_knn_agreement`** and the `audit_t4` table (§4.3), the `t4_*` columns, and
  `umap_taxonomy.png` / `umap_coords.parquet` from §4.8's artifact list.
- Delete the **embedding checkpoint** and **embedding matrix** rows from §4.8's pinning table,
  the `models/` cache directory, `ticketds models pull` (§6.5 step 5b) and the egress-blocked-CI
  assertion for `s45`.
- §4.5: drop the `audit.umap` seed purpose and the ONNX `intra_op_num_threads` clause. The
  `OMP_NUM_THREADS=1` rule still applies to tiers 2–3 (`liblinear`/`saga` genuinely do drift);
  keep the `t2_probs` 6-significant-digit rounding rule, which is the mitigation that actually
  earns its place.
- §6.2 cost table: remove the three `s45` rows (~35 s / ~6 min embedding, kNN, UMAP). The cheap-tier
  critical path drops to tier 1–3 only.
- **D-11 (checkpoint + residency) and D-12 (determinism vs wall-clock) are closed by this
  document** — D-11 disappears; D-12 shrinks to the sklearn-only case.
- §4.6 `s46_triage`: remove `t4_neighbour_disagree` from the `flag_reasons` enum, `flag_tiers` 4.
- **Do not silently lose one thing.** `t4_dist_to_centroid` was doubling as the OOD signal for
  [`llm-fallback-policy.md` §2 case 4](./llm-fallback-policy.md). That requirement is real and it
  does **not** belong here: it is a *serving-time* signal, its threshold is set on the production
  model's validation set, and the production classifier already has an encoder. Move it to the
  classifier spec / fallback policy as an explicit requirement; do not carry an embedding stage in
  the dataset pipeline to provide it.
- Dependency list: drop `sentence-transformers`, `torch`, `umap-learn`, `transformers` from the
  pipeline environment. This is the single largest reduction in the pipeline's dependency surface.

**`docs/proposals/finetuning-dataset-pipeline.md`**
- §3.1 C1b: cascade becomes conflicts → TF-IDF+LR probs → cleanlab → LLM judge → human. Drop
  "embedding-kNN agreement + a once-only UMAP diagnostic" and the "fitted-model artifact class"
  consequence shrinks to a vectoriser + an LR (no downloaded checkpoint).
- §4.3 D-9 (which embedding checkpoint): **closed — no checkpoint.**
- §3.2/§4.1 cost lines: cheap-tier CPU ~2.5 min → **~2 min at 5k**, ~12 min → **~2 min at 50k**
  (tier 2 dominates and is linear-ish); hardware note drops the +1.5 GB checkpoint allowance.
- §5 milestone 5e: "conflict detection, TF-IDF baseline, confident learning, kNN + UMAP" → drop
  the last two; the milestone's deliverable ("a full label-quality report with no LLM in it")
  is unaffected and arrives sooner.

**If, against this recommendation, tier 4 is kept**, it must be held to a gate, not to a routing
rule: it may enter the pipeline only if, on a labelled error set (injected or human-adjudicated),
**tier-3 top-N ∪ tier-4 top-M beats tier-3 top-(N+M) on injected-error recall, mean over ≥ 3 seeds,
with non-overlapping spreads.** It fails that gate today by 5.5–9.4pp. "Corroborating only" is not
a weaker gate; it is the same gate with the cost hidden in reviewer sort order.

---

## 6. Risks of cutting it (and how they would show up)

| Risk | Detection |
|---|---|
| Real corpora have semantic-but-not-lexical duplicate clusters that MinHash misses (cleaning §4.1.7 names the Postgres pool cluster) and embeddings would find | This is a **dedup** question, not a label-audit question. If it bites, it shows up as label conflicts inside groups that tier 1 never formed. Re-open embeddings as a *dedup* instrument then, with its own gate |
| The taxonomy question genuinely needs a geometric view | The tier-2 pair-confusion table has an asymmetry column and a per-service AP; if a product owner cannot act on it, that is the trigger to revisit — but revisit with a *quantitative* neighbour-purity table, not a scatter plot |
| Real annotator noise is not like my injected noise, and tier 4 does better on it | The random-audit stratum S4 (§4.3.5) is the instrument that would reveal it: if human review of S4 finds errors that tier 3's ranking placed far down, re-run this comparison on those rows. That is the flip condition in §7 |
| OOD detection quietly disappears with `s45` | Explicit hand-off to the classifier spec, above. Track it as an open item, not a deletion |

---

## 7. What would flip this verdict

1. **A real labelled error set on real data.** Once tier 6 review returns adjudicated verdicts for a
   few hundred rows, recompute §3.7's recall curve with *real* errors instead of injected ones. If
   tier 4's ranking beats tier 3's at matched budget on the ambiguity-driven subset, it comes back.
2. **A corpus whose labels come from CatBoost or from multiple annotators**, where noise is neither
   author-assigned nor internally consistent. This fixture is the weakest possible evidence *for*
   any label-error tier (README: "labels are author-assigned, not blind-annotated") — and tier 3
   still cleared it while tier 4 did not.
3. **A different formulation of the flag** that (a) is normalised against the per-row null in §3.5,
   (b) is not mean-Jaccard (which is cardinality-biased — try per-service neighbour vote rate,
   `P(service s in neighbourhood | s not in label set)`), and (c) excludes template cliques from
   both directions. That is a research task, and it has to pass the §5 gate before it costs a
   dependency. I measured the cheapest version of (a) — excess over null — and it did not help.
4. **The encoder arriving for another reason.** If the pipeline ends up hosting an encoder anyway
   (e.g. semantic dedup, or the frozen-embedding baseline moves in-pipeline), the marginal cost of
   tier 4 collapses to the kNN pass. It would still have to pass the §5 gate — negative value at
   matched budget is not fixed by being cheap.

---

## 8. Measurement caveats (in the style of cleaning spec §9.7)

- **My Phase-1 text is an approximation.** No NER, no boilerplate stripping, no log/traceback
  reduction, no `ё`-folding. This shifts tier-2 macro-AP by ~1pp against the spec's number and
  changes the tier-3 score-threshold counts (129 vs 161 in the union selector). It affects tier 3
  and tier 4 *symmetrically* — both consume the same text — so the comparisons hold, but the
  absolute counts are not the spec's counts. Notably, boilerplate stripping would change the
  alert-template rows' embeddings substantially; that could reduce the 29% template artefact, and
  the implementation should not assume my exact figure carries.
- **The ablation shares the spec's ΔB flaw.** Flags were computed from the global OOF matrix, not
  from an inner CV inside each outer training fold. The control condition (drop 41 random rows) is
  subject to the same protocol, so the tier-4-vs-random comparison — the one the verdict rests on —
  is unaffected.
- **Injected noise is not real noise.** 3% rate, two hand-designed processes. The `ambiguous`
  process was built to favour tier 4 and still did. Real noise may be lumpier (one annotator, one
  service, one month) and neither tier was tested against that.
- **Hand judgements are mine, not an annotation guideline's.** 32 rows, one non-expert pass, on
  synthetic text with author-assigned labels. The precision figures in §3.6 are estimates with wide
  intervals. The structural findings (§3.5) do not depend on them.
- **Single machine, single BLAS.** The bit-identical result across thread counts says nothing about
  cross-CPU reproducibility, which is the risk architecture §4.5 actually names. I did not test
  ONNX Runtime INT8, which is what the spec's throughput figure assumes.
- **UMAP was run at one parameter setting** (`n_neighbors=15, min_dist=0.1, metric=cosine`,
  3 seeds). Other settings give other layouts — which is itself part of the point.
- **`k` = 10 throughout**, per cleaning §4.3.4; architecture §6.4 mentions `k=15`. I did not sweep
  `k`. A sweep would not rescue an instrument that loses by 2× on its own best-case noise process.
