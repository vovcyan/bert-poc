# Implementation Plan: CSV → dataset pipeline (`ticketds`)

Source spec: [`docs/specs/csv-dataset-pipeline.md`](../docs/specs/csv-dataset-pipeline.md)
Companions: [`ticket-services-classifier.md`](../docs/specs/ticket-services-classifier.md),
[`dataset-construction-runbook.md`](../docs/specs/dataset-construction-runbook.md),
[`llm-fallback-policy.md`](../docs/specs/llm-fallback-policy.md)

**Task list target:** this document is authoritative for task detail (acceptance criteria,
verification, dependencies). [`tasks/todo.md`](./todo.md) carries the same tasks as a bare
checklist, in order, for `/build` and for tracking progress. No external tracker is designated in
this repo (no `CLAUDE.md` / `AGENTS.md`), so the default markdown targets apply.

**Standing bar:** every task must also clear
[`.claude/references/definition-of-done.md`](../.claude/references/definition-of-done.md).
Acceptance criteria below answer "did we build the right thing?"; the Definition of Done answers
"is it finished to our standard?". Both, every time.

---

## Overview

Build `ticketds`: a deterministic, re-runnable Python pipeline that turns
`data/raw/tickets_export.csv` (5,013 rows) into a versioned training dataset
(`train/val/test.parquet`), a quarantine table, two human review queues, a blind annotation
sampling frame, a `manifest.json` that pins exactly what was produced, and a generated
`datasheet.md` that states what the data cannot support.

Nine stages (S0 ingest → S8 freeze) plus a shared `ticketds.redact` library that the model service
will import at inference time. The binding constraints are not modelling constraints: byte-identical
reproducibility (§1.7, §12.2), row accounting with no silent drops (§3 rule 1), and redaction recall
gates on `SECRET`/`CARD`/`BANK_ID`/`GOV_ID` (§4.6).

35 tasks in 7 phases. Ordering is dependency-driven and each task leaves the pipeline runnable.

---

## Architecture Decisions

Carried from the spec (already argued there — restated here only where it shapes task ordering),
plus the decisions this plan adds.

- **Walking skeleton first (Phase 0), then one stage per slice.** A pipeline's vertical slice is a
  stage: module + schema at both boundaries + quarantine reasons + metrics + a `report` subcommand
  + tests. Building all schemas, then all stages, then all reports would leave nothing runnable
  until the end. After T5 the command `python -m ticketds.cli run` works and stays working.
- **The redactor is a library, not a stage** (§4.5c). It is built and measured here but owned
  jointly with the model service, so it gets its own phase (T8–T15) and its own quality gates,
  ahead of the stage that calls it.
- **Redaction rules and the placeholder vocabulary are data, not code** (§4.5, Appendix A). Every
  rule declares `name`, `pattern`, `entity_type`, `confidence`, and ≥1 `example` that the test suite
  asserts it matches — which is what makes the zero-match build gate (§4.3) mechanical.
- **LLM stages come after everything deterministic** (Phase 5). They are the only stages that touch
  the network, they are gated on a disk cache for reproducibility (§5.4), and the pipeline must be
  fully valuable with S5/S6 skipped — the default config ships `allow_raw_text: false`, which skips
  S6 entirely (§5.3).
- **Every `[measured]` claim in the spec becomes a `report` subcommand, not a scratch artifact**
  (Appendix D). Each stage task carries its report subcommand as an acceptance criterion, so the
  spec's numbers stay checkable as the data changes.
- **Human-blocked work is a task with an owner, not a footnote.** T14 (span gold annotation) blocks
  the redaction recall gate, which blocks four of the ten success criteria. It is scheduled
  explicitly so the ~10 person-hours are booked before Phase 2 ends, not discovered at Phase 6.

### Decision: no Jupyter notebooks in this project

Asked and validated. **Verdict: no notebooks in the repository or in the pipeline path. Local,
throwaway exploration in a scratchpad is fine and is already the house convention.** This is not a
style preference — three of the reasons are hard-rule violations in the spec.

**Why not, grounded in the spec:**

1. **They break the one property the pipeline exists to have.** §1.7 and success criteria 2 and 10
   require two builds to produce identical hashes and a teammate to reproduce them from the README.
   A notebook's state depends on execution order, which is not captured in the file. Anything a
   notebook computes is unverifiable by `ticketds.cli verify`.
2. **They commit unredacted ticket text to git.** §11 *Never*: "Commit raw or partially-redacted
   ticket text outside `data/raw/`." A `.ipynb` stores cell outputs inline in its JSON, so a single
   `df.head()` on an S0/S1 frame writes PII-bearing rows into the repository — and into git history,
   where deleting the notebook later does not remove them. The corpus carries names, ИНН, cards and
   API keys (§4.3); this is exactly the artifact class the spec forbids.
3. **The quality gates cannot reach them.** §9 requires `mypy --strict` on `src/ticketds` and `ruff`
   everywhere; §10 requires unit, property, golden and integration tests. Notebook code is neither
   type-checked nor tested by default. Redaction is safety-critical (§4), and nothing
   safety-critical may live where the gates do not run.
4. **They are a train/serve skew mechanism.** §4.5c and classifier spec §2.9 require the model
   service to import the *same* `ticketds.redact` module and the *same* `model_input` builder.
   Re-implementing preprocessing in a cell is precisely how a model gets trained on `<EMAIL>` and
   served raw addresses.
5. **The spec already ships the artifact notebooks are usually wanted for.** Appendix D turns every
   `[measured]` claim into `ticketds.cli report --stage <name>`: versioned, tested, deterministic,
   recorded in the manifest. A notebook produces a screenshot; a `report` subcommand produces a
   claim a reviewer can re-run. `datasheet.md` (§S8) covers the narrative artifact.
6. **`.ipynb` diffs are not reviewable**, and §8 makes the reviewable surface explicit: the diff on
   `manifest.json` in a pull request is the review. Adding `jupyter`/`ipykernel` also trips §11
   *Ask first: adding a dependency* for zero pipeline value.

**The honest counterargument, and why it does not land here.** Notebooks are genuinely good at
tight EDA loops with plots — and redaction-rule iteration *is* a tight loop. But the spec already
provides that loop without the hidden state: `ticketds.cli stage redact --config …` runs one stage
against a cached predecessor (§3, §7), and `report --stage redact` prints the entity table. Fast
feedback was designed in; it does not need a second, unversioned mechanism.

**What is allowed.** Throwaway exploration in the session scratchpad, at the same status classifier
spec §9.6 gives its benchmark script: *not part of any deliverable, not imported by anything, not
committed*. `.py` or `.ipynb` both fine there. The rule is on the graduation path, not the
scratching: **if a finding matters, it lands as a `report` subcommand plus a test, not as a
notebook someone is asked to re-run.**

**Enforcement (folded into T1 and T34):** `*.ipynb` in `.gitignore`; a CI check that fails on any
committed notebook; no `notebooks/` directory in the project structure (§8 does not have one).
If the team later overrides this, the minimum bar is `nbstripout` in a pre-commit hook (which also
removes the PII-in-outputs failure), never imported by `src/ticketds`, and never the source of a
`[measured]` claim — but the recommendation stands: don't.

---

## Task List

### Phase 0: Foundation — a pipeline that runs

- [ ] T1: Project scaffold, dependency pins, and quality gates
- [ ] T2: Config layer — Pydantic models, `dataset.v1.yaml`, `taxonomy.yaml`
- [ ] T3: Stage contract — `StageResult`, quarantine, Pandera schemas, row accounting
- [ ] T4: CLI skeleton — `run` / `stage` / `report` / `verify` / `eval-redaction`

### Checkpoint A: Foundation

### Phase 1: Ingest and labels

- [ ] T5: S0 `ingest` — RFC 4180 read, strict decode, typed frame, quarantine
- [ ] T6: S0 mojibake detection and CP1251 repair
- [ ] T7: S1 `normalize-labels` — taxonomy validation and dirty flags

### Checkpoint B: Deterministic head of the pipeline

### Phase 2: The redaction library

- [ ] T8: Placeholder vocabulary and the span-application engine
- [ ] T9: Layer 1 — structural rules in `redaction.yaml`
- [ ] T10: Layers 2–3 — context rules with checksum-as-confidence
- [ ] T11: Layer 4 — NER for `PERSON` / `ORG` / `ADDR`
- [ ] T12: S2 `redact` stage and the span sidecar
- [ ] T13: Gold-set sampler and `eval-redaction`
- [ ] T14: **[human, Eng]** Annotate the 200-row span gold set
- [ ] T15: Redaction regression harness

### Checkpoint C: Redaction is measured, not asserted

### Phase 3: Dedup and enrichment

- [ ] T16: S3 `dedup` — exact and MinHash near-duplicate clustering
- [ ] T17: S4 `enrich` — `model_input`, lengths, payload, bursts
- [ ] T18: S4 language detection — payload-stripped vs naive

### Checkpoint D: Splitter inputs complete

### Phase 4: Splits

- [ ] T19: S7 `split` — temporal, gapped, cluster-aware
- [ ] T20: Random diagnostic split and leakage comparison
- [ ] T21: Per-service counts, `measurable` flag, split integrity gates

### Checkpoint E: A dataset that can be trained on

### Phase 5: LLM stages

- [ ] T22: OpenRouter client — structured output, retry, call ceiling
- [ ] T23: Content-addressed LLM cache and `--dry-run` cost estimate
- [ ] T24: S5 prompt v1, prompt hashing, evidence validation
- [ ] T25: S5 Pass A — blind measurement sample with bootstrap CI
- [ ] T26: S5 Pass B — triage ranking and `review/labels.jsonl`
- [ ] T27: S6 egress gating and stratified sampling
- [ ] T28: S6 audit — span verification and `review/redaction.jsonl`

### Checkpoint F: LLM stages, cached and gated

### Phase 6: Freeze, manifest, datasheet

- [ ] T29: S8 `freeze` — output schema and split artifacts
- [ ] T30: `manifest.json` assembly and the `verify` command
- [ ] T31: `annotation_queue.jsonl` — blind, iteratively stratified
- [ ] T32: `datasheet.md` generation
- [ ] T33: Integration test — full pipeline on `slice_200.csv`, reproducible

### Checkpoint G: Complete

### Phase 7: Ship readiness

- [ ] T34: CI workflow running every build-failing gate
- [ ] T35: README quickstart and the committed manifest

---

## Phase 0: Foundation

### Task 1: Project scaffold, dependency pins, and quality gates

**Description:** Create the `src/ticketds` package layout from spec §8, pin every dependency at the
versions §2 names as floors, and wire the four quality commands from §7 so that every later task has
a gate to run. No pipeline logic. This task also encodes the no-notebooks decision.

**Acceptance criteria:**
- [ ] `pip install -r requirements.txt -r requirements-dev.txt` succeeds on Python 3.11 with every
      version pinned exactly (Polars ≥1.0, Pandera, Pydantic v2, `datasketch`,
      `iterative-stratification==0.1.9`, `presidio-analyzer`, `py3langid`, Typer; dev: pytest,
      hypothesis, ruff, mypy)
- [ ] `ruff check .`, `ruff format --check .`, `mypy src/ticketds` and `pytest -q` all pass on the
      empty package; ruff line length 100, mypy strict on `src/ticketds`
- [ ] `.gitignore` covers `data/stage/`, `data/cache/`, `data/processed/` (except
      `manifest.json`), `.venv/`, and `*.ipynb`
- [ ] `pytest -m integration` is registered as a marker and deselected from the default run

**Verification:**
- [ ] `ruff check . && ruff format --check . && mypy src/ticketds && pytest -q`
- [ ] Manual: `git status` is clean after a no-op pipeline run creating `data/stage/`

**Dependencies:** None

**Files likely touched:** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`,
`.gitignore`, `src/ticketds/__init__.py`

**Estimated scope:** S

---

### Task 2: Config layer — Pydantic models, `dataset.v1.yaml`, `taxonomy.yaml`

**Description:** Implement Appendix B as typed, validated config. `configs/taxonomy.yaml` holds the
20 services with definitions and 3 positive / 2 negative examples each (data, never a constant in
code — §S1). Load, validate, and expose a resolved config object that serialises deterministically
for the manifest.

**Acceptance criteria:**
- [ ] `Config.load("configs/dataset.v1.yaml")` returns a typed object matching Appendix B exactly,
      including nested `llm.label_review` and `llm.redaction_audit.strata`
- [ ] `redaction.enabled: false` is rejected at load with an explicit error (§4.1 — safety is not
      an experiment knob); `normalisation.enabled: false` is accepted
- [ ] `taxonomy.yaml` parses to exactly 20 kebab-case services, each with a definition and 3+/2−
      examples; a duplicate or non-kebab-case service fails load
- [ ] The resolved config serialises to a stable JSON string whose SHA-256 is identical across two
      loads in different processes

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_config.py`
- [ ] Manual: corrupt one field in `dataset.v1.yaml` and confirm the error names the field

**Dependencies:** T1

**Files likely touched:** `src/ticketds/config.py`, `configs/dataset.v1.yaml`,
`configs/taxonomy.yaml`, `tests/unit/test_config.py`

**Estimated scope:** M

---

### Task 3: Stage contract — `StageResult`, quarantine, Pandera schemas, row accounting

**Description:** Implement the contract every stage obeys (§9): `run(df, cfg) -> StageResult`,
schema validated on entry and exit, rejected rows returned as data with a reason, a `metrics` dict
per stage. Add the row-accounting assertion (§3 rule 3) as shared machinery so no stage can lose a
row silently. Schemas are stubs for stages not yet built; each stage task fills its own in.

**Acceptance criteria:**
- [ ] `StageResult(frame, quarantined, metrics)` and `quarantine(df, reason, stage)` exist;
      quarantined rows carry every source column plus `quarantine_reason` and `quarantine_stage`
- [ ] A shared assertion enforces `rows_in == rows_out + rows_quarantined` and raises naming the
      stage and the shortfall when violated
- [ ] `schema.py` exposes one Pandera (Polars backend) schema per stage boundary; validation
      failure raises with the offending column and row count
- [ ] A deliberately lossy fake stage fails the accounting assertion in a test

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_stage_contract.py`
- [ ] `mypy src/ticketds` passes with strict mode on the new modules

**Dependencies:** T2

**Files likely touched:** `src/ticketds/stages/base.py`, `src/ticketds/schema.py`,
`src/ticketds/pipeline.py`, `tests/unit/test_stage_contract.py`

**Estimated scope:** M

---

### Task 4: CLI skeleton — `run` / `stage` / `report` / `verify` / `eval-redaction`

**Description:** Build the Typer CLI from §7 with a stage registry, sequencing, and per-stage
caching to `data/stage/*.parquet` so any stage can run alone against a cached predecessor. Commands
exist and dispatch; stages are registered as they are built.

**Acceptance criteria:**
- [ ] `python -m ticketds.cli run --config configs/dataset.v1.yaml` executes registered stages in
      order and writes `data/stage/s<N>_<name>.parquet` for each
- [ ] `python -m ticketds.cli stage <name> --config …` runs one stage against its cached
      predecessor and fails with a clear message if that cache is missing
- [ ] `--dry-run` and `--audit-redaction` flags are accepted and recorded in the run context
- [ ] `report`, `verify`, `eval-redaction` are registered and exit non-zero with "not implemented"
      rather than silently succeeding

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_cli.py`
- [ ] Manual: `python -m ticketds.cli run --config configs/dataset.v1.yaml` with zero registered
      stages exits 0 and writes nothing but a run log

**Dependencies:** T3

**Files likely touched:** `src/ticketds/cli.py`, `src/ticketds/pipeline.py`,
`tests/unit/test_cli.py`

**Estimated scope:** M

---

### Checkpoint A: Foundation

- [ ] `ruff check . && ruff format --check . && mypy src/ticketds && pytest -q` all pass
- [ ] `python -m ticketds.cli run --config configs/dataset.v1.yaml` runs end to end (no stages yet)
- [ ] Config hash is stable across processes — the manifest's foundation is sound
- [ ] Human review before Phase 1

---

## Phase 1: Ingest and labels

### Task 5: S0 `ingest` — RFC 4180 read, strict decode, typed frame, quarantine

**Description:** Read the CSV with a real RFC 4180 reader (1,771 descriptions contain embedded
newlines, 805 escaped quotes, 4,307 commas in prose — never `split(',')`). Decode strictly:
`encoding_errors: replace` is forbidden because it destroys bytes before repair can run (§S0). Emit
all 10 source columns as `Utf8` plus `row_index`, quarantining malformed rows with reasons.

**Acceptance criteria:**
- [ ] 5,013 rows in, 5,013 accounted for (kept + quarantined); `id` unique; `updated_at >=
      created_at` for every kept row
- [ ] Quarantine reasons implemented: `column_count_mismatch`, `missing_id`, `duplicate_id`,
      `unparseable_created_at`, `undecodable`
- [ ] An `undecodable` row stores a SHA-256 of its original bytes and **never** the bytes themselves
- [ ] Passing `encoding_errors: replace` in config fails at load, not at read

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s0_ingest.py`
- [ ] Manual: `python -m ticketds.cli stage ingest --config configs/dataset.v1.yaml` reports
      5,013 rows in / 5,013 accounted for

**Dependencies:** T4

**Files likely touched:** `src/ticketds/stages/s0_ingest.py`, `src/ticketds/schema.py`,
`tests/unit/test_s0_ingest.py`, `tests/fixtures/malformed_rows.csv`

**Estimated scope:** M

---

### Task 6: S0 mojibake detection and CP1251 repair

**Description:** Detect the documented corruption — Windows-1251 bytes decoded as Latin-1 before the
file was written, so the file is valid UTF-8 and `U+FFFD` scanning finds nothing. Detect a run of ≥4
Latin-1 supplement characters, repair via `text.encode('latin-1').decode('cp1251')`, and keep both
forms. This is a **redaction bypass**, not a cosmetic defect: the 9 affected rows hide a person's
name, a forwarded email header, a production `.env` paste and a requisites block that no §4 detector
would fire on while mangled.

**Acceptance criteria:**
- [ ] Exactly 9 rows are detected and repaired on the committed CSV; `TICKET-16991` recovers to
      `"Панов Сергей Викторович, действует на основании устава"`
- [ ] Original and repaired text are both retained; a `mojibake_repaired` flag is set
- [ ] No row survives S0 still matching the mojibake signature — asserted as a build-failing gate
- [ ] `report --stage ingest` reproduces the mojibake count and the repair for each row

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s0_mojibake.py`
- [ ] Manual: `python -m ticketds.cli report --stage ingest --config configs/dataset.v1.yaml`
      prints 9 repaired rows

**Dependencies:** T5

**Files likely touched:** `src/ticketds/stages/s0_ingest.py`, `src/ticketds/metrics.py`,
`tests/unit/test_s0_mojibake.py`

**Estimated scope:** M

---

### Task 7: S1 `normalize-labels` — taxonomy validation and dirty flags

**Description:** Turn the free-text `services` column into a validated, sorted, deduplicated set,
and separate "no label" (`null`, `label_status = unlabelled`) from "empty label". An unknown service
is a quarantine, never a silent drop — a service renamed upstream is exactly what this check exists
to catch. Record which normalisations fired: mechanical sloppiness is a free prior on semantic
error, and S5 uses it.

**Acceptance criteria:**
- [ ] Adds `services_norm` (`List[Utf8]`, sorted, deduplicated), `n_services`, `label_status ∈
      {labelled, unlabelled}`, `label_dirty_flags`
- [ ] Every `services_norm` member is in the taxonomy; unknown services quarantine with
      `unknown_service`; labelled + unlabelled + quarantined = 5,013
- [ ] Measured effects reproduced: 39 whitespace fixes, 38 case fixes, 40 intra-row dedups,
      391 rows `unlabelled`, 0 unknown services
- [ ] The 391 unlabelled rows are never emitted into train/val/test (enforced later in T21; flagged
      here)

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s1_normalize_labels.py`
- [ ] Manual: `python -m ticketds.cli report --stage normalize-labels --config …` reproduces the
      cardinality histogram (0:391, 1:1569, 2:1988, 3:1034, 4:31) and mean cardinality 1.90

**Dependencies:** T6

**Files likely touched:** `src/ticketds/stages/s1_normalize_labels.py`, `src/ticketds/metrics.py`,
`src/ticketds/schema.py`, `tests/unit/test_s1_normalize_labels.py`

**Estimated scope:** M

---

### Checkpoint B: Deterministic head of the pipeline

- [ ] `run` executes S0→S1 on all 5,013 rows with full row accounting
- [ ] `report --stage ingest` and `report --stage normalize-labels` reproduce every `[measured]`
      claim in spec §1.2, §1.4 and §S0
- [ ] No row survives S0 matching the mojibake signature
- [ ] Human review before Phase 2 — this is the last cheap moment to revisit open question 1
      (empty-`services` rows) and 6 (keep vs quarantine repaired mojibake)

---

## Phase 2: The redaction library

This phase builds `ticketds.redact` — the module the model service imports at the same version
(§4.5c). It is the highest-risk work in the project: a missed secret in training data is an
incident, and a model can memorise and later emit what it was trained on.

### Task 8: Placeholder vocabulary and the span-application engine

**Description:** Implement the closed placeholder vocabulary (Appendix A: 13 redaction + 4
normalisation) and the engine that turns detected spans into redacted text — ordering
(redaction before normalisation, most-specific first, `<NUM>` always last), overlap resolution, and
configurable collapsing of consecutive identical placeholders. No detectors yet; the engine is
tested against synthetic spans.

**Acceptance criteria:**
- [ ] `redact(text, cfg) -> RedactionResult` with `.text` and `.spans`; placeholders are uppercase,
      angle-bracketed, and drawn only from Appendix A
- [ ] Property: `redact(redact(x)) == redact(x)` — placeholders must not re-trigger rules
- [ ] Property: `apply_spans(text, r.spans) == r.text` — the sidecar is not a separate truth
- [ ] Property: emitted placeholders ⊆ the closed vocabulary, over `hypothesis`-generated text
- [ ] `<NUM>` cannot pre-empt a safety rule: a card-like run is consumed by `<CARD>` first
- [ ] `<IP> <IP> <IP>` collapses to `<IP>` when `collapse_repeats: true`, and does not when false

**Verification:**
- [ ] Tests pass: `pytest -q tests/property/ tests/unit/test_placeholders.py`
- [ ] `mypy src/ticketds` strict passes on `src/ticketds/redact/`

**Dependencies:** T2 (config); can run in parallel with Phase 1

**Files likely touched:** `src/ticketds/redact/__init__.py`,
`src/ticketds/redact/placeholders.py`, `tests/property/test_redaction_properties.py`,
`tests/unit/test_placeholders.py`

**Estimated scope:** M

---

### Task 9: Layer 1 — structural rules in `redaction.yaml`

**Description:** Write the structural detectors as YAML data: emails, URLs, IPs, UUIDs, JWTs, AWS
keys, `sk_`/`whsec_` prefixes, PEM headers, connection strings with inline passwords,
`Authorization:` headers, hostnames/pods/buckets/queues. Every rule declares `name`, `pattern`,
`entity_type`, `confidence`, and **at least one `example` the test suite asserts it matches** —
which is what makes the zero-match gate mechanical rather than aspirational.

**Acceptance criteria:**
- [ ] Every rule in `configs/redaction.yaml` has ≥1 `example`, and a parametrised test asserts each
      rule matches each of its own examples
- [ ] Hit counts on the full corpus are within tolerance of §4.3: Email 566/468 rows, URL 568/542,
      Phone 286/267, IPv4 1218/366, bearer 194/190, AWS AKID 125/124, connstr 98/98, JWT 9/9,
      PEM 4/4
- [ ] **Build-failing gate:** any detector matching zero spans across the corpus fails the build,
      with an explicit allowlist entry required to record why zero is genuinely expected
- [ ] Fine-grained `entity_type` maps to the coarse placeholder per Appendix A (e.g. `AWS_AKID` →
      `<SECRET>`)

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_redaction_rules.py`
- [ ] Manual: `python -m ticketds.cli report --stage redact --config …` prints the §4.3 entity table

**Dependencies:** T8

**Files likely touched:** `configs/redaction.yaml`, `src/ticketds/redact/rules.py`,
`tests/unit/test_redaction_rules.py`

**Estimated scope:** M

---

### Task 10: Layers 2–3 — context rules with checksum-as-confidence

**Description:** Add the keyword-anchored and Russian-entity detectors that Presidio does not ship:
ИНН, КПП, ОГРН, ОГРНИП, СНИЛС, БИК, р/с, к/с, passport, IBAN, plus masked and partial card forms.
Implement §4.4's decision: **checksums adjust confidence, they never gate redaction.** Luhn, ИНН and
ОГРН check digits set `confidence`, and a failing checksum still redacts and flags the row for S6.

**Acceptance criteria:**
- [ ] Checksum failure never suppresses a redaction; a test asserts a check-digit-invalid ИНН is
      still replaced by `<ORG_ID>` with `confidence: low`
- [ ] Hit counts within tolerance of §4.3: ИНН 89/86 rows, КПП 52/51, Stripe-style 51/51, card-like
      39/39, р/с к/с 31/16, ОГРН 19/19, БИК 15/15, СНИЛС 13/13
- [ ] Masked and partial card forms match (`4111 11** **** 1111`, `карта ****1111`) — the §4.3 gap
      of 50 rows closes measurably
- [ ] Spaced IBANs match (`FR00 3000 3000 0000 0000 0000`, ≥16 instances) — the unspaced pattern
      that matched 0 is not shipped
- [ ] Partial self-redaction detected on 168 rows under a stated predicate, with the predicate
      recorded next to the number

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_redaction_rules.py -k context`
- [ ] Manual: `report --stage redact` shows the §4.3 README-gap table with the card gap closed

**Dependencies:** T9

**Files likely touched:** `configs/redaction.yaml`, `src/ticketds/redact/rules.py`,
`src/ticketds/redact/checksums.py`, `tests/unit/test_redaction_checksums.py`

**Estimated scope:** M

---

### Task 11: Layer 4 — NER for `PERSON` / `ORG` / `ADDR`

**Description:** Wire Presidio's global recognizers plus a Russian spaCy pipeline
(`ru_core_news_lg`) for the open-class entities no pattern can catch. This is the heaviest and
least precise layer, and open question 4 asks whether it earns its cost — so it ships behind a
config flag and its cost/precision are reported, letting S6 answer the question with evidence.

**Acceptance criteria:**
- [ ] Presidio global recognizers are enabled for `EMAIL_ADDRESS`, `CREDIT_CARD`, `IBAN_CODE`,
      `IP_ADDRESS`, `PERSON`, `LOCATION`, `PHONE_NUMBER`, `URL`, `UUID`, `CRYPTO`, `MAC_ADDRESS`,
      `DATE_TIME`, `NRP`, `MEDICAL_LICENSE`
- [ ] `redaction.ner.enabled` toggles the layer; with it off the pipeline still runs and the
      manifest records the choice
- [ ] Custom Russian recognizers registered for entities Presidio does not ship (verified: it has
      no ИНН/СНИЛС/ОГРН/КПП/БИК/passport recognizer)
- [ ] Per-layer wall-clock and per-entity precision on the gold set are reported, so open question 4
      can be closed with a number

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_redaction_ner.py`
- [ ] Manual: run S2 with the layer on and off; confirm both complete and the manifest differs

**Dependencies:** T10

**Files likely touched:** `src/ticketds/redact/ner.py`, `configs/redaction.yaml`,
`requirements.txt`, `tests/unit/test_redaction_ner.py`

**Estimated scope:** M

---

### Task 12: S2 `redact` stage and the span sidecar

**Description:** Wire the library into the pipeline as a stage: redact `title` and `description`,
emit `title_redacted` / `description_redacted`, and write `stage/s2_spans.parquet` with one row per
detected span. The sidecar is what lets an auditor ask "did we ever miss an ОГРН" without a full
re-run, and what S6 uses to reason about coverage.

**Acceptance criteria:**
- [ ] Sidecar columns: `row_index`, `field`, `start`, `end`, `entity_type`, `placeholder`,
      `detector`, `confidence`
- [ ] The sidecar reconstructs the redacted text from the original exactly, for every row
- [ ] Idempotence holds on real corpus rows, not just generated text
- [ ] `report --stage redact` reproduces the §4.3 entity table, the README-gap table, and the Luhn
      result (6 of 39 card-like runs pass)
- [ ] `redaction.yaml`'s hash is recorded for the manifest — a rule change is a dataset version
      change

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s2_redact.py`
- [ ] Manual: `python -m ticketds.cli stage redact --config …` then `report --stage redact`

**Dependencies:** T11, T7

**Files likely touched:** `src/ticketds/stages/s2_redact.py`, `src/ticketds/schema.py`,
`src/ticketds/metrics.py`, `tests/unit/test_s2_redact.py`

**Estimated scope:** M

---

### Task 13: Gold-set sampler and `eval-redaction`

**Description:** Build the tooling for §4.6: a stratified sampler that selects the 200 rows a human
will annotate (80 with a detected span, 60 with a payload but **no** detected span — the
highest-value stratum, 40 random no-payload, 20 partial self-redaction), an annotation shell for
`tests/fixtures/span_gold.jsonl`, and the `eval-redaction` command computing span recall, span
precision and over-redaction rate per entity type against the gates.

**Acceptance criteria:**
- [ ] Sampling is deterministic given `seed` and reproduces the four strata at the stated sizes
- [ ] `eval-redaction --gold tests/fixtures/span_gold.jsonl` reports recall, precision and
      over-redaction rate per entity type
- [ ] Gates enforced: recall 1.00 on `SECRET`/`CARD`/`BANK_ID`/`GOV_ID` → build fails; recall <0.95
      elsewhere → build fails; precision <0.90 → warn
- [ ] Runs against a small committed sample gold file so the command is testable before T14 lands
- [ ] Output states that these numbers are a **floor on quality, not an estimate of production
      recall** — the gold set is drawn from the same synthetic distribution (311 rows contain
      `EXAMPLE`/`DO-NOT-USE`)

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_eval_redaction.py`
- [ ] Manual: `python -m ticketds.cli eval-redaction --gold tests/fixtures/span_gold.sample.jsonl`

**Dependencies:** T12

**Files likely touched:** `src/ticketds/redact/eval.py`, `src/ticketds/cli.py`,
`tests/fixtures/span_gold.sample.jsonl`, `tests/unit/test_eval_redaction.py`

**Estimated scope:** M

---

### Task 14: **[human, owner: Eng]** Annotate the 200-row span gold set

**Description:** A human annotates every sensitive span in the 200 sampled rows. **[estimate]** ~200
rows × ~3 min = ~10 person-hours, once. This is spec open question 2, and it **blocks the §4.6
recall gate, which blocks success criteria 4, 5, 8 and 9.** Scheduled here so the hours are booked
before Phase 2 closes rather than discovered in Phase 6.

**Acceptance criteria:**
- [ ] `tests/fixtures/span_gold.jsonl` contains 200 annotated rows in the sampler's schema
- [ ] Every span carries `start`, `end`, `text`, `entity_type` from the Appendix A fine-grained set
- [ ] Annotation was done against the **original** text, not the redacted output
- [ ] `eval-redaction` runs clean against it and its numbers land in the datasheet

**Verification:**
- [ ] `python -m ticketds.cli eval-redaction --gold tests/fixtures/span_gold.jsonl` executes and
      reports per-entity recall
- [ ] Manual: spot-check 10 rows against the raw CSV

**Dependencies:** T13

**Files likely touched:** `tests/fixtures/span_gold.jsonl`

**Estimated scope:** S (engineering) / ~10 person-hours (annotation) — **human-blocked**

---

### Task 15: Redaction regression harness

**Description:** Every confirmed miss becomes a permanent test. Build the golden-test harness over
`tests/fixtures/redaction_regressions.jsonl` so that a miss found by S6 (or by a human) can be
fixed once and stay fixed forever.

**Acceptance criteria:**
- [ ] Each regression entry is `{text, expected_spans, entity_type, source, added_at}` and is
      asserted by a parametrised test
- [ ] Adding an entry that current rules do not catch fails the suite, and the failure message names
      the entity type and the uncaught span
- [ ] A documented one-command path from "S6 confirmed a miss" to "regression entry added"
- [ ] §11 is enforced socially and mechanically: disabling a rule to make a test pass is not a fix

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_redaction_golden.py`
- [ ] Manual: add a deliberately uncaught entry, confirm red; add the rule, confirm green

**Dependencies:** T13

**Files likely touched:** `tests/unit/test_redaction_golden.py`,
`tests/fixtures/redaction_regressions.jsonl`, `src/ticketds/redact/eval.py`

**Estimated scope:** S

---

### Checkpoint C: Redaction is measured, not asserted

- [ ] `eval-redaction` passes its gates against the completed gold set (T14)
- [ ] No detector returns zero matches across the corpus
- [ ] Property tests (idempotence, closed vocabulary, span reconstruction) pass under `hypothesis`
- [ ] The `redaction.yaml` hash is wired for the manifest and for `preprocess.json`
- [ ] Human review before Phase 3 — and a decision on open question 4 (is `<PERSON>` NER worth it?)

---

## Phase 3: Dedup and enrichment

### Task 16: S3 `dedup` — exact and MinHash near-duplicate clustering

**Description:** Identify near-duplicate clusters so the splitter can keep them on one side of a
boundary. Runs **after** S2, because replacing request ids and hostnames with placeholders collapses
two descriptions of the same outage onto each other and makes duplicates visible.

**Acceptance criteria:**
- [ ] Exact: SHA-256 of `title_redacted + "\n" + description_redacted` finds 6 rows
- [ ] Near: MinHash (128 permutations) over character 5-grams, LSH banding for candidates, exact
      Jaccard to confirm; ≥0.80 → 464 pairs marked `dup_exact`, ≥0.60 → 599 pairs / 371 rows
      clustered
- [ ] `dup_cluster_id` assigned by connected components over the ≥0.60 graph; adds `text_sha256`,
      `dup_max_jaccard`, `is_exact_dup`
- [ ] Clustering is deterministic given the seed — two runs produce identical cluster ids
- [ ] `report --stage dedup` prints the cluster count, size histogram, and the raw-vs-redacted
      comparison (583/429/348 raw vs 599/464/371 redacted)

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s3_dedup.py`
- [ ] Manual: `python -m ticketds.cli report --stage dedup --config …` reproduces §1.3's table

**Dependencies:** T12

**Files likely touched:** `src/ticketds/stages/s3_dedup.py`, `src/ticketds/metrics.py`,
`src/ticketds/schema.py`, `tests/unit/test_s3_dedup.py`

**Estimated scope:** M

---

### Task 17: S4 `enrich` — `model_input`, lengths, payload, bursts

**Description:** Compute everything the splitter and slice-based evaluation need. The
`model_input` builder is the shared preprocessing function the model service imports — it must be
byte-identical at inference or you get train/serve skew, so it is imported here, never
reimplemented.

**Acceptance criteria:**
- [ ] `model_input` = `"[priority: {p}] {title}\n{description}"` on redacted text, built by the
      shared function and non-empty for every non-quarantined row
- [ ] `char_len` and `est_tokens` computed over the assembled input; the length distribution
      reproduces p50 222 / p90 688 / p99 1,040 / max 1,691 for descriptions
- [ ] `has_payload` set from any S2 span or the pasted-block heuristic (~30% of rows)
- [ ] `burst_id` from `created_at` bucketed to 6h × dominant service above the 99th-percentile
      count; 8 bursts found, largest 109 tickets on 2025-11-18
- [ ] A test asserts the assembly function is imported, not duplicated, in the stage

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s4_enrich.py`
- [ ] Manual: `report --stage enrich` prints the length distribution and burst table

**Dependencies:** T16

**Files likely touched:** `src/ticketds/stages/s4_enrich.py`, `src/ticketds/preprocess.py`,
`src/ticketds/metrics.py`, `tests/unit/test_s4_enrich.py`

**Estimated scope:** M

---

### Task 18: S4 language detection — payload-stripped vs naive

**Description:** Language is a first-class evaluation slice in classifier spec §6.3, and a slice
built on a bad rule measures nothing. A naive Cyrillic-ratio rule calls 1,123 rows "mixed" where the
generator recorded 237 — a 4.7× over-count, because a Russian ticket quoting an English stack trace
is Russian prose. Run a real langid model over the **prose only**, stripping the payload blocks S2
already located, and record both verdicts.

**Acceptance criteria:**
- [ ] `lang ∈ {ru, en, other}` plus confidence, computed on payload-stripped text via `py3langid`
- [ ] Both the naive character-ratio verdict and the payload-stripped verdict are stored
- [ ] The disagreement rate is computed and carried into the manifest and the datasheet
- [ ] The naive rule reproduces 1,123 "mixed" rows, confirming the finding rather than hiding it

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s4_lang.py`
- [ ] Manual: `report --stage enrich` prints the RU:EN ratio and the disagreement rate

**Dependencies:** T17

**Files likely touched:** `src/ticketds/stages/s4_enrich.py`, `src/ticketds/metrics.py`,
`tests/unit/test_s4_lang.py`

**Estimated scope:** S

---

### Checkpoint D: Splitter inputs complete

- [ ] `run` executes S0→S4 with full row accounting on all 5,013 rows
- [ ] `report --stage dedup` and `report --stage enrich` reproduce §1.3, §S4 and §4.3 claims
- [ ] `model_input` is built by the shared function that the model service will import
- [ ] Human review before Phase 4

---

## Phase 4: Splits

### Task 19: S7 `split` — temporal, gapped, cluster-aware

**Description:** Sort by `created_at`; train = oldest 70%, 7-day gap, val = 15%, 7-day gap, test =
newest 15%. Rows landing in a gap are quarantined, not deleted. Any `dup_cluster_id` spanning a
boundary is assigned wholesale to the **earlier** split and its later members quarantined: keeping a
duplicate in train is harmless, keeping one in test inflates the score.

**Acceptance criteria:**
- [ ] Boundaries land at 2026-04-06 and 2026-06-04 over the 4,622 labelled rows, producing
      train 3,237 / val 621 / test 627 with 137 rows (3.0%) in the gaps
- [ ] Gap rows quarantine with `quarantine_reason = "split_gap"`, `quarantine_stage = "s7"`
- [ ] Cross-boundary cluster members quarantine with `dup_exact` or `dup_cross_boundary`, retaining
      `dup_cluster_id` so the surviving twin is findable; the two counts are reported separately
- [ ] Row accounting still balances: no row is deleted anywhere in S7

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s7_split.py`
- [ ] Manual: `python -m ticketds.cli report --stage split --config …` prints split sizes and gap
      loss

**Dependencies:** T18

**Files likely touched:** `src/ticketds/stages/s7_split.py`, `src/ticketds/schema.py`,
`tests/unit/test_s7_split.py`

**Estimated scope:** M

---

### Task 20: Random diagnostic split and leakage comparison

**Description:** Produce a random split purely as the drift diagnostic classifier spec §2.6
requires — never trained on — and measure the leakage difference. The gap between random and
temporal numbers is the drift magnitude, and it is a number the product owner should see. Also
measure and report organisation overlap, since §6.3 decides not to enforce group separation but to
be honest about it.

**Acceptance criteria:**
- [ ] `report --stage split --compare-strategies` reproduces 6.2% (random) vs 3.7% (temporal) of
      test rows having a ≥0.60 near-duplicate in train
- [ ] The random split is written as a diagnostic artifact and is clearly not a training input
- [ ] Organisation overlap reported: all 184 organisations in the temporal test split also appear in
      train; plus the share of test rows whose org has >50 training tickets
- [ ] Both numbers reach the manifest and the datasheet

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_split_leakage.py`
- [ ] Manual: `report --stage split --compare-strategies`

**Dependencies:** T19

**Files likely touched:** `src/ticketds/stages/s7_split.py`, `src/ticketds/metrics.py`,
`tests/unit/test_split_leakage.py`

**Estimated scope:** M

---

### Task 21: Per-service counts, `measurable` flag, split integrity gates

**Description:** With one positive in test, a service's F1 is a coin flip with a decimal point.
Compute per-service positives per split, mark services below `measurable_min_test_positives: 20` as
`measurable: false`, and make the split integrity rules build-failing tests so they cannot be
forgotten under deadline.

**Acceptance criteria:**
- [ ] Per-service train/val/test positive counts written to the manifest, reproducing §6.5
      (`terraform-provider` 13/1/1, `message-queue` 25/5/**0**, `cdn` 26/6/4, `dns` 31/5/5,
      `managed-redis` 67/12/17)
- [ ] Services with <20 test positives flagged `measurable: false` and listed by name
- [ ] **Build-failing gates:** no row present in two splits; no `dup_cluster_id` spanning a
      boundary; no `label_status == "unlabelled"` row in train, val or test
- [ ] `message-queue`'s zero test positives surface as an explicit undefined-F1 warning, not a
      silent zero

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_split_integrity.py`
- [ ] Manual: `report --stage split` lists unmeasurable services by name

**Dependencies:** T20

**Files likely touched:** `src/ticketds/metrics.py`, `src/ticketds/stages/s7_split.py`,
`tests/unit/test_split_integrity.py`

**Estimated scope:** M

---

### Checkpoint E: A dataset that can be trained on

- [ ] `run` executes S0→S4→S7 (LLM stages skipped) and produces three splits
- [ ] Every §10 split gate passes: no cross-split rows, no cluster spanning a boundary, no
      unlabelled rows in a split
- [ ] `report --stage split --compare-strategies` reproduces the leakage delta and the org overlap
- [ ] Human review before Phase 5 — decide open question 5 (is 20 the right `measurable` floor?)

---

## Phase 5: LLM stages

S5 and S6 exist for **opposite** reasons and must not be built the same way. S5 is a *sampling*
problem: find the wrong rows cheaply; being wrong costs one noisy label. S6 is a *recall* problem:
find what the rules missed; being wrong is an incident. The LLM never writes a label and never
edits text.

### Task 22: OpenRouter client — structured output, retry, call ceiling

**Description:** One HTTP client for both stages, using OpenRouter's OpenAI-compatible surface so
the pipeline is not locked to a vendor. Structured output with a JSON schema, exponential backoff
with jitter on 429/5xx, and a hard `llm.max_calls` ceiling so a bug cannot spend unbounded money.

**Acceptance criteria:**
- [ ] Structured output enforced by JSON schema; a malformed response is retried, then surfaced as
      a typed error, never silently coerced
- [ ] Exponential backoff with jitter on 429/5xx; a test asserts the retry schedule without sleeping
- [ ] `llm.max_calls` is a hard stop — exceeding it aborts the stage with a clear error
- [ ] `temperature = 0` set, with a comment recording that determinism comes from the cache, not
      from sampling parameters
- [ ] `cache_read_input_tokens` is read off each response so prompt-cache failure is visible (the
      minimum cacheable prefix is 512 tokens on Opus 5, 1,024 on Sonnet 5, but 4,096 on Haiku 4.5 —
      our ~2,000-token prefix silently does not cache there)

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_llm_client.py` (network stubbed)
- [ ] Manual: one live call against a trivial prompt with `max_calls: 1`

**Dependencies:** T4

**Files likely touched:** `src/ticketds/llm/client.py`, `tests/unit/test_llm_client.py`,
`requirements.txt`

**Estimated scope:** M

---

### Task 23: Content-addressed LLM cache and `--dry-run` cost estimate

**Description:** Cache every response to `data/cache/llm/{key}.json`, keyed by
`sha256(model_id + prompt_version + rendered_prompt + params)`. This is what makes re-running the
pipeline free, offline, and reproducible — the property §1.7 requires and that a low temperature
cannot deliver on its own. Add `--dry-run`, which renders prompts and prints projected token count
and cost without calling anything.

**Acceptance criteria:**
- [ ] A second run with an unchanged config and warm cache makes zero network calls
- [ ] Changing the prompt or the model invalidates exactly the affected keys and nothing else
- [ ] Cache hit/miss counts and model ids are collected for the manifest
- [ ] `run --dry-run` prints projected calls, tokens and cost, and makes no network call — asserted
      by a test that fails if the HTTP layer is touched

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_llm_cache.py`
- [ ] Manual: `python -m ticketds.cli run --config configs/dataset.v1.yaml --dry-run`

**Dependencies:** T22

**Files likely touched:** `src/ticketds/llm/cache.py`, `src/ticketds/cli.py`,
`tests/unit/test_llm_cache.py`

**Estimated scope:** M

---

### Task 24: S5 prompt v1, prompt hashing, evidence validation

**Description:** Write `prompts/label_review.v1.md` as constrained adjudication over the closed
20-service taxonomy, with the taxonomy definitions and few-shot examples in a **stable prefix** so
prompt caching engages and per-ticket content last. Require an `evidence` quote: a model forced to
quote the span justifying `billing` cannot hand-wave, and it is the cheapest hallucination check
available. A prompt change is a dataset change, so its hash goes in the manifest.

**Acceptance criteria:**
- [ ] Output schema constrains `service` by `enum` to the 20 services, plus `accept` (boolean) and
      `evidence` (a quoted span)
- [ ] Evidence validation: a quote not present in the ticket discards the verdict automatically
- [ ] The stable prefix is ~2,000 tokens and is byte-identical across calls; the per-ticket suffix
      carries only ticket content
- [ ] The prompt file's SHA-256 is recorded for the manifest
- [ ] S5 sends **redacted** text only — a test asserts no raw field can reach the S5 renderer

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s5_prompt.py`
- [ ] Manual: `run --dry-run` prints the rendered prompt for one ticket and its token count

**Dependencies:** T23, T7

**Files likely touched:** `prompts/label_review.v1.md`, `src/ticketds/stages/s5_label_review.py`,
`tests/unit/test_s5_prompt.py`

**Estimated scope:** M

---

### Task 25: S5 Pass A — blind measurement sample with bootstrap CI

**Description:** Random sample of 500 labelled rows; the LLM proposes a service set **blind** —
never seeing the stored `services` — and we compare. Blind because showing the stored label measures
whether the model can be talked into agreeing; random because sampling suspicious-looking rows
measures your suspicion, not the corpus. This produces the unbiased label-noise estimate that goes
in the datasheet.

**Acceptance criteria:**
- [ ] 500 rows sampled at random from labelled rows, deterministic given `seed`
- [ ] The prompt renderer cannot access `services` for Pass A — enforced by the record schema, not
      by convention
- [ ] Agreement rate reported with a bootstrap confidence interval
- [ ] Evidence-span validity ≥0.95 gate enforced
- [ ] The estimate is reported, not gated — it is a measurement of the corpus, not a pass/fail

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s5_pass_a.py` (client stubbed)
- [ ] Manual: `stage llm-label-review --config …` against a warm cache reports agreement and CI

**Dependencies:** T24

**Files likely touched:** `src/ticketds/stages/s5_label_review.py`, `src/ticketds/metrics.py`,
`tests/unit/test_s5_pass_a.py`

**Estimated scope:** M

---

### Task 26: S5 Pass B — triage ranking and `review/labels.jsonl`

**Description:** Same blind prompt, full sweep, producing a `disagreement_score` per row from the
weighted set difference, the model's confidence, `label_dirty_flags` from S1, and `n_services == 0`.
Rank, take the top K = 400, write the queue with both label sets and the model's evidence. This
replaces 77 person-hours of reviewing all 4,622 labelled rows with ~15 — and Pass A is what lets us
state what that trade missed instead of pretending it missed nothing.

**Acceptance criteria:**
- [ ] `review/labels.jsonl` holds exactly K=400 ranked rows with ticket text, stored labels,
      proposed labels, and quoted evidence
- [ ] All 391 unlabelled rows enter the queue by construction
- [ ] `llm_disagreement_score` written back to the frame for the output schema
- [ ] **The stage never overwrites a label.** A test asserts no code path writes to `services_norm`
      from an LLM response — the LLM ranks, a human decides
- [ ] Queue precision gate (≥0.30 corrected on the first human pass) is implemented as a recorded
      metric with a documented measurement procedure

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s5_pass_b.py`
- [ ] Manual: inspect 5 queue entries; each has readable evidence quoted from its ticket

**Dependencies:** T25

**Files likely touched:** `src/ticketds/stages/s5_label_review.py`, `src/ticketds/metrics.py`,
`tests/unit/test_s5_pass_b.py`

**Estimated scope:** M

---

### Task 27: S6 egress gating and stratified sampling

**Description:** S6 must see the **original** text — you cannot audit a redactor using only its
output — so it is the one stage that sends raw text, and it does so only under
`llm.allow_raw_text: true`, which the default config ships as `false`. Build the three-way
behaviour matrix and the stratified sampler. On this synthetic corpus the switch is theatre; on
production data it is the control that makes the stage legal, and building it now costs nothing.

**Acceptance criteria:**
- [ ] `run` with `allow_raw_text: false` **skips** S6 and the manifest records
      `s6: skipped (allow_raw_text=false)`
- [ ] `run --audit-redaction` with the flag `false` **fails immediately**, naming the flag — never a
      silent skip when the stage was asked for
- [ ] `run --audit-redaction` with the flag `true` runs S6
- [ ] Strata sampled deterministically: no-span-with-payload (~274, recomputed from `has_payload`),
      low-confidence spans (~200), partial self-redaction (168), random control (150) ≈ 790 rows
- [ ] The manifest distinguishes audited from unaudited builds, so nobody has to remember which is
      which

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s6_gating.py`
- [ ] Manual: run all three invocations; confirm skip, hard failure, and run

**Dependencies:** T23, T12

**Files likely touched:** `src/ticketds/stages/s6_redaction_audit.py`, `src/ticketds/cli.py`,
`tests/unit/test_s6_gating.py`

**Estimated scope:** M

---

### Task 28: S6 audit — span verification and `review/redaction.jsonl`

**Description:** Each call sends three things — the original text, the redacted text, and the span
list with character offsets — and asks one narrow question: report only identifiers **not covered by
a listed range**. Sending the redacted text alone makes misses invisible; sending the raw text alone
buries the real misses under hundreds of confirmations. Returned spans are then verified in code:
the model nominates, set arithmetic is the pipeline's, and rules do the redacting.

**Acceptance criteria:**
- [ ] Every returned span is verified programmatically — does the offset actually contain the quoted
      text? — and hallucinated spans are dropped automatically
- [ ] Any returned span intersecting a known S2 span is dropped in code, never trusted from the model
- [ ] Surviving spans are written to `review/redaction.jsonl` for a human, with the entity type and
      the model's stated reason
- [ ] **The LLM never edits text** — a test asserts no code path writes model output into any text
      column
- [ ] **Publishability gate:** a confirmed miss in `SECRET`, `CARD`, `BANK_ID` or `GOV_ID` marks the
      dataset version not publishable until a rule covers it and its regression test passes
- [ ] Confirmed misses flow into `redaction_regressions.jsonl` via the T15 path

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s6_audit.py` (client stubbed, including a hallucinated
      span and an overlapping span)
- [ ] Manual: `run --audit-redaction` with the flag set; inspect 5 queue entries

**Dependencies:** T27, T15

**Files likely touched:** `src/ticketds/stages/s6_redaction_audit.py`,
`prompts/redaction_audit.v1.md`, `tests/unit/test_s6_audit.py`

**Estimated scope:** M

---

### Checkpoint F: LLM stages, cached and gated

- [ ] A second `run` with a warm cache makes zero network calls and produces identical outputs
- [ ] `--dry-run` prints projected cost and touches no network
- [ ] No code path lets an LLM write a label or edit text
- [ ] S6's three-way gate behaves exactly as §5.3 specifies
- [ ] Human review before Phase 6 — and confirm open question 3 (does queue depth 400 match human
      capacity?)

---

## Phase 6: Freeze, manifest, datasheet

### Task 29: S8 `freeze` — output schema and split artifacts

**Description:** Write `train/val/test.parquet` in the Appendix C schema, plus `quarantine.parquet`
carrying every source column with its reason and stage. Parquet, not CSV: a CSV round-trip loses the
distinction between empty string and null, and 391 rows depend on exactly that.

**Acceptance criteria:**
- [ ] Every Appendix C column present with its stated type, including `y` as a 20-wide
      `List[UInt8]` in taxonomy order and `human_reviewed` defaulting to false
- [ ] `quarantine.parquet` carries every source column plus `quarantine_reason` and
      `quarantine_stage`; every reason used anywhere in the pipeline appears in a documented enum
- [ ] Row accounting closes: 5,013 = train + val + test + quarantined, asserted at write time
- [ ] `organization_id` is present for group reporting and documented as **not a feature**

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_s8_freeze.py`
- [ ] Manual: load `train.parquet` in Polars and confirm dtypes match Appendix C exactly

**Dependencies:** T21, T26

**Files likely touched:** `src/ticketds/stages/s8_freeze.py`, `src/ticketds/schema.py`,
`tests/unit/test_s8_freeze.py`

**Estimated scope:** M

---

### Task 30: `manifest.json` assembly and the `verify` command

**Description:** The manifest is the reviewable artifact — it is committed while the parquet is not,
because a diff on it in a pull request shows exactly what a config change did to the dataset. Then
`verify` proves the claim: an unchanged config plus unchanged input plus a warm cache reproduces
every output hash, or the pipeline is broken.

**Acceptance criteria:**
- [ ] Manifest carries: input CSV SHA-256, resolved config, git commit, `platform.python_version`,
      every pinned package version, per-stage rows in/out/quarantined, SHA-256 of every output file,
      LLM cache hit/miss counts and model ids, prompt hashes, `redaction.yaml` hash, wall-clock per
      stage, and the S6 audited/unaudited marker
- [ ] `python -m ticketds.cli verify --version v1` recomputes and compares every hash, exiting
      non-zero on any mismatch
- [ ] Wall-clock and other non-deterministic fields are excluded from the reproducibility comparison
      but retained in the manifest
- [ ] **Build-failing gate:** a repeated build producing a different manifest hash fails

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_manifest.py`
- [ ] Manual: `run` twice, then `verify --version v1` — passes

**Dependencies:** T29

**Files likely touched:** `src/ticketds/pipeline.py`, `src/ticketds/cli.py`,
`tests/unit/test_manifest.py`

**Estimated scope:** M

---

### Task 31: `annotation_queue.jsonl` — blind, iteratively stratified

**Description:** This pipeline cannot build a gold evaluation set — that needs ≥2 blind annotators
and adjudication, and this CSV's labels came from the generator that wrote the text. So it emits a
**sampling frame** instead, stratified over (month × language × cardinality). This is the one place
`iterative-stratification` is used, because here we have full freedom over which rows to send to
humans and getting rare services represented is the entire point.

**Acceptance criteria:**
- [ ] Stratified over month × language × cardinality using `iterative-stratification`, deterministic
      given `seed`
- [ ] **Blind:** no `services` field exists in the record schema at all — omitted, not blanked, so
      it cannot leak through a UI bug. A test asserts the key is absent
- [ ] Taxonomy definitions are attached to the queue file
- [ ] The §6.2 fallback (a simple grouped temporal sample) is implemented and tested, so a break in
      the quiet 0.1.9 dependency does not block the build

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_annotation_queue.py`
- [ ] Manual: `grep -c services data/processed/v1/annotation_queue.jsonl` returns 0

**Dependencies:** T29

**Files likely touched:** `src/ticketds/stages/s8_freeze.py`,
`src/ticketds/annotation.py`, `tests/unit/test_annotation_queue.py`

**Estimated scope:** M

---

### Task 32: `datasheet.md` generation

**Description:** Generate the datasheet following the *Datasheets for Datasets* structure, opening
with §1.8's warning — **any F1 computed on this CSV describes the generator, not the world** — so a
teammate who picks up `test.parquet` six months from now hits it before they hit a number. Generated
by the pipeline, never written by hand: the pipeline cannot socially prevent someone putting
macro-F1 = 0.86 on a slide, but it can make the warning impossible to miss.

**Acceptance criteria:**
- [ ] Opens with the §1.8 warning, and states in those words that until humans annotate the
      queue, `test.parquet` is a **smoke-test set**
- [ ] Contains: class balance, cardinality histogram, RU:EN ratio and the langid disagreement rate,
      token-length distribution, per-service test counts with `measurable: false` services named,
      organisation overlap, the random-vs-temporal leakage delta, and the redaction recall estimate
      with its confidence interval
- [ ] States the MinHash blind spot: lexical overlap only, so semantically identical tickets in
      different words do not cluster
- [ ] Marks builds where S6 was skipped as **unaudited** and not eligible to train a shipping model
- [ ] All seven *Datasheets* sections present: motivation, composition, collection, preprocessing,
      uses, distribution, maintenance
- [ ] A test asserts the first paragraph contains the warning — regeneration cannot quietly drop it

**Verification:**
- [ ] Tests pass: `pytest -q tests/unit/test_datasheet.py`
- [ ] Manual: read the generated datasheet end to end; every number traces to a `report` command

**Dependencies:** T30, T31

**Files likely touched:** `src/ticketds/datasheet.py`, `tests/unit/test_datasheet.py`

**Estimated scope:** M

---

### Task 33: Integration test — full pipeline on `slice_200.csv`, reproducible

**Description:** A deterministic 200-row slice with the LLM client stubbed, running every stage end
to end in seconds, asserting manifest reproducibility. This is the test that keeps success criteria
1, 2 and 3 true as the code changes, rather than proving them once by hand.

**Acceptance criteria:**
- [ ] `tests/fixtures/slice_200.csv` is deterministically derived from the real CSV and includes
      mojibake rows, duplicate clusters, unlabelled rows and payload-bearing rows
- [ ] `pytest -q -m integration` runs S0→S8 with a stubbed LLM client in under a minute
- [ ] The test asserts identical output hashes across two consecutive runs
- [ ] The test asserts row accounting: rows in = rows out + quarantined, at every stage boundary

**Verification:**
- [ ] Tests pass: `pytest -q -m integration`
- [ ] Manual: delete `data/stage/` and re-run; hashes are unchanged

**Dependencies:** T32

**Files likely touched:** `tests/integration/test_full_pipeline.py`,
`tests/fixtures/slice_200.csv`, `tests/conftest.py`

**Estimated scope:** M

---

### Checkpoint G: Complete

- [ ] All ten §12 success criteria are mechanically checkable and checked
- [ ] `verify --version v1` passes on two consecutive builds
- [ ] `datasheet.md` opens with the §1.8 warning and names every `measurable: false` service
- [ ] Every `[measured]` claim in the spec is reproducible via a `report` subcommand (Appendix D)
- [ ] Human review before Phase 7

---

## Phase 7: Ship readiness

### Task 34: CI workflow running every build-failing gate

**Description:** Make the §10 gates run on every push. A zero-match detector is the quietest
possible bug in a safety-critical path, so it has to be loud somewhere — CI is where.

**Acceptance criteria:**
- [ ] CI runs `ruff check`, `ruff format --check`, `mypy src/ticketds`, `pytest -q` and
      `pytest -q -m integration`
- [ ] Every §10 build-failing gate is enforced: zero-match detector; span recall <1.00 on
      `SECRET`/`CARD`/`BANK_ID`/`GOV_ID`; span recall <0.95 elsewhere; a row still matching the
      mojibake signature after S0; a row in two splits; a cluster spanning a boundary; an
      unlabelled row in a split; a repeated build with a different manifest hash
- [ ] A committed `*.ipynb` fails CI (see the no-notebooks decision above)
- [ ] The pipeline job runs without network access, proving the warm-cache offline property

**Verification:**
- [ ] Manual: push a branch with a deliberately disabled detector; confirm CI goes red
- [ ] Manual: push a branch with a committed notebook; confirm CI goes red

**Dependencies:** T33

**Files likely touched:** `.github/workflows/ci.yml`, `pyproject.toml`,
`tests/unit/test_repo_hygiene.py`

**Estimated scope:** S

---

### Task 35: README quickstart and the committed manifest

**Description:** Success criterion 10: a teammate who has not read the spec runs the pipeline from
the README alone and gets hash-identical output. Commit `manifest.json` (not the parquet) so pull
requests show what a config change did to the dataset.

**Acceptance criteria:**
- [ ] README carries the §7 command set: setup, `run`, `run --audit-redaction`, `stage`,
      `--dry-run`, `report`, `eval-redaction`, `verify`, and the quality gates
- [ ] README states the default build skips S6 and what that means for publishability
- [ ] `data/processed/v1/manifest.json` is committed; the parquet, stage and cache directories are
      gitignored
- [ ] A teammate who has not read the spec reproduces the documented hashes — verified with a real
      person, not assumed

**Verification:**
- [ ] Manual: fresh clone, follow the README, run `verify --version v1` — passes
- [ ] `git status` clean after a full run

**Dependencies:** T34

**Files likely touched:** `README.md`, `data/processed/v1/manifest.json`, `.gitignore`

**Estimated scope:** S

---

## Parallelization Opportunities

| Track | Tasks | Notes |
|---|---|---|
| **Must be sequential** | T1 → T2 → T3 → T4 | Everything imports the config and stage contract |
| **Parallel after T4** | Phase 1 (T5–T7) ‖ Phase 2 redaction library (T8–T11) | The library only needs the config; the stage wiring (T12) is the join point |
| **Parallel after T4** | T22–T23 (LLM client + cache) | No pipeline dependency; only needs the CLI and config |
| **Human track** | T14 annotation (~10 person-hours) | Start as soon as T13 emits the sample; it blocks Checkpoint C |
| **Must be sequential** | T16 → T17 → T18 → T19 → T20 → T21 | Each stage consumes the previous stage's columns |
| **Must be sequential** | T29 → T30 → T32 | The manifest needs outputs; the datasheet needs the manifest |
| **Needs coordination** | T17 `model_input` builder | Shared contract with the model service (classifier spec §2.9) — settle the signature before parallel work touches it |

Three agents can work usefully at once after Checkpoint A: one on Phase 1, one on the redaction
library, one on the LLM client and cache.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Span gold set (T14) is not annotated, so the redaction recall gate never runs | **High** — blocks success criteria 4, 5, 8, 9 and the whole point of §4 | Scheduled as an explicit task with an owner and an hour estimate; T13 ships a sample gold file so the command is testable meanwhile |
| A detector silently matches zero and looks like "this entity isn't in the corpus" | **High** — this already happened twice in §4.3 (IBAN 0/16, passport 0/~14) | Zero-match is a build-failing gate (T9), not a log line; every rule carries an example the suite asserts |
| Checksum filtering creeps back in as a "false-positive fix" | **High** — ИНН/СНИЛС/ОГРН have deliberately invalid check digits; recall would go to zero invisibly | §4.4 decision encoded as a test (T10): a check-digit-invalid ИНН must still redact |
| `iterative-stratification` 0.1.9 is quiet upstream and advertises Python 3.4–3.9 | Medium — blocks the annotation queue | Pinned; §6.2's grouped-temporal fallback implemented and tested in T31 |
| Prompt caching silently does not engage, tripling LLM cost | Low — the whole spread is $16–45 | T22 reads `cache_read_input_tokens` and surfaces a zero across repeated calls |
| Mojibake rows bypass redaction | **High** — the 9 rows hide a name, a forwarded email header and a production `.env` | Repair in S0 before S2 (T6), plus a build-failing gate that no row survives matching the signature |
| Someone trains on `test.parquet` and reports macro-F1 as a production number | **High** — the most likely way this project goes wrong is social, not technical | Generated datasheet opens with the warning (T32); unmeasurable services named; a test asserts the warning survives regeneration |
| LLM output drifts into the dataset as labels or edited text | **High** — distillation nobody chose, and it destroys evaluation independence | Asserted by tests in T26 and T28: no code path writes LLM output into a label or a text column |
| S6 raw-text egress on production data | Low here (synthetic corpus), **High** the day the fixture is replaced | `allow_raw_text` gate built now (T27), recorded in the manifest for every run |

---

## Open Questions

Carried from spec §13. Blocking ones first; each needs an owner before the affected task starts.

| # | Question | Blocks | Owner |
|---|---|---|---|
| **2** | Who annotates the 200-row span gold set, and when? (~10 person-hours) | **T14, Checkpoint C, and every redaction success criterion** | Eng |
| **1** | Are the 391 empty-`services` rows "no service applies" or "not triaged"? | T7 / Checkpoint B — currently quarantined, the safe default, possibly wasteful | Product |
| **6** | Keep the repaired mojibake rows, or quarantine them? | T6 / Checkpoint B — keeping is the recommendation; repair is proven on all 9 | Eng |
| **3** | Does queue depth 400 match available human capacity? | T26 — sets the S5 saving and the ≥0.30 precision gate | Eng lead |
| **4** | Is `<PERSON>` NER worth its cost on this corpus? | T11 — possible answer: ship without it and let S6 measure what was lost | Eng |
| **5** | Is 20 test positives the right `measurable` floor? | T21 — arbitrary today; a power calculation would replace the guess | Eng |
| **7** | When does the fixture get replaced by a real export? | Everything in §1.8 changes that day; until then no number here means anything about production | Product |

Two questions this plan answers rather than defers:

- **Jupyter notebooks?** No — see the decision above. Exploration goes in a scratchpad; findings
  graduate into `report` subcommands.
- **Where do tasks live?** `tasks/plan.md` for detail, `tasks/todo.md` for the checklist. No
  external tracker is configured in this repository.
