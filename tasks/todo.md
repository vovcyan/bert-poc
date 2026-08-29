# Tasks: CSV → dataset pipeline (`ticketds`)

Ordered checklist for implementation. **Full detail — acceptance criteria, verification commands,
dependencies, files touched — lives in [`tasks/plan.md`](./plan.md)**; this file is the tracking
surface. Work top to bottom: the order is the dependency order.

Every task must also clear
[`.claude/references/definition-of-done.md`](../.claude/references/definition-of-done.md).

Standing verification for every task:
`ruff check . && ruff format --check . && mypy src/ticketds && pytest -q`

---

## Phase 0: Foundation

- [ ] **T1** Project scaffold, dependency pins, and quality gates — *S* — deps: none
- [ ] **T2** Config layer: Pydantic models, `dataset.v1.yaml`, `taxonomy.yaml` — *M* — deps: T1
- [ ] **T3** Stage contract: `StageResult`, quarantine, Pandera schemas, row accounting — *M* — deps: T2
- [ ] **T4** CLI skeleton: `run` / `stage` / `report` / `verify` / `eval-redaction` — *M* — deps: T3

### Checkpoint A: Foundation
- [ ] All four quality commands pass on the scaffold
- [ ] `run` executes end to end with no stages registered
- [ ] Config hash is stable across processes
- [ ] Human review before Phase 1

## Phase 1: Ingest and labels

- [ ] **T5** S0 `ingest`: RFC 4180 read, strict decode, typed frame, quarantine — *M* — deps: T4
- [ ] **T6** S0 mojibake detection and CP1251 repair (9 rows; a redaction bypass) — *M* — deps: T5
- [ ] **T7** S1 `normalize-labels`: taxonomy validation and dirty flags — *M* — deps: T6

### Checkpoint B: Deterministic head of the pipeline
- [ ] S0→S1 runs on all 5,013 rows with full row accounting
- [ ] `report --stage ingest` and `report --stage normalize-labels` reproduce §1.2, §1.4, §S0
- [ ] No row survives S0 matching the mojibake signature
- [ ] Human review; resolve open questions 1 and 6

## Phase 2: The redaction library

- [ ] **T8** Placeholder vocabulary and span-application engine (+ property tests) — *M* — deps: T2
- [ ] **T9** Layer 1: structural rules in `redaction.yaml` (+ zero-match gate) — *M* — deps: T8
- [ ] **T10** Layers 2–3: context rules, checksum-as-confidence — *M* — deps: T9
- [ ] **T11** Layer 4: NER for `PERSON` / `ORG` / `ADDR` — *M* — deps: T10
- [ ] **T12** S2 `redact` stage and the span sidecar — *M* — deps: T11, T7
- [ ] **T13** Gold-set sampler and `eval-redaction` — *M* — deps: T12
- [ ] **T14** **[human, Eng, ~10 person-hours]** Annotate the 200-row span gold set — deps: T13
- [ ] **T15** Redaction regression harness — *S* — deps: T13

### Checkpoint C: Redaction is measured, not asserted
- [ ] `eval-redaction` passes its gates against the completed gold set
- [ ] No detector returns zero matches
- [ ] Idempotence, closed-vocabulary and span-reconstruction properties hold
- [ ] Human review; resolve open question 4

## Phase 3: Dedup and enrichment

- [ ] **T16** S3 `dedup`: exact + MinHash near-duplicate clustering — *M* — deps: T12
- [ ] **T17** S4 `enrich`: `model_input`, lengths, payload, bursts — *M* — deps: T16
- [ ] **T18** S4 language detection: payload-stripped vs naive — *S* — deps: T17

### Checkpoint D: Splitter inputs complete
- [ ] S0→S4 runs with full row accounting
- [ ] `report --stage dedup` and `report --stage enrich` reproduce §1.3 and §S4
- [ ] `model_input` is built by the shared function the model service imports
- [ ] Human review before Phase 4

## Phase 4: Splits

- [ ] **T19** S7 `split`: temporal, gapped, cluster-aware — *M* — deps: T18
- [ ] **T20** Random diagnostic split and leakage comparison — *M* — deps: T19
- [ ] **T21** Per-service counts, `measurable` flag, split integrity gates — *M* — deps: T20

### Checkpoint E: A dataset that can be trained on
- [ ] Three splits produced; every §10 split gate passes
- [ ] Leakage delta (6.2% vs 3.7%) and org overlap (184/184) reported
- [ ] Human review; resolve open question 5

## Phase 5: LLM stages

- [ ] **T22** OpenRouter client: structured output, retry, call ceiling — *M* — deps: T4
- [ ] **T23** Content-addressed LLM cache and `--dry-run` cost estimate — *M* — deps: T22
- [ ] **T24** S5 prompt v1, prompt hashing, evidence validation — *M* — deps: T23, T7
- [ ] **T25** S5 Pass A: blind measurement sample with bootstrap CI — *M* — deps: T24
- [ ] **T26** S5 Pass B: triage ranking and `review/labels.jsonl` — *M* — deps: T25
- [ ] **T27** S6 egress gating and stratified sampling — *M* — deps: T23, T12
- [ ] **T28** S6 audit: span verification and `review/redaction.jsonl` — *M* — deps: T27, T15

### Checkpoint F: LLM stages, cached and gated
- [ ] A warm-cache re-run makes zero network calls and reproduces outputs
- [ ] `--dry-run` touches no network
- [ ] No code path lets an LLM write a label or edit text
- [ ] S6's skip / hard-fail / run matrix behaves exactly as §5.3 specifies
- [ ] Human review; resolve open question 3

## Phase 6: Freeze, manifest, datasheet

- [ ] **T29** S8 `freeze`: output schema and split artifacts — *M* — deps: T21, T26
- [ ] **T30** `manifest.json` assembly and the `verify` command — *M* — deps: T29
- [ ] **T31** `annotation_queue.jsonl`: blind, iteratively stratified — *M* — deps: T29
- [ ] **T32** `datasheet.md` generation — *M* — deps: T30, T31
- [ ] **T33** Integration test: full pipeline on `slice_200.csv`, reproducible — *M* — deps: T32

### Checkpoint G: Complete
- [ ] All ten §12 success criteria mechanically checked
- [ ] `verify --version v1` passes on two consecutive builds
- [ ] Every `[measured]` spec claim reproducible via a `report` subcommand
- [ ] Human review before Phase 7

## Phase 7: Ship readiness

- [ ] **T34** CI workflow running every build-failing gate — *S* — deps: T33
- [ ] **T35** README quickstart and the committed manifest — *S* — deps: T34

### Final
- [ ] A teammate who has not read the spec reproduces the documented hashes from the README alone
