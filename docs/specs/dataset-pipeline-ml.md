# Dataset construction pipeline: `tickets_export.csv` → versioned, training-ready multi-label dataset

Status: specification (not implemented). **No pipeline code exists and none is created by this document.**
Author: ml-researcher
Date: 2026-08-18
Audience: implementing developer. `system-architect` owns §14 integration; `dataset-provider` owns the raw export.
Companions: [`ticket-services-classifier.md`](./ticket-services-classifier.md) (model spec — §2, §4.6, §6 are load-bearing here),
[`dataset-construction-runbook.md`](./dataset-construction-runbook.md) (provenance Cases A/B/C/D, gold set, order of operations),
[`../proposals/ticket-services-classifier.md`](../proposals/ticket-services-classifier.md) (§3 label-provenance trap, §5.3 one shared preprocessing code path),
[`dataset-pipeline-architecture.md`](./dataset-pipeline-architecture.md) (`system-architect`'s design for the same
pipeline — stage graph, `ticketprep` package, artifact contracts, trust zones). **That document owns the software;
this one owns what the software must do to the data and how we prove the choices were right.** Where the two touch,
its ordering findings (normalise → redact → dedup, language after redaction) agree with §4.5 here, which is a useful
independent confirmation; §14.2 flags the one place where its design and this spec's correctness requirement must be
reconciled.

---

## 0. Scope, repo state, and how to read the numbers

**Existing assets [measured]:** `/home/user/bert-poc` contains `data/raw/tickets_export.csv` (3.0 MB, 5,013 rows,
10 columns) plus its `README.md`, and four documents under `docs/`. There is **no pipeline code, no notebook, no
`labels.json`, no snapshot, no dedup index, no redaction rules, no annotation tooling** — nothing to reuse, so
everything below is specified from scratch. Reuse is nevertheless the rule where it exists: the split policy,
the metric definitions and the `labels.json` contract come from `ticket-services-classifier.md` and are *referenced*,
not re-invented.

**Measurement discipline.** Every `[measured]` figure in this document was computed by me on
`data/raw/tickets_export.csv` with throwaway scripts listed in §15. Tokenisation figures use the real
[`FacebookAI/xlm-roberta-base` tokenizer](https://huggingface.co/FacebookAI/xlm-roberta-base) (`tokenizer.json`,
`tokenizers` 0.23.1), which is the default checkpoint in the model spec §4.2. `[estimate]` means arithmetic with the
working shown. Where I re-measured a claim from the corpus README, it reproduced within detector tolerance in every
case I checked (9 mojibake rows, 240→231 line-numbered rows, 173→168 self-redacted rows, 560 near-duplicate pairs
over 168 rows at J≥0.80, 99-row alert clique, 232 rows under 80 characters, per-service counts, cardinality
histogram). The two places my number differs materially are flagged in place: the refile clusters (README 72 rows;
my keyword probe finds 10 — my detector, not their count, is the weak one) and repeated boilerplate (README
135 sentences / 47.9% of instances; mine 238 / 40.5% — a sentence-splitter difference). **Treat the README as
reliable; it earned that.**

**The one thing this document exists to decide.** The raw text carries bank details, PII, company identifiers,
secrets and machine-generated payloads. The question is not "should we redact" (compliance answers that) but
**what transform maximises downstream `R@P≥0.90` per unit of privacy risk and per token of budget** — and that is an
empirical question with a measurable answer. §4 specifies the transform, §5 specifies the experiment that chooses
between the variants. If you read only one section, read §5.

**Framing correction, stated explicitly.** The task as given was "turn the CSV into a training-ready dataset". Two
things changed my framing while reading the corpus and its README:

1. **The dataset produced from this file cannot be used to make any quality claim.** Labels are author-assigned by a
   generator; there is no inter-annotator agreement and no human ground truth. What this pipeline delivers is a
   *pipeline*, exercised at realistic volume, plus a **v1 dataset fit for engineering and mechanism validation only**
   (§9.2 states the permitted and forbidden uses; §11 splits the acceptance gates into what this file can and cannot
   certify). Calling it "training-ready" without that qualifier would be the most expensive kind of quiet mistake.
2. **Redaction on this corpus measures better than it will in production, by construction.** [measured] 92% of
   secret-shaped spans have an obvious-fake marker (`EXAMPLE`, `DO-NOT-USE`, `TEST`) within the span plus 120
   characters, and **0 of 89 ИНН instances have a valid check digit**. Any detector tuned or scored on this file
   inherits that bias. §4.7 makes a hand-labelled redaction eval set a blocking prerequisite rather than a nicety.

---

## 1. Problem statement

**Deliverable:** a deterministic, versioned function `raw export → dataset snapshot`, where the snapshot is
(a) a set of records `{ticket_id, org_id, created_at, text_in, labels, provenance, split, dedup_group, lang, flags}`,
(b) the artifacts that make it reproducible (`labels.json`, `preprocess.json`, `dedup_manifest`, `dataset_card.md`),
and (c) a measured statement about how each stage affects the model.

**Inputs:** `id`, `organization_id`, `title`, `description`, `priority`, `services`, `labels`, `flags`,
`created_at`, `updated_at`.
**Output text contract:** one string per ticket, identical in training and serving — model spec §4.6:
`"[priority: {p}] {title}\n{normalised description}"`, optionally with the E5 `query: ` prefix.
**Output label contract:** a 20-dimensional multi-hot vector over the closed vocabulary in `labels.json` (§6).

**What "good enough" means for the pipeline** (product terms): the pipeline is good enough when
(i) a ticket can be reprocessed at serving time by the *same* code and produce a byte-identical string,
(ii) no evaluation number it produces is inflated by leakage the pipeline could have removed,
(iii) no direct identifier survives into any stored artifact above the leak-rate gate in §4.7, and
(iv) the redaction policy it applies has been *chosen by measurement* (§5), not by taste.

**Non-goals.** Choosing the model, tuning thresholds, deciding where inference runs, designing the audit schema
(that is `system-architect`'s, per proposal §5.4), and any claim about production quality from this file.

---

## 2. The input file: what it is and what it is not

Structural statistics from the corpus README are the ones that transfer to production; lexical ones are not
(compositional surface realisation). I re-measured the structural ones I depend on.

| Property | Value | Source |
|---|---|---|
| Rows / columns | 5,013 / 10 | [measured] |
| `created_at` span | 2025-06-10 → 2026-08-09, 409 distinct days, median 13 tickets/day | [measured] |
| Organisations | 210, Zipf-ish (max 364, 236, 159, 132, 130 tickets) | [measured] |
| Labelled rows / positives | 4,622 / 8,771; cardinality 0:391, 1:1,569, 2:1,988, 3:1,034, 4:31 | [measured] |
| Distinct `services` surface strings | 59 → 20 after normalisation; 39 surface forms are dirty | [measured] |
| Rows with embedded newline (payload proxy) | 1,771 (35.3%) | [measured] |
| Exact duplicate descriptions / title+description | 6 / 0 | [measured] |
| XLM-R token length of the model input | mean 118.6, p50 79, p75 168, p90 240, p95 314, p99 417, max 762 | [measured] |
| Rows with a payload vs without, mean tokens | 214.4 vs 66.3 | [measured] |

Per-service positive counts [measured] — the tail is the story:

| billing | console-ui | auth | api-gateway | compute | access-control | object-storage | monitoring | subscriptions | networking |
|---|---|---|---|---|---|---|---|---|---|
| 851 | 793 | 762 | 751 | 715 | 667 | 645 | 590 | 560 | 452 |

| notifications | managed-postgres | logging | integrations | backups | managed-redis | dns | cdn | message-queue | terraform-provider |
|---|---|---|---|---|---|---|---|---|---|
| 442 | 404 | 324 | 321 | 271 | **98** | **41** | **38** | **31** | **15** |

Five services sit at or below the "cannot be learned or evaluated reliably" line of model spec §2.8. This is
deliberate in the fixture and is also realistic. §8 quantifies exactly how badly it interacts with the split policy.

**What the file is not:** not a gold set, not provenance-reconstructable (`ticket_messages` is absent, so runbook §1–§3
Cases A/B/C/D cannot be executed here at all), not a source of any accuracy claim. The `services` column here is
*generator-assigned*, which is neither Case A nor Case C — it is a fifth case the runbook does not have, and §10
names it **Case G**.

---

## 3. Pipeline stage order (the contract)

Stages are numbered because **the order is load-bearing and I have measurements showing what breaks when it is
wrong** (§4.5). Stages 2–8 are the "shared code path" that proposal §5.3 requires to run identically at train and
serve time; stages 9–13 are dataset-build-only and must **not** exist in the serving path.

| # | Stage | Train | Serve | Why here |
|---|---|---|---|---|
| 1 | RFC4180 CSV parse into records | ✓ | — | 1,771 rows have embedded newlines and 814 embedded quotes: **never** line-split or regex the raw file |
| 2 | Unicode NFC + control-char strip (keep `\n`, `\t`) | ✓ | ✓ | Block detectors need line structure |
| 3 | **Mojibake repair**, document-scoped | ✓ | ✓ | §4.5 R1 — regexes cannot see through it |
| 4 | Paste-shell normalisation: line-number gutters, `>` quote markers, soft-wrap rejoin | ✓ | ✓ | §4.5 R3 — detectors are line-anchored |
| 5 | **Value-span detection & transform** (typed placeholders) | ✓ | ✓ | §4.2 |
| 6 | **Block segmentation & type collapse** (policy-dependent) | ✓ | ✓ | §4.2, granularity chosen by §5 |
| 7 | Language identification on the **prose channel only** | ✓ | ✓ | §4.5 R5 — evaluation slice depends on it |
| 8 | Text template + tokenise + head/tail truncation | ✓ | ✓ | Model spec §4.6 |
| 9 | Label normalisation → multi-hot over `labels.json` | ✓ | — | §6 |
| 10 | Dedup (exact → MinHash → embedding) | ✓ | — | §7 |
| 11 | Split assignment (temporal, gapped, burst-aware) | ✓ | — | §8 |
| 12 | Label QA funnel | ✓ | — | §9 |
| 13 | Snapshot freeze, hashing, dataset card, gates | ✓ | — | §11 |

**Two rules with no exceptions.**
**R-A: redaction precedes tokenisation.** Truncating first and redacting after leaves identifiers in the discarded
tail (harmless) *and* changes which tokens survive (harmful, and irreproducible).
**R-B: one implementation.** Stages 2–8 ship as a single versioned artifact (`preprocess.json` + the code that
consumes it) with a `preprocess_version` recorded in every snapshot, every metrics report and every prediction.
A mismatch between the training-time and serving-time version is a hard error, not a warning.

---

## 4. Content normalisation and redaction

### 4.1 The three questions, asked per span

For every span category, three independent answers decide the transform:

1. **Does the span carry class signal?** Measured here as *lift*: `P(service | detector fires) ÷ P(service)`.
2. **Is it a privacy or compliance liability?** Direct identifier (name, phone, email, PAN, СНИЛС, passport),
   quasi-identifier (org name, ИНН, address, hostname), or credential (key, token, PEM). 152-FZ constrains where
   the *raw* text may live (model spec §2.9); the pipeline's job is to make the derived corpus carry as little as
   possible.
3. **What does it cost in tokens?** Every span redacted is budget returned to real content — measured in §4.6.

The answer is almost never "delete". **The most common right answer is: destroy the value, keep the type.** The
measurements below show why: payload *kind* is one of the strongest features in this corpus, and it survives
value-destruction intact.

### 4.2 Span taxonomy and prescribed transforms

Signal column = the strongest measured lift for a service when the detector fires (`n` = rows firing) [measured].
Liability: **D**irect identifier, **Q**uasi-identifier, **C**redential, **–** none.
Transform: `KEEP` (untouched) · `TYPE` (typed placeholder, value destroyed) · `TYPE+ATTR` (typed placeholder that
retains one low-cardinality attribute) · `DROP` (span removed entirely).

> **How to read the lift numbers.** Lift is `P(service | detector fires) ÷ P(service)`, computed with my prototype
> detectors over all 5,013 rows. It is **correlational and corpus-specific**: in this fixture payloads were attached
> to match the ticket's topic by construction (corpus README), so lift here is an upper bound on how cleanly the
> association will appear in production, and lifts computed on ≤50 rows (`REDIS_CLI` n=9, `DNS_ZONE` n=10) have wide
> uncertainty. What lift is being used for is **ranking which span types deserve a retained type or attribute** —
> it is a design input, not a claim about production. §5 is what decides.

#### Value-level spans (stage 5)

| # | Span | n rows | Class signal [measured] | Liab. | Transform | Justification |
|---|---|---|---|---|---|---|
| 1 | E-mail address | 468 | notifications ×2.1, integrations ×1.7 | D | `TYPE` → `<EMAIL>` | Signal is carried by the *context* ("письмо не пришло"), not the address. Never keep the domain: it is a customer identifier |
| 2 | Phone number | 266 | notifications ×1.7 | D | `TYPE` → `<PHONE>` | Same. No per-country distinction needed |
| 3 | Personal name / DOB / postal address | ~several hundred (README) | none isolable | D | `TYPE` → `<PERSON>`, `<DOB>`, `<ADDR>` | Requires NER (§4.4 L3); the highest-recall-risk category |
| 4 | Passport / СНИЛС / driving licence | 27 spans | none (`GOV_ID` lift ≤ ×2.4, n=14, noise) | D | `TYPE` → `<GOV_ID>` | Zero signal, maximum liability. No attribute retained |
| 5 | Card PAN (full or masked) | 128 | **billing ×4.6**, subscriptions ×2.4 | D | `TYPE` → `<PAN>` | The *presence* of card material is a strong billing cue; the digits are worthless and toxic. Masked forms map to the **same** placeholder (§4.5 R4) |
| 6 | RU bank details (р/с, к/с, БИК), IBAN, SWIFT | 36 | **billing ×5.1**, subscriptions ×4.5 | D/Q | `TYPE` → `<BANK_ACCT>` | Strongest single billing cue measured at value level. Keep the type, destroy the number |
| 7 | ИНН / КПП / ОГРН / legal entity name | 93 | **billing ×3.9**, subscriptions ×2.5 | Q | `TYPE` → `<ORG_ID>`, `<ORG_NAME>` | Quasi-identifier: an ИНН re-identifies a customer exactly. Type retained because "accounting documents" cues billing |
| 8 | Money amount | 102 | **billing ×4.3** | – | `TYPE+ATTR` → `<MONEY:RUB>` | No liability, real signal. Currency is the attribute worth keeping; the magnitude is not |
| 9 | API key / bearer token / OAuth secret / webhook secret | 200 (all secret shapes) | object-storage ×1.8, auth ×1.5 | **C** | `TYPE` → `<API_KEY>`, `<BEARER>`, `<WEBHOOK_SECRET>` | Credentials must never be stored, even in a "raw" snapshot. Also emit a counter (§4.8) |
| 10 | JWT | 9 | – | **C** | `TYPE` → `<JWT>` | 1,605 tokens for 9 spans [measured] — 178 tokens each. Pure budget waste |
| 11 | PEM / SSH private key block | 2 | – | **C** | `TYPE` → `<PEM_KEY>` | Same; also the highest-severity leak class |
| 12 | DB connection URI with password | 98 | **managed-postgres ×2.5** | **C** | `TYPE+ATTR` → `<DB_URI:postgres>` | The *scheme* is the signal (`postgres://` vs `redis://` vs `amqp://`) and carries no secret. Keep exactly that |
| 13 | URL | 542 | monitoring ×2.4, api-gateway ×1.7 | Q | `TYPE+ATTR` → `<URL:host_role>` | 10,268 tokens [measured]. Query strings carry tokens; paths carry customer ids. Retain only a **host role** from a closed allowlist (§4.3) |
| 14 | Hostname / FQDN | 1,138 | **notifications ×7.3** (`smtp*`/`mail*`), **cdn ×10.8** (`cdn*`/`edge*`) | Q | `TYPE+ATTR` → `<HOST:smtp>`, `<HOST:cdn>`, `<HOST>` | The single best argument in this document for granularity: a flat `<HOST>` throws away a ×7–×11 cue. Role vocabulary is closed and product-owned |
| 15 | IPv4 / IPv6 / CIDR | 366 / 43 | networking ×2.1, cdn ×3.2 | Q | `TYPE+ATTR` → `<IP:private>` / `<IP:public>` | Private-vs-public is a networking cue and is not identifying. The address is |
| 16 | UUID, request/trace/correlation/idempotency id | 383 / 142 | object-storage ×1.7 | Q | `TYPE` → `<ID>` | 18,505 tokens combined (3.1% of the corpus) [measured] for near-zero signal. Highest budget-per-signal ratio in the file |
| 17 | k8s pod/replicaset name, bucket, queue, namespace | 131 | message-queue ×6.5 (`INFRA_IDS`, n=124) | Q | `TYPE+ATTR` → `<POD>`, `<BUCKET>`, `<QUEUE>` | The *kind of resource* is the signal; the name is a customer identifier |
| 18 | Timestamp (ISO, syslog, CLF, bare clock) | 584+ | – | – | `TYPE` → `<TS>` | Zero signal. **Must be matched first** (§4.5 R2) |
| 19 | Hex hash / base64 blob | 100 / 232 | – | Q | `TYPE` → `<HASH>`, `<B64>` | 13,235 tokens [measured]. Base64 may *contain* credentials — never keep |
| 20 | File path | 298 | object-storage ×2.3 (`*.py`, n=122) | Q | `TYPE+ATTR` → `<PATH:py>`, `<PATH:tf>`, `<PATH:sql>` | Extension is cheap and non-identifying; directory names leak project names. **Honest caveat [measured]: `.tf`, `.sql` and `.yml` paths fire 0 times in this file** — the IaC/DB signal lives in the *block* detectors (rows 24–27), not in path extensions. Keep the attribute, expect it to matter mainly for `.py`/`.go`/`.js` here |
| 21 | Telegram handle / @mention | 385 | notifications ×2.2 | D | `TYPE` → `<HANDLE>` | – |
| 22 | Invoice / акт / договор / ticket number | 102 (`MONEY`-adjacent) | billing (co-fires) | Q | `TYPE` → `<DOC_ID>` | – |
| 23 | Customer self-redaction artifact (`sk_test_51H***REDACTED***`, `пароль: ******`) | 168 | – | – | `TYPE` (same target as the unredacted form) | §4.5 R4: `sk_test_***` must become `<API_KEY>`, not survive as a distinctive literal |

#### Block-level spans (stage 6)

Lift measured over rows where the block detector fires [measured]; this is the table that decides the granularity
question, because it shows the *kind* of a payload is worth far more than its contents.

| # | Block kind | n rows | Class signal [measured] | Liab. | Transform | Justification |
|---|---|---|---|---|---|---|
| 24 | PostgreSQL error / statement pair | 53 | **managed-postgres ×5.4** | Q | `TYPE+ATTR` → `<SQL_ERROR: deadlock detected>` | Retain the **error string**, drop table/column values, host, pid, timing. The error string is the whole signal |
| 25 | SQL query / `EXPLAIN` output | 35 | **managed-postgres ×8.2**, logging ×5.3 | Q | `TYPE+ATTR` → `<SQL_QUERY:EXPLAIN>` | Highest lift in the file. Literals in `WHERE` clauses are customer data — drop them, keep the verb and plan node kinds |
| 26 | `docker-compose` / service YAML | 26 | **managed-postgres ×7.6** | Q | `TYPE+ATTR` → `<COMPOSE: image=postgres>` | Image names are the signal; env values are secrets |
| 27 | Terraform HCL / `terraform apply` error | 37 | object-storage ×4.4, backups ×2.5 | Q | `TYPE+ATTR` → `<TF_HCL: resource_type>` | Note honestly: it does **not** top out on `terraform-provider` in this file (n=15 there; §8), it cues the *resource being managed*. Keep the resource type token |
| 28 | `redis-cli` output / Redis metrics | 9 | **managed-redis ×51.2** | – | `TYPE+ATTR` → `<REDIS_CLI: evicted_keys>` | Largest lift measured anywhere. Deleting this block would be the single most damaging redaction choice available |
| 29 | DNS zone file / `dig` output | 10 | **networking ×10.0** | Q | `TYPE+ATTR` → `<DNS_ZONE:CNAME>` | Record type retained, names dropped |
| 30 | postfix/SMTP log | 45 | **notifications ×7.1**, integrations ×7.3 | D (recipients) | `TYPE+ATTR` → `<SMTP_LOG: status=bounced>` | Status is the signal; addresses are direct identifiers |
| 31 | Payment / 3-DS decline dump | 49 | **billing ×4.7**, subscriptions ×3.7 | D | `TYPE+ATTR` → `<PAYMENT_DECLINE: do_not_honor>` | Decline reason retained, PAN/RRN/auth code destroyed. This is the canonical "keep the kind, drop the values" case |
| 32 | Bank requisites block | 36 | **billing ×5.1** | D/Q | `TYPE` → `<BANK_DETAILS>` | No attribute is safe to retain here |
| 33 | Company identifier block | 107 | **billing ×4.0** | Q | `TYPE` → `<COMPANY_IDS>` | – |
| 34 | k8s events / `kubectl` output | 51 | **networking ×5.2**, compute ×2.6 | Q | `TYPE+ATTR` → `<K8S_EVENT: OOMKilling>` | Reason retained |
| 35 | nginx access/error line | 177 | console-ui ×2.2, networking ×2.4, cdn ×3.0 | Q | `TYPE+ATTR` → `<NGINX_LOG: 503 upstream_timeout>` | **Status code is the signal** and is not identifying. Do not drop it |
| 36 | nginx / web-server config | 28 | **api-gateway ×3.8**, networking ×3.6 | Q | `TYPE+ATTR` → `<NGINX_CONF: proxy_pass>` | Directive names retained |
| 37 | HTTP request/response dump | 80 | api-gateway ×2.2, integrations ×2.1 | **C** (headers) | `TYPE+ATTR` → `<HTTP_DUMP: 403>` | Header **names** may be retained, **values never**. `Authorization`/`Cookie`/`X-Signature` values are credentials |
| 38 | `curl` invocation | 94 | **api-gateway ×2.6** | **C** | `TYPE+ATTR` → `<CURL: POST>` | – |
| 39 | Browser console error | 63 | **console-ui ×3.9**, auth ×2.1 | – | `TYPE+ATTR` → `<BROWSER_ERROR: net::ERR_...>` | Error symbol retained |
| 40 | Python traceback / Java stack / Go panic | 123 / 86 / 99 | object-storage ×2.3, integrations ×2.5, managed-redis ×2.6 | Q | `TYPE+ATTR` → `<PY_TRACEBACK: KeyError>` | Keep exception class + top frame **module**; drop file paths, line numbers, addresses, locals |
| 41 | systemd / journalctl lines | 49 | backups ×4.2 | Q | `TYPE+ATTR` → `<SYSLOG: unit=..., status=...>` | Unit name is a project identifier — retain only a normalised unit *class* if the allowlist covers it, else drop |
| 42 | `.env` paste | 57 | notifications ×2.6 | **C** | `TYPE+ATTR` → `<ENV_FILE: KEYS_ONLY>` | Keys are informative, values are secrets by definition |
| 43 | JSON body | 211 | api-gateway ×1.6 | Q | `TYPE+ATTR` → `<JSON: keys>` | Retain top-level key names only (bounded to ~10); values are customer data |
| 44 | Quoted/forwarded e-mail thread, signature block | 176 | notifications ×1.7 (weak) | **D, dense** | **`DROP`** → `<QUOTED_MAIL>` | The only category I recommend deleting outright: weakest measured signal of any block kind, highest density of names/addresses/phones, and it is where the corpus hides whole customer identities |
| 45 | Infra id list (`project_id:`, `cluster_id:`…) | 124 | message-queue ×6.5, networking ×2.1 | Q | `TYPE` → `<INFRA_IDS>` | The list's *presence* is a weak cue; the ids are pure identifiers |
| 46 | Cron output, arbitrary code fragments | 66 | backups ×3.1 | Q | `TYPE+ATTR` → `<CODE:py>` | Language attribute only |

**Where redaction destroys signal — stated plainly.** Rows 24–39 are cases where a naive "strip the pasted block"
policy deletes the strongest available evidence: a `psql` `ERROR:` line (×5.4 for `managed-postgres`), an `EXPLAIN`
plan (×8.2), `redis-cli` output (×51.2 for `managed-redis` — a service with only 98 positives, i.e. one that needs
every cue it can get), a `dig`/zone dump (×10.0), an SMTP status line (×7.1), a payment decline dump (×4.7 billing),
an nginx 503 line (×2.2–3.0). **In all of these the prescribed transform is `TYPE+ATTR`: the payload's kind and one
diagnostic attribute survive; every value dies.**

### 4.3 Placeholder granularity is a tunable, and it is the main ablation axis

Five granularity levels, from most to least aggressive. The literature on pseudonymisation strategies uses exactly
this ladder — deletion / uniform placeholder / category placeholder / unique-per-entity placeholder / realistic
surrogate — and reports *no significant downstream loss* for the placeholder strategies on a sensitive-text
classification task
([Cloaked Classifiers, arXiv 2406.17875](https://arxiv.org/html/2406.17875v1): English macro-F1 64.63±2.0 original
vs 65.46±1.0 pseudonymised), consistent with clinical findings that pre-training on automatically de-identified text
costs nothing downstream
([Vakili et al., LREC 2022](https://aclanthology.org/2022.lrec-1.451.pdf);
[end-to-end pseudonymisation of clinical BERT, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11197357/)).
**That is prior evidence, not a result on this corpus** — it justifies expecting the cost to be small, and it is
exactly why the decision must still be measured here (§5).

| Level | Definition | Example | Expectation |
|---|---|---|---|
| **G0** | No redaction | `4111 1111 1111 1111` | Reference arm only. Not shippable (152-FZ, §4.8) |
| **G1** | Uniform placeholder | `<REDACTED>` | Destroys the type signal. [measured] also *saves fewer tokens* than G2 (8.5% vs 9.9%) because `<REDACTED>` is 5 subwords |
| **G2** | Typed placeholder | `<PAN>` | **Recommended default for values** |
| **G3** | Typed + one attribute | `<PAYMENT_DECLINE: do_not_honor>`, `<HOST:smtp>` | **Recommended default for blocks.** The attribute vocabulary must be a closed allowlist, or it becomes a re-identification channel |
| **G4** | Realistic surrogate (consistent fake values) | `4111 1111 1111 1111` → another fake PAN | Rejected for v1: it produces text that *looks* unredacted, so a leak is undetectable by grepping for placeholders, and it makes the leak-audit gate (§4.7) unimplementable |

**Two implementation facts that change the arithmetic** [measured]:

- Placeholders are **not free**: under XLM-R, `<EMAIL>` is 4 tokens, `<UUID>` 4, `<PY_TRACEBACK>` 9,
  `<PAYMENT_DUMP>` 8, `<REDACTED>` 5. With 7,337 value-placeholder instances across the corpus, that is real budget.
- **Adding placeholders as tokenizer special tokens** makes each one 1 token and recovers a further 3.6% (values
  only) to 5.6% (with block collapse) of the corpus token budget — total saving vs raw goes 9.9% → **13.2%** and
  25.9% → **30.0%**. Model spec §4.6 already permits added special tokens; this measurement is the reason to do it,
  and the reason the embedding resize must happen **before** any vocabulary trimming.

### 4.4 Detector architecture: three layers, in this order

**L1 — deterministic patterns + validators.** Regex/pattern detectors for rows 1–23 and 35–46 of §4.2, each with an
explicit precision target, plus arithmetic validators: Luhn for PAN, mod-97 for IBAN, the ИНН/ОГРН/СНИЛС check
digits, base64 length%4 and alphabet, JWT three-segment base64url decode, PEM BEGIN/END pairing.

> **Validators score, they do not gate.** [measured] on this file: **0 of 89 ИНН spans have a valid check digit**,
> only **2 of 35 distinct 13–19-digit runs pass Luhn** (both the published test PANs). A check-digit-gated detector
> would redact **nothing** here and would look perfect in production — the classic evaluation inversion. Use the
> validator to raise confidence (`valid → redact silently`; `invalid but shape+context match → redact and count as
> "shape-only"`), never as a precondition for redaction. Report the shape-only rate; in production it should fall.

**L2 — structural block detectors.** Line classifiers → contiguous-run grouping → block typing (rows 24–46). This
layer is what makes `TYPE+ATTR` possible. It must be line-anchored (`^`/`$` with `re.M`) and must tolerate blocks
that are truncated mid-line: [measured] 1,233 descriptions end without terminal punctuation after a newline, and the
corpus README documents 155 deliberately truncated payloads. Detectors that require a closing delimiter (PEM END,
closing brace, final stack frame) must have a **line-count-bounded fallback**.

**L3 — NER for the residue (names, addresses, org names).** L1/L2 cannot find "Андрей Кузнецов, главный бухгалтер".
Options:

| Option | Fit | Cost | Note |
|---|---|---|---|
| [Microsoft Presidio](https://microsoft.github.io/presidio/analyzer/adding_recognizers/) as the orchestration layer + custom `PatternRecognizer`s | Good | low | Ships predominantly English/US recognisers ([default set](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/conf/default_recognizers.yaml)); **ИНН/ОГРН/СНИЛС/р-с/passport must be written as custom recognisers** — assume none of them exist out of the box |
| [Natasha/Slovnet](https://github.com/natasha/slovnet) Russian NER (PER/LOC/ORG) | Good for RU | ~30 MB, CPU-fast (~25 docs/s reported) | Reported by the authors as 1–2% below BERT SOTA at 1/60 the size — the right trade for a preprocessing stage that must also run at serve time |
| A Russian BERT NER checkpoint (NEREL/RuNNE-style) | Highest recall | 100s of MB + latency in the serving path | Only if the §4.7 recall target is missed. **This is a serving-latency decision** — hand the budget to `system-architect`, do not silently add 50 ms |
| English NER (spaCy/Presidio default) for the EN slice | Required | low | 28.5% of the corpus is English |

**L3 runs last and is allowed to be low-precision**: over-redacting a common noun costs a token; under-redacting a
name costs a person. Where L3 disagrees with L1 on a span, **the more aggressive transform wins**.

**Layer conflict resolution:** produce all candidate spans with `(start, end, type, layer, confidence)`, then resolve
by a single deterministic pass: longest span wins; ties broken by layer (L1 > L2-attribute > L3); overlaps never
partially applied. This must be a pure function — the same input must always produce the same output, or the
snapshot hash is meaningless.

### 4.5 Ordering constraints, each with the measurement that justifies it

**R1 — mojibake repair before every regex, at document scope.** [measured] 9 rows are Windows-1251 text decoded as
Latin-1 (`ÈÍÍ 6470590803`, `Êòî íà ñâÿçè`), repairable with `bytes(latin-1) → decode(cp1251)`. Scope matters:
repairing only the *runs* that look mojibake'd (≥4 consecutive high-Latin characters) recovers **0** of the 2 ИНН +
2 КПП spans hidden in those rows, because the give-away run and the sensitive token are on different lines and
`ÈÍÍ` is only 3 characters. **Document-scoped repair recovers all 4** and provably does not damage clean Cyrillic
(clean Cyrillic is outside U+0080–U+00FF). Detection rule: repair a document iff it contains ≥1 run of ≥4 characters
in U+00C0–U+00FF **and** the repaired text has a higher Cyrillic-character ratio than the original.

**R2 — timestamps before address-shaped patterns.** [measured] a standard IPv6 regex fires 903 times on this corpus
and **67.9% of those matches are clock times** (`10:06:28`). After a timestamp-first pass, it fires **43 times** —
exactly the RFC 3849 `2001:db8::` addresses. That is a 95% false-positive reduction from ordering alone, and each of
those FPs would have rewritten a log timestamp as `<IP>`, corrupting the block detectors downstream.

**R3 — paste-shell normalisation before block detection.** [measured] 231 rows carry a line-number gutter
(`  12 | ...`) and 176 carry `>`-quoted lines. Both break `^`-anchored line detectors. Strip the gutter and the
quote prefix into a per-line attribute *before* stage 6, and re-attach nothing.

**R4 — partial self-redaction must converge to the same placeholder.** [measured] 168 rows contain customer-applied
masking (`***`, `REDACTED`, `X{4,}`, `пароль: ******`). `sk_test_51H***REDACTED***` must become `<API_KEY>`, and
`4111 11** **** 1111` must become `<PAN>` — the same targets as their unmasked forms. If they do not converge, the
model learns "customer masked it themselves" as a feature, and worse, the masked form survives as a distinctive
literal in the snapshot. Concretely: masked-form patterns are part of the same detector, not a separate one, and
`***`-only spans with no recoverable type map to `<REDACTED>`.

**R5 — language identification after payload segmentation, on the prose channel.** [measured] with a Cyrillic-ratio
rule, 908 rows fall in the ambiguous 0.30–0.70 band on raw text; **455 of them move above 0.70 (i.e. become
unambiguously Russian) once payload lines are removed**. Pasted ASCII logs drag Russian tickets into a "mixed"
bucket. Since model spec §6.3 makes RU/EN/mixed a *gated* evaluation slice, computing the language on raw text
mislabels roughly 9% of the corpus's slice membership. Compute it on prose only, and store `lang` and
`lang_confidence` as snapshot columns.

**R6 — bounded matching, always.** Descriptions reach 1,691 characters and 155 payloads are truncated mid-line, so
open-ended patterns can be forced to scan the whole document. [measured] the specific PEM case is *not* explosive in
CPython (0.034 ms/doc unbounded vs 0.022 ms bounded on a 2 KB unterminated PEM), so I will not overstate it — but
[measured] a nested-quantifier pattern (`^(?:[A-Za-z0-9]+\s?)+$`) costs 0.500 ms on an **80-character**
non-matching line versus 0.002 ms for a bounded character class, a 250× gap that grows exponentially with line
length. Rules: no nested quantifiers; cap every `.*`/`[\s\S]*` with an explicit `{0,N}`; match line-scoped where
possible; enforce a **per-document time budget (proposal: 20 ms)** and, on exceeding it, **quarantine the row** —
never fall through to emitting raw text.

**R7 — truncation last.** Head+tail truncation (model spec §4.6) operates on the redacted, template-assembled
string. Never before stage 5.

### 4.6 Token budget: what each policy actually buys [measured]

Input string `"[priority: p] title\n<description under policy>"`, XLM-R tokenizer, all 5,013 rows.

| Policy | Total tokens | Δ vs raw | p50 | p90 | p99 | max | ≤128 tok | ≤256 tok | tokens lost to truncation @256 |
|---|---|---|---|---|---|---|---|---|---|
| **V0** raw | 594,546 | — | 79 | 240 | 417 | 762 | 69.4% | 91.3% | 36,859 |
| **V1** typed values (G2/G3) | 535,627 | **−9.9%** | 78 | 203 | 347 | 604 | 71.7% | 95.0% | 14,457 |
| **V1c** uniform `<REDACTED>` (G1) | 543,812 | −8.5% | 78 | 208 | 357 | 627 | 71.2% | 94.5% | 16,557 |
| **V2** V1 + block type-collapse (G3) | 440,668 | **−25.9%** | 74 | 157 | 231 | 345 | 82.7% | 99.6% | 547 |
| **V3** V1 + block deletion | 423,419 | −28.8% | 74 | 145 | 211 | 334 | 86.2% | 99.8% | 278 |
| V1 with placeholders as special tokens | 516,139 | −13.2% | — | — | — | — | — | — | — |
| V2 with placeholders as special tokens | 415,944 | **−30.0%** | — | — | — | — | — | — | — |

On payload-carrying rows alone (n=1,771), mean tokens go 214.4 (V0) → 182.9 (V1) → 131.1 (V2) → 121.6 (V3)
[measured].

Readings that matter:

- The model spec's "**10–30% of the token budget**" claim (§4.6) is **confirmed on this corpus**: 9.9% for
  value-level redaction alone, 30.0% for type-collapse with special tokens.
- **V2 changes what `max_len` means.** At 256 tokens V0 truncates away 36,859 tokens of real content; V2 truncates
  away 547. Equivalently, V2 at `max_len=192` covers more content than V0 at 256 — and per model spec §4.5,
  256→128 tokens roughly halves CPU latency. **The redaction policy is therefore also a latency lever**, and that
  belongs in the §5 decision alongside quality.
- G1 (uniform `<REDACTED>`) is dominated: fewer tokens saved *and* less signal than G2. Do not ship it; keep it as
  an ablation arm only, because it is the policy people reach for by default.

### 4.7 Proving the detectors work without fooling ourselves

**The problem, quantified.** [measured] every one of the 200 rows containing a secret-shaped span also contains an
obvious-fake marker, and **92% of the 215 secret spans have `EXAMPLE`/`DO-NOT-USE`/`TEST` within the span+120
characters**. All ИНН/ОГРН/СНИЛС check digits are invalid; all IPs are documentation ranges; all domains are
`example.*`/`.invalid`. A detector suite scored on this file will report a recall that **does not transfer**.
Any recall number computed on `tickets_export.csv` must be reported with that sentence attached.

**Required: a hand-labelled redaction evaluation set, built on production text.**

| Property | Specification |
|---|---|
| Size | **500 tickets**, stratified: 250 payload-carrying, 100 prose-only, 50 mojibake/encoding-odd, 50 self-redacted, 50 truncated-payload |
| Sampling | Seeded hash over ticket ids, from the same window as the training snapshot, **disjoint from the model gold set** |
| Annotation | Character-offset spans with the §4.2 category, by 2 annotators; adjudicate all disagreements. Guideline is a versioned deliverable |
| Cost [estimate] | 500 × ~3 min = ~25 person-hours + ~5 h adjudication. This is *in addition* to the ~70 h model gold set (runbook §4.2) |
| Metric | Per-category **entity recall** and **precision**, plus a document-level leak rate. Adopt the direct-identifier / quasi-identifier distinction and the weighted-recall framing of the [Text Anonymization Benchmark (Pilán et al., Computational Linguistics 48(4), 2022)](https://aclanthology.org/2022.cl-4.19/) — direct identifiers are weighted far above quasi-identifiers, because their recall is the one with a legal consequence |

**Targets (proposed; product/legal own the final numbers, §13 Q3):**

| Category class | Recall target | Precision target | Rationale |
|---|---|---|---|
| Credentials (keys, tokens, JWT, PEM, DB URI passwords) | **≥ 0.99** | ≥ 0.90 | A leaked live key is an incident, not a metric |
| Direct identifiers (PAN, phone, e-mail, СНИЛС, passport, person name) | **≥ 0.98** | ≥ 0.85 | 152-FZ exposure; over-redaction is cheap |
| Quasi-identifiers (ИНН/ОГРН, org name, hostname, IP, bucket, path) | ≥ 0.90 | ≥ 0.90 | Precision matters more here — these carry signal (§4.2) |
| Structural payload blocks | ≥ 0.85 | ≥ 0.95 | A false block detection deletes prose, which *is* the signal |

**Leak audit as a build gate (runs on every snapshot, not just on the eval set).** After stage 8, re-run an
*independent* high-recall detector suite (deliberately different patterns from the production ones, plus the L3 NER
at a lower threshold) over 100% of the snapshot and count residual hits:

| Gate | Threshold | On failure |
|---|---|---|
| Residual credential spans | **0** | Build fails. No override |
| Residual direct identifiers | ≤ 1 per 10,000 tickets, and 0 with a valid Luhn/mod-97/check digit | Build fails |
| Residual quasi-identifiers | ≤ 20 per 10,000 tickets, reviewed and listed in the dataset card | Build warns, requires sign-off |
| Rows quarantined by R6 timeout | ≤ 0.1% | Build warns; quarantined rows are counted in the card and excluded from the snapshot |
| Placeholder-density outliers (rows >60% placeholder tokens) | listed | Manual review — usually a detector running away |

The audit report is an artifact and is referenced by the dataset card. Note the honest limit: the audit measures
*detector agreement*, not truth. It catches regressions; only the human eval set above measures recall.

### 4.8 Adjacent obligations that are not mine to design

- Redaction must run **before the corpus leaves the production perimeter** (model spec §2.9). Where that boundary
  sits is `system-architect`'s and legal's call; the pipeline requirement is that stages 1–8 are runnable inside it
  with no external service dependency (this rules out any hosted PII-detection API for the RU corpus).
- Detected credentials should raise a **counter and an alert**, never a log line containing the value. Whether that
  triggers a customer secret-rotation notice is a product/security decision.
- Memorisation-based extraction results (e.g. [Carlini et al., 2021](https://arxiv.org/abs/2012.07805)) concern
  generative LMs; a 20-way encoder classifier is a much smaller extraction surface. **Do not use that as comfort:**
  the realistic leak path here is the *stored snapshot, the logs, and the error-analysis exports* — all of which are
  plain text — not the weights. Deduplication additionally reduces the memorisation risk that does exist
  ([Kandpal et al., ICML 2022](https://arxiv.org/abs/2202.06539)).

---

## 5. The redaction ablation — how the policy is chosen (key deliverable)

**Nothing in §4.2/§4.3 ships on my say-so.** The transforms above are hypotheses with measured lift behind them;
the policy that ships is the one that wins this experiment.

### 5.1 Arms

Six arms, each a complete `preprocess.json`. All other pipeline stages (labels, dedup, splits, seeds) are byte-identical.

| Arm | Values | Blocks | Purpose |
|---|---|---|---|
| **A0** raw | none | none | Reference ceiling. Never shipped; run in the training environment only, and destroy the artifacts after |
| **A1** uniform | G1 `<REDACTED>` | none | The default people reach for |
| **A2** typed values | G2 | none | Best of the value-only arms, hypothesised |
| **A3** typed values + type-collapsed blocks | G2 | G3 (`TYPE+ATTR`) | **Hypothesised winner overall** (quality ≈ A2, −26% tokens) |
| **A4** typed values + bare-type blocks | G2 | G2 (`<SQL_ERROR>`, no attribute) | Isolates the *attribute*'s contribution — the granularity question |
| **A5** payload-stripped | G2 | `DROP` | The "just delete the pastes" policy, and the floor for how much payloads matter |

Optional **A6** (if A3 wins and budget allows): A3 with placeholders added as tokenizer special tokens and
embeddings resized — tests whether the 30% budget saving converts into quality at fixed `max_len`.

### 5.2 Measurement protocol

- **Downstream model:** the *cheap* one first. Run the full matrix with TF-IDF (char 3–5 + word 1–2) + one-vs-rest
  logistic regression, which trains in minutes, then re-run **only the top 2 arms plus A0 and A5** with the
  fine-tuned encoder (model spec §4.2). Rationale: 6 arms × 5 seeds × a base encoder is ~6 GPU-hours of sweep for a
  decision the linear model can mostly make [estimate from model spec §5.5].
- **Splits:** the §8 split, frozen before the ablation starts. Hyperparameters and thresholds are selected on
  **validation only**; the test set is touched **once per arm**, and touching it again creates a new arm in the
  experiment log (model spec §6.4).
- **Seeds:** **5 seeds** per arm (10 in the <5k-label regime, model spec §5.4). Report **mean ± sd**.
  A single-seed number for any arm is not a result and must not appear in the comparison table.
- **Fixed across arms:** data snapshot id, split assignment, dedup manifest, `labels.json`, loss, decode rule,
  threshold-selection procedure, metric code, `max_len`. Anything else changed → separate table, separate label.

### 5.3 Metrics reported per arm

| Metric | Why |
|---|---|
| **Val `R@P≥0.90`** (primary; micro) | The product metric (model spec §6.2) |
| **Macro-AP** (threshold-free) | Selection metric that does not entangle thresholding |
| **Per-service `R@P90` for the 5 tail services** | This is where payload cues matter most (`redis-cli` ×51, `SQL_QUERY` ×8.2) and where aggregates hide the damage |
| **Slice deltas: RU / EN / mixed; payload-carrying vs prose-only** | Redaction acts almost entirely on payload rows; a policy can be neutral overall and terrible on 30% of tickets |
| **Total tokens and p90/p99 length; fraction ≤128 and ≤256** | The budget/latency consequence (§4.6) |
| **Effective content ratio** = non-placeholder tokens ÷ total tokens | Catches an arm that "saves tokens" by spending them on placeholders |
| **Leak rate** (§4.7 audit, per 10k tickets, direct + quasi separately) | The constraint side of the trade |
| **Wall-clock preprocessing time per 1k tickets** | It runs at serve time too |

### 5.4 Sample size and what the experiment can actually resolve

[measured] the temporal split of this file gives **752 validation tickets carrying ~1,318 positives**. A paired
comparison of two arms on the same tickets (paired bootstrap, 10,000 resamples) resolves differences of roughly
**±3 percentage points** of micro-recall at this size [estimate: normal approximation on ~1,300 paired positive
decisions]. Practical consequences, stated before the experiment runs so nobody re-reads the result later:

- Differences **< 2pp** between arms are **not resolvable** on this corpus. If A2 and A3 land within 2pp, the
  decision falls to the secondary criteria (tokens, leak rate) — which is fine, and is why they are in the table.
- Per-service resolution for the tail is **impossible here**: `terraform-provider` has 1 validation positive
  [measured]. Tail metrics from this file are diagnostics, not evidence (§8.4).
- On production data with the runbook's 1,000–1,500-ticket validation set, the same limits apply. **If the ablation
  must resolve <2pp, the validation set has to grow** — say so to the product owner rather than reporting noise.

### 5.5 Decision rule (write this down before running)

Let `Δ(arm) = val R@P90(arm) − val R@P90(A2)`, with a paired-bootstrap 95% CI.

1. **Compliance filter first.** Any arm failing the §4.7 leak gates is eliminated regardless of quality. A0 is
   eliminated by definition. This is not negotiable and is not traded off against points.
2. **Among surviving arms, pick the cheapest arm that is not significantly worse than the best**: choose the arm
   with the lowest token total whose CI on `Δ` versus the best-performing arm **includes 0**. Cost is measured as
   total tokens (a proxy for both latency and truncation loss).
   **2b — tail guard (applies before rule 2 can select a cheaper arm).** An arm is disqualified if, for any service
   with ≥30 val positives, its average precision is **more than 2pp below** the best arm's, or if the payload-row
   slice is more than 2pp below. A micro metric dominated by `billing`/`console-ui`/`auth` will not notice a tail
   collapse; this guard is what stops rule 2 from trading the tail away for tokens. Services under 30 positives are
   reported, not gated (§8.4).
3. **Tie-break** on: (a) tail-service macro-AP, (b) payload-row slice, (c) implementation simplicity.
4. **If A5 (payload-stripped) is not significantly worse than A3**, ship A5 — it is simpler, cheaper and leaks
   least. I do not expect this, given the lift table, but the experiment must be allowed to say so.
5. **If A0 (raw) beats the best compliant arm by > 5pp**, that is a *finding to escalate*, not a licence to ship
   raw: it means the transform is destroying signal we know how to keep, and the response is to add attributes to
   G3 (row-by-row, guided by which slices lost the most), not to loosen redaction.
6. **Freeze** the winner as `preprocess.json v1`, hash it, and record the arm table in the dataset card. Changing
   the redaction policy afterwards **invalidates every metric measured against the old snapshot** — it is a new
   snapshot and a new `preprocess_version`.

### 5.6 Pilot run of this exact protocol [measured, and its interpretation is limited]

I ran the matrix end-to-end on this corpus with the cheap model (TF-IDF char 3–5 + word 1–2, one-vs-rest logistic
regression, `class_weight='balanced'`; temporal 70/15/15; thresholds tuned at P≥0.90 on the 15% calibration slice;
5 seeds by bootstrap-resampling the training set) to prove the protocol is executable and to calibrate effect sizes.

**Result [measured on the synthetic corpus — describes the generator, not the world].** Test slice = most recent
15% (752 tickets, ~1,283 positives); thresholds from the middle 15%; mean ± sd over 5 seeds.

| Arm (pilot label) | Macro-AP | Micro-precision at the fixed thresholds | Micro-recall (`R@P90` operating point) | Total tokens |
|---|---|---|---|---|
| **A0** raw (`V0_raw`) | 0.744 ± 0.010 | 0.881 ± 0.011 | 0.630 ± 0.009 | 594,546 |
| **A1** uniform `<REDACTED>` (`V1c`) | 0.755 ± 0.013 | 0.883 ± 0.012 | 0.651 ± 0.020 | 543,812 |
| **A2** typed values (`V1`) | 0.754 ± 0.014 | 0.881 ± 0.011 | 0.649 ± 0.018 | 535,627 |
| **A3** typed values + type-collapsed blocks (`V2`) | 0.770 ± 0.013 | 0.875 ± 0.011 | 0.674 ± 0.022 | 440,668 |
| **A5** payload-stripped (`V3`) | **0.773 ± 0.012** | 0.874 ± 0.011 | **0.676 ± 0.017** | 423,419 |

**What the pilot shows, and what it does not.**

- The protocol runs end to end and produces exactly the table §5.5 consumes. That was the main purpose.
- **Every redaction arm beats raw** on this corpus (+1.9 to +4.6pp micro-recall, +1.0 to +2.9pp macro-AP). Raw text
  is the worst arm. That direction is worth stating because the intuition "redaction can only remove information"
  is wrong: high-entropy identifier spans are noise that a bag-of-n-grams model spends capacity on.
- **A3 and A5 are within noise of each other** (0.674 ± 0.022 vs 0.676 ± 0.017; well under the ±3pp resolution of
  §5.4), and both sit ~2.3pp above A1/A2 — which is *at* the edge of what a 752-ticket validation slice resolves,
  not comfortably beyond it. Applying §5.5 rule 2 mechanically, the pilot **selects A5 (payload-strip)** —
  the cheapest arm not significantly worse than the best. That contradicts my §4.2 recommendation, and I am leaving
  the contradiction in the document rather than hiding it, because it is exactly what the decision rule is for.
- **Why I still expect A3 to win on production data**, stated as a hypothesis with its test:
  1. The pilot model is TF-IDF + linear. It cannot exploit a `TYPE+ATTR` attribute the way a contextual encoder can:
     `<PAYMENT_DECLINE: do_not_honor>` is, to a char-n-gram model, just another token — while the payload it
     replaced supplied dozens of correlated n-grams. The arms must be re-run on the encoder before the choice is
     made; §5.2 already requires this.
  2. The block-kind lift (§4.2) is concentrated in **tail** services (`redis-cli` ×51 for `managed-redis`,
     `SQL_QUERY` ×8.2 for `managed-postgres`, `DNS_ZONE` ×10.0), which contribute almost nothing to a **micro**
     metric dominated by `billing`/`console-ui`/`auth`. A micro-recall tie can hide a tail collapse — which is
     precisely why §5.3 requires the per-tail-service and payload-row slices, and why they, not the headline, decide
     rule 3.
  3. Payloads here are drawn from ~60 templates (corpus README), so their marginal information content is lower than
     real pastes'. This biases the pilot **towards** deletion.
- The per-service view on the arms that matter [measured, 3 seeds, average precision on the test slice]:

| Service | test positives | A0 raw | A3 type-collapse | A5 payload-strip |
|---|---|---|---|---|
| managed-redis | 18 | 0.785 ± 0.049 | **0.813 ± 0.053** | 0.807 ± 0.065 |
| managed-postgres | 76 | 0.821 ± 0.022 | **0.816 ± 0.020** | 0.807 ± 0.017 |
| networking | 117 | 0.782 ± 0.010 | **0.783 ± 0.008** | 0.774 ± 0.012 |
| billing | 107 | 0.823 ± 0.018 | 0.833 ± 0.015 | **0.837 ± 0.015** |
| api-gateway | 96 | 0.804 ± 0.026 | 0.825 ± 0.023 | **0.829 ± 0.022** |
| console-ui | 96 | 0.813 ± 0.014 | **0.820 ± 0.003** | 0.814 ± 0.009 |
| notifications | 53 | 0.865 ± 0.018 | **0.869 ± 0.013** | 0.865 ± 0.017 |
| logging | 42 | 0.817 ± 0.021 | 0.870 ± 0.023 | **0.873 ± 0.022** |

The three services whose block-kind lift is highest — `managed-postgres` (`SQL_QUERY` ×8.2), `networking`
(`DNS_ZONE` ×10.0, `K8S_EVENT` ×5.2) and `managed-redis` (`REDIS_CLI` ×51.2) — are the three where A3 beats A5, and
they are the only ones. **The direction predicted by the lift table appears; the magnitude (≈0.9pp AP) is inside
the seed noise at 3 seeds and these test sizes.** That is the honest reading: suggestive, not decided. It is also a
concrete demonstration of §5.4 — this corpus cannot resolve the question that matters, so the production ablation
must, with the encoder, 5 seeds, and the per-service slice as a first-class output.

**Bottom line on the pilot: mechanism validation, not evidence about production.** Labels are generator-assigned,
payload templates are ~60 fixed shapes, and lexical statistics do not transfer (corpus README, "Known gaps"). What
it does establish: the harness runs, the arms are separable, the effect sizes are single-digit percentage points,
raw text is not the best arm, and the protocol produces exactly the table §5.5 consumes.

---

## 6. Label normalisation and the label contract

### 6.1 Hygiene [measured]

| Defect | Rows | Fix |
|---|---|---|
| Space around the separator (`"logging, subscriptions"`) | 49 | `strip()` each element |
| Mixed case (`"Billing"`) | 38 | `lower()` |
| Internal duplicate (`"logging,logging"`) | 40 | Set semantics |
| Distinct raw surface strings | 59 → **20** after normalisation; 39 dirty forms | — |
| Empty `services` | 391 | §6.3 |

`norm_services(s) = sorted(set(lower(strip(x)) for x in split(s, ',') if strip(x)))` — the same function as runbook
§2, applied in Python at build time. **Any element that does not map into `labels.json` after normalisation is a
hard build error, not a silent drop.** [measured] there are none in this file; in production, a rename will produce
some, and it must stop the build (model spec §4.8: renames/merges require a product-owner mapping table).

### 6.2 The `labels.json` contract

```
{ "label_set_version": "1.0.0",
  "labels": ["access-control","api-gateway","auth","backups","billing","cdn","compute",
             "console-ui","dns","integrations","logging","managed-postgres","managed-redis",
             "message-queue","monitoring","networking","notifications","object-storage",
             "subscriptions","terraform-provider"],
  "retired": [],
  "aliases": {} }
```

- Order defines the head index. **Indices are append-only; a retired service's index is never reused** (model spec
  §4.8). Retirement moves the name to `retired` and keeps the slot.
- `aliases` maps historical spellings to canonical names; it is populated by the product owner, not inferred.
- Every snapshot, model, threshold file and prediction records `label_set_version`. Consumers reject on mismatch
  rather than mapping by position (model spec §9.2).
- The 20 names above are **[measured]** as the exact normalised vocabulary of this file, and match the corpus
  README's taxonomy.

### 6.3 The 391 empty-`services` rows: abstention or untriaged? — **untriaged. Treat as unlabelled.**

The corpus README calls ~100 of them genuinely unactionable "abstention cases". The remaining ~290 are untriaged.
Since the two are not distinguishable from the row itself, and the cost of confusing them is asymmetric, I measured
what they look like [measured]:

| Signal | Empty-`services` rows | Corpus |
|---|---|---|
| Median description length | 192 chars | 222 chars |
| Rows under 80 chars | 39 / 391 (10.0%) | 232 / 5,013 (4.6%) |
| Rows carrying a pasted payload | 103 / 391 (26.3%) | 1,771 / 5,013 (35.3%) |
| Rows with non-empty `labels` | 126 / 391 (32.2%) | — |
| Priority | normal 237, high 83, low 41, **urgent 30** | — |
| Flags | 40 escalated, 9 vip, 7 duplicate, 6 sla_breach | — |

If these were mostly abstention cases they would be short, payload-free, low-priority junk. They are not: they are
near-normal in length, a quarter carry pasted evidence, 30 are `urgent` and 40 are `escalated`. **Decision:**

- **Excluded from supervised training and from all evaluation.** Treating "no service" as 20 negatives teaches the
  model that a correctly-labelled ticket is unlabelled — this is precisely the positive-unlabelled trap the runbook
  §3 warns about for Case A, in a more extreme form.
- **Retained in the snapshot** with `label_status = "unlabelled"`, for (a) domain-adaptive pretraining (model spec
  §4.7), (b) inference-time smoke tests, (c) the abstention behaviour study.
- **The abstention question is answered by annotation, not by the empty column.** If the product needs to know how
  often a ticket is genuinely unactionable, put a sample of these rows in the human gold set with an explicit
  `unactionable` option (runbook §4). Until then, **the model spec's abstention behaviour (§4.6.1) is evaluated by
  coverage, not by fitting empty targets.**
- **Never** impute labels for these rows with a model. That is the distillation trap (proposal §3) with extra steps.

---

## 7. Deduplication tuned to this corpus

### 7.1 What is actually in here [measured]

| Structure | Size | Lexical similarity | Caught by MinHash? |
|---|---|---|---|
| Exact duplicate descriptions | 6 rows | 1.0 | Trivially |
| Auto-alert template clique | **99 rows across 51 orgs** | internal pairwise Jaccard min 0.67, median 0.72, max 0.92 | **Threshold-critical** — see below |
| Copy-paste refile clusters | README says 72 rows / 26 clusters; my "пишу повторно" probe finds only 10 [measured, partial detector] | 0.6–0.9 | Mostly |
| Semantic cluster (PG connection pool) | 63 candidate rows | median pairwise Jaccard **0.03**; exactly **1** pair ≥ 0.4 | **No — completely invisible** |
| Incident burst 2025-11-18 | 109 rows | median pairwise Jaccard **0.01**; **0** pairs ≥ 0.6 | **No, and it should not be** |

The burst measurement matters for how the two mechanisms divide labour: **bursts are not near-duplicates**. They are
different customers describing one root cause in their own words. Dedup will not touch them; only the temporal gap
(§8) protects against them. Conflating the two mechanisms is a common and expensive mistake.

### 7.2 Prescribed cascade

**Pass 0 — exact.** Hash the normalised (post-stage-8) text. Collapse exact duplicates, keep the earliest,
record the group.

**Pass 1 — MinHash/LSH on character 5-grams** ([Broder 1997](https://ieeexplore.ieee.org/document/666900);
the standard corpus-scale recipe is [Lee et al., ACL 2022, *Deduplicating Training Data Makes Language Models
Better*](https://aclanthology.org/2022.acl-long.577/)). Configuration measured on this corpus:

| Config | Candidate pairs | Pairs ≥0.8 | Recall of known clique pairs |
|---|---|---|---|
| K=128, 32 bands (r=4) | 9,487 | 560 | **1.000** |
| K=128, 16 bands (r=8) | 3,145 | 545 | 0.970 |
| K=256, 32 bands (r=8) | 5,059 | 560 | 1.000 |

**Recommended: K=128, r=4, 32 bands** — full recall at the 0.8 threshold on this corpus, and the 560/168 figures
reproduce the corpus README exactly, which is a useful cross-check that the implementation is correct.

> **Implementation warning [measured].** My first implementation drew MinHash multipliers up to 2⁶¹ and multiplied
> in `numpy.int64`. The products overflow **silently**, degrading the permutations: the alert clique then fragmented
> into components of 25/22/13/13/12 rows instead of one, with no error and no warning. Use a 31-bit prime modulus
> with 32-bit multipliers (or a vetted library), and **regression-test the implementation against a known cluster**.
> This class of bug does not announce itself; it just quietly under-deduplicates.

**Threshold [measured] — and it is not the obvious one:**

| Threshold | Clusters | Rows in clusters | Rows dropped if collapsed to 1 representative | Note |
|---|---|---|---|---|
| J ≥ 0.9 | 32 | 68 | 36 | Misses the template family |
| **J ≥ 0.8** | 32 | 168 | 136 | The alert clique **fragments into 7 components** (25/22/13/13/12/7/6) covering 98 of 99 rows: pairs are individually ≥0.8 but not transitively connected |
| **J ≥ 0.7** | 28 | 177 | 149 | The clique becomes **one 99-row component** — the correct grouping |
| J ≥ 0.6 | 90 | 307 | 217 | Starts absorbing unrelated rows (corpus README: 71 unintended pairs at 0.60) |

**Recommendation: cluster at J ≥ 0.7 with connected components, but *do not* collapse to one representative.**
Instead assign a `dedup_group_id` and use it as a **grouping key for splitting** (§8) and as a **training sample
weight** (`w = 1/√|group|`). Reasons: (a) template tickets are real production traffic and deleting them biases the
class prior — the alert clique is 99 monitoring tickets [measured, `monitoring` in 100% of them]; (b) a group id is
reversible, a deletion is not; (c) the model spec's §2.6 requirement is that near-duplicates must not *straddle*
the split boundary, which grouping satisfies without discarding data.
Exception: **collapse exact duplicates only** (6 rows here).

> **Why `dedup_group_id` and not `organization_id` is the grouping key.** [measured] the 99-row alert clique spans
> **51 organisations** and 100% of its rows carry `monitoring`. Grouping by organisation would scatter that one
> template across every split — the exact leak the grouping was supposed to prevent — while grouping by dedup
> cluster contains it. Organisation is the wrong unit here; near-duplicate cluster is the right one. (§8.2 shows
> org-grouping also destroys the temporal property, so it loses on both counts.)

**Pass 2 — embedding-based near-duplicate detection, for what MinHash cannot see.** The PG connection-pool cluster
has median pairwise Jaccard 0.03 [measured] — no lexical method will ever find it. Use the same multilingual
encoder as the classifier (frozen, mean-pooled; model spec §4.3), cluster by cosine similarity, in the manner of
[SemDeDup (Abbas et al., 2023)](https://arxiv.org/abs/2303.09540). Specification:

- Compute embeddings once per snapshot; store them as a snapshot artifact (they are reused by §9's kNN label QA).
- **The cosine threshold must be calibrated, not guessed**: take the 99-row alert clique (known positives) and 500
  random pairs (known negatives), plot the similarity distributions, choose the threshold at the point where the
  false-merge rate on the random pairs is ≤1%, and **report both distributions in the dataset card**. Expect
  something in the 0.85–0.95 range for a mean-pooled multilingual encoder [estimate — measure it].
- **Semantic groups get a separate id** (`semantic_group_id`) and are used **only** to prevent cross-split leakage
  and for reporting — never to delete rows. Semantic similarity is not duplication: two customers hitting the same
  root cause are two genuine training examples.
- **Bursts are exempt from Pass 2 by construction** (a burst is semantically one incident). Tag burst rows with
  `burst_id` (§8.3) and let the *split* handle them; do not let the semantic clusterer eat 358 rows.

**Reporting:** dedup counts by pass, cluster-size histogram, the largest 10 clusters with a sample title each, and
the number of cross-split pairs removed. A large number here is a finding about the corpus (runbook §4.3).

---

## 8. Splits for this specific file

Policy inherited from model spec §2.6 / runbook §6: **temporal, grouped, with a ≥1-week gap.** Measured against this
file, that policy has one hard conflict and one hard limit.

### 8.1 Measured behaviour of the prescribed split

| Split variant | Train / Val / Test | Test orgs also in train | Test tickets from train-orgs | terraform-provider positives in test |
|---|---|---|---|---|
| Pure temporal 70/15/15 | 3,509 / 752 / 752 | 186 of 186 | **752 of 752 (100%)** | 1 |
| Temporal + 7-day gaps | 3,509 / 669 / 681 (154 rows dropped into the gaps) | 177 of 177 | **681 of 681 (100%)** | 1 (message-queue: **0**) |
| Org-grouped by median org time | 3,510 / 763 / 740 | **0** | 0 | 2 |

### 8.2 The conflict: temporal and org-grouping are mutually exclusive here — pick temporal

[measured] Under a pure temporal split, **every** test organisation also appears in train (186/186), because 210
organisations file across 15 months. Enforcing "no org straddles the boundary" therefore requires discarding the
entire test set, or abandoning temporality — and org-grouping by median org time does exactly that: **99.9% of
train tickets are later than the earliest test ticket** [measured], i.e. the split stops measuring the thing it
exists to measure (drift, taxonomy change, phrasing shift; model spec §2.6, [Lyu et al., TOSEM
2021](https://dl.acm.org/doi/fullHtml/10.1145/3447876)).

**Decision: temporal split is primary. Org-grouping is demoted from a constraint to a diagnostic.** Concretely:

1. Sort by `created_at`; train = oldest 70%, val = next 15%, test = most recent 15%; **≥7-day gaps** at both
   boundaries ([measured] costs 154 rows, 3.1% — cheap).
2. **Group-aware, not org-aware:** no `dedup_group_id` and no `semantic_group_id` may straddle a boundary. Where one
   does, the later members are dropped from the later split. [measured] under the temporal split this costs
   **10 test rows** at J≥0.8 and **23** at J≥0.6 — trivially affordable, and it removes the actual leak mechanism
   that org-grouping was a blunt proxy for.
3. **Report the org-overlap number in the dataset card** (100% here) and add an **unseen-org test slice** if one
   exists. [measured] on this file it contains **0 tickets**, so the card must say "unseen-org generalisation is
   not measurable on this snapshot" rather than leaving a reader to assume it was checked.
4. **Required diagnostic (model spec §2.6):** report random-split and temporal-split validation numbers once. The
   gap is the drift magnitude.

### 8.3 Burst awareness

[measured] 536 tickets fall on days with ≥25 tickets (top days: 109 on 2025-11-18, 99 on 2026-03-05, 74, 56, 42,
40, 38, 28; median day is 13 tickets). Under the temporal split, 80 land in val and 74 in test (10.6% / 9.8%).

- Tag every ticket with `burst_id` (day-level detection: >3× the trailing-28-day median daily count, merged with a
  ±12 h window) and carry the tag into the snapshot.
- **A burst must not straddle a split boundary.** [measured] with boundaries at 2026-04-04 and 2026-06-04, no burst
  day falls within 7 days of a boundary — but the margin is thin: the 2026-05-27 burst (38 tickets) is **8 days**
  from a boundary and the 2026-04-14 burst (42) is **10 days**. A 14-day gap would cut into both. So the rule must
  be enforced explicitly and re-checked on every rebuild, not assumed from this one measurement: if a burst
  straddles or falls inside the gap, move the whole burst into the earlier split and record it.
- **Report burst-slice metrics separately.** A model evaluated on a test set that is 10% one incident is being
  scored on one scenario 74 times. The card reports test composition by burst.

### 8.4 The rare tail: what is feasible, stated numerically

[measured] with the temporal+gapped split:

| Service | Corpus positives | Train | Val | Test | Verdict |
|---|---|---|---|---|---|
| managed-redis | 98 | 67 | 12 | 17 | Trainable; **test estimate unusable** (n<30) |
| dns | 41 | 31 | 5 | 5 | Not evaluable |
| cdn | 38 | 26 | 6 | 4 | Not evaluable |
| message-queue | 31 | 25 | 5 | **0** | **Not evaluable; not even present in test** |
| terraform-provider | 15 | 13 | 1 | 1 | Not learnable, not evaluable |

To estimate a per-service precision to ±10pp you need roughly 100 positives in the split
[estimate: normal approximation, p≈0.9]. At 15 corpus-wide positives for `terraform-provider`, **no split of this
file can produce a usable estimate** — a fact about the corpus, not about the split algorithm.

**Prescription:**

1. The dataset card and every metrics table carry `n_positives` per service per split, and services with test
   `n < 30` are reported as **"insufficient data"** — never given a precision number (model spec §4.8, §6.3).
2. **Do not** rescue the tail by stratifying the temporal split on labels. That leaks the label into the split and
   breaks the temporal property.
3. **Do** build a separate **tail diagnostic pool**: all positives for the 5 tail services, held out of nothing,
   used for qualitative error analysis only, never for a headline metric. Mark it `eval_role = "diagnostic"`.
4. The real fix is upstream: either targeted annotation of tail services (runbook §5.2 arithmetic) or the keyword
   rule / kNN path for them (model spec §4.8). **The pipeline's job is to make the shortage visible, not to hide
   it behind an average.**

### 8.5 Split artifact

Split assignment is stored per-row in the snapshot and hashed. It is **never** recomputed at training time: a
recomputed split with a different sort tie-break is a different experiment. `split ∈ {train, val, test, gap,
excluded}` with `exclusion_reason ∈ {unlabelled, quarantined, dedup_cross_split, burst_boundary}`.

---

## 9. Label QA on this corpus — and what it means when the author is a generator

The three-stage funnel (kNN + confident learning → LLM adjudication → human) is the right mechanism, but its
*meaning* changes completely depending on who produced the labels.

### 9.1 The funnel

**Stage 1 — automatic candidate generation (cheap, runs on 100%).**
- kNN over the frozen multilingual embeddings already computed in §7 Pass 2: flag rows whose label set disagrees
  with the majority of their k=10 nearest neighbours (in the same language, excluding same-`dedup_group`
  neighbours — otherwise a template clique votes for itself).
- Confident learning over out-of-fold predicted probabilities:
  [`cleanlab.multilabel_classification.filter.find_label_issues(..., multi_label=True)`](https://docs.cleanlab.ai/stable/cleanlab/multilabel_classification/filter.html),
  built on [Northcutt et al., JAIR 2021](https://arxiv.org/abs/1911.00068). Requires 5-fold out-of-fold
  probabilities from the cheap model — hours, not days.
- **Both signals are needed**: kNN catches "this ticket is unlike its label's neighbourhood", confident learning
  catches "the model is confidently at odds with the stored set". Rank by the union, deduplicate by
  `dedup_group_id`.
- Expected volume [estimate]: 5–15% of rows flagged. On 4,622 labelled rows that is 230–700 candidates.

**Stage 2 — LLM adjudication (triage only, never authoritative).**
- Only for flagged rows. Prompt with the taxonomy definitions and the redacted text; require a structured verdict
  {agree / add service X / remove service Y / unclear} plus a one-line reason.
- **Hard constraints:** (a) 152-FZ — no hosted foreign API on RU ticket text (model spec §2.9, §13 Q4); a
  self-hosted model or nothing. (b) An LLM verdict **never** writes a label. It re-ranks the human queue. (c) LLM
  verdicts are recorded with the model id and prompt hash, and are **excluded from any evaluation set** —
  see [`llm-fallback-policy.md`](./llm-fallback-policy.md) for the contamination rules.
- Measure it before trusting it: on the 300 double-annotated gold tickets, report LLM-vs-adjudicated agreement. If
  it is below the flag-precision of stage 1 alone, drop stage 2 — it is adding cost and a contamination risk for
  nothing.

**Stage 3 — human adjudication (authoritative).**
- Top-N by combined score, blind to the stored value (runbook §4.1 rule 4). N set by budget; ~200–400 rows is a
  day of one senior agent [estimate].
- Every change is written as a **correction record** (`ticket_id, old_set, new_set, reason, annotator, timestamp`),
  never as an in-place edit. The snapshot is rebuilt from raw + corrections, so the pipeline stays a pure function.

### 9.2 What this means **on this file** — read this before quoting any QA number

The corpus README is unambiguous: labels are author-assigned by the generator, there is no blind annotation, no
inter-annotator agreement, no α. That makes every row of this file **Case G — generator-assigned** — a case the
runbook's A/B/C/D partition does not contain, and it is *closer to Case B than to Case A*: the label agrees with
whatever produced the text, by construction.

Consequences, stated without hedging:

| Question | Answer on this file |
|---|---|
| Can the funnel be *built and measured for throughput* here? | **Yes** — flag rates, cluster behaviour, runtime, queue sizes are all real |
| Can flag **precision** be measured here? | **No.** "Correct" is defined by the generator; a flagged row that the generator labelled differently is not necessarily an error |
| Can Krippendorff's α (MASI) be computed? | **No.** There is one annotator and it is a program. α is undefined, not low |
| Can the dataset be used to claim a model is good? | **No.** Any accuracy number describes the generator (corpus README, "Known gaps") |
| Can it be used to choose the redaction policy? | **Partially** — the §5 pilot validates the *mechanism* and the effect-size scale. The production decision needs production data (§5.6) |
| Can it be used for engineering, throughput, latency, memory, split logic, gate wiring? | **Yes — this is what it is for** |

**Therefore §11's acceptance gates split into two tiers**: engineering gates (checkable here) and evidence gates
(checkable only on human-labelled production data). A v1 declared on this file passes tier 1 only, and the dataset
card must say so in its first paragraph.

---

## 10. Boilerplate, generator artefacts, and the shortcut risk

The corpus README states 135 sentences occur ≥20× and account for 47.9% of sentence instances. With my sentence
splitter [measured]: 12,876 distinct sentences over 33,349 instances; **238 sentences occur ≥20× and account for
40.5% of all sentence instances** (top: `Мешает работать, но не блокирует.` ×123, `Из-за этого не можем закрыть
задачу в спринте.` ×117). The exact number depends on the splitter; the order of magnitude is the point.

**What this does to the pipeline:**

1. **Do not strip boilerplate in v1.** Generic "moves" (greeting, impact, ask, sign-off) exist in real tickets too;
   removing them makes this corpus *less* like production, not more. What is unrealistic here is their
   *concentration*, and stripping them would also remove genuine impact/urgency phrasing.
2. **Do measure it and expose it.** Add a snapshot column `boilerplate_ratio` = share of sentence instances that
   appear ≥20× corpus-wide. Report model metrics sliced by it. If performance is much better on high-boilerplate
   rows, the model is reading the seam.
3. **Shortcut detection is mandatory, and cheap.** [measured] I found a stray CJK character `長` inside one
   generator phrasing, replicated verbatim across 4 rows — all four labelled `access-control`. A single rare token
   that perfectly predicts a class is exactly the artefact a fine-tuned encoder will happily latch onto. Required
   diagnostic on every snapshot: **for every token appearing in 3–50 rows, compute the label-conditional purity and
   list the top 50**. Anything with purity 1.0 over ≥3 rows is inspected by a human before the snapshot is frozen.
4. **A model trained here learns the generator.** Expect implausibly high metrics; treat any `R@P90` above the
   human ceiling reported for real annotation work as a red flag rather than a success (model spec §6.2).
5. Structural statistics transfer; lexical ones do not (corpus README). So: **use this file to validate label
   balance handling, cardinality handling, timing/split logic, duplication logic and payload handling. Do not use
   it to tune vocabulary, `max_len`, or anything that depends on lexical diversity** — the token-length figures in
   §4.6 are the honest exception, because payload text is the realistic part of this corpus and the arithmetic is
   about characters, not about vocabulary richness.

---

## 11. Acceptance gates and the dataset card

### 11.1 Tier 1 — engineering gates (checkable on this file; **all must pass** for v1)

| # | Gate | Threshold | Measured status here |
|---|---|---|---|
| G1 | Parse: rows in = rows out + quarantined; no field truncation; embedded newlines/quotes preserved | 5,013 accounted for | [measured] 5,013 parsed |
| G2 | Label normalisation: every element maps into `labels.json` | 100% | [measured] 20/20, 0 unmapped |
| G3 | Determinism: two builds from the same input produce identical row hashes | byte-identical | to be verified |
| G4 | Preprocessing is a pure function: `f(f(x)) = f(x)` (idempotent under re-application) | 100% of rows | to be verified — **placeholders must not be re-detected as new entities** |
| G5 | Leak audit (§4.7): residual credentials | **0** | to be verified |
| G6 | Leak audit: residual direct identifiers | ≤1 / 10k, none check-digit-valid | to be verified |
| G7 | R6 quarantine rate | ≤0.1% | to be verified |
| G8 | Mojibake: no document retains a ≥4-char high-Latin run after stage 3 | 0 rows | [measured] 9 rows before, 0 after doc-scoped repair |
| G9 | Dedup: no `dedup_group`/`semantic_group` straddles a split boundary | 0 | [measured] 10 rows to drop at J≥0.8 |
| G10 | Splits: ≥7-day gaps; no burst straddles a boundary | enforced | [measured] holds with 154 rows dropped |
| G11 | Every snapshot row carries: `provenance`, `label_status`, `split`, `dedup_group_id`, `lang`, `burst_id`, `preprocess_version` | 100% | to be verified |
| G12 | Snapshot content hash recorded; `labels.json`, `preprocess.json`, dedup manifest and audit report attached | present | to be verified |
| G13 | Shortcut scan (§10.3) run and reviewed | reviewed | [measured] 1 artefact found (`長`, 4 rows) |
| G14 | Round-trip: serving-path preprocessing of 500 sampled raw rows reproduces the snapshot text byte-for-byte | 100% | to be verified — **this is the train/serve skew gate** |

### 11.2 Tier 2 — evidence gates (require human-labelled production data; **cannot pass on this file**)

| # | Gate | Threshold |
|---|---|---|
| E1 | Redaction eval set (§4.7) exists, adjudicated, and recall targets met per category | per §4.7 table |
| E2 | Ablation §5 executed with ≥5 seeds; winner chosen by §5.5; arm table in the card | complete |
| E3 | Gold set exists with Krippendorff α (MASI) ≥ 0.67 (runbook §4.2) | α ≥ 0.67 |
| E4 | Label provenance reconstructed (runbook §1–§3); no non-human label in val/test | 100% |
| E5 | Label-QA funnel flag precision measured against adjudicated human labels | reported |

### 11.3 Dataset card contents (required, versioned with the snapshot)

1. **First paragraph: permitted and forbidden uses** (§9.2 table). For a snapshot built from `tickets_export.csv`:
   "engineering validation only; no quality claim may be derived from this dataset."
2. Provenance: source export, extraction query/file hash, row counts in/out, quarantine counts and reasons.
3. Label provenance composition (Case A/B/C/D/G counts) and the label-status breakdown (labelled / unlabelled).
4. Class balance table with per-split `n_positives`, cardinality histogram, co-occurrence matrix.
5. Split definition: boundaries, gap size, rows dropped to gaps, burst composition per split, org-overlap number,
   unseen-org slice size (**0 here**), and the random-vs-temporal drift diagnostic.
6. Dedup report: pass-by-pass counts, thresholds, cluster-size histogram, calibration plot for the embedding
   threshold, cross-split removals.
7. **Redaction policy**: the winning arm, the full §4.2 transform table as applied, `preprocess_version`,
   placeholder inventory with counts, effective content ratio, token-budget table (§4.6).
8. **Leak audit results** and the redaction eval-set metrics, with the "measured recall overstates production
   recall" caveat spelled out when the eval was run on synthetic text.
9. Known biases and gaps: language imbalance (66.8/28.5/4.7), tail services and their evaluability, boilerplate
   concentration, generator artefacts (§10), no `ticket_messages`, no α.
10. Licensing/PII: data is first-party; 152-FZ constraints on where the snapshot may live; retention and deletion
    policy for the snapshot and for error-analysis exports (owner: legal/`system-architect`).
11. Reproducibility: pinned versions, seeds, hashes of every artifact, the exact scripts and their commit.

---

## 12. Risks and failure modes

| # | Risk | How it manifests | Detection |
|---|---|---|---|
| 1 | **Redaction destroys the tail's only cue** | Tail services degrade while aggregate metrics look unchanged | §5.3 tail-slice metric per arm; lift table recomputed after redaction |
| 2 | **Detector recall measured on synthetic text** | Confident 0.99 recall, real leaks in production | [measured] 92% fake-marker co-location; gate on the §4.7 human eval set, not on this file |
| 3 | **Train/serve skew in preprocessing** | Offline good, shadow mode bad (proposal §7 shadow gate) | G14 round-trip gate; `preprocess_version` mismatch is a hard error |
| 4 | **Placeholder collisions become a shortcut** | Model keys on `<PAYMENT_DECLINE>` for `billing` and ignores prose | Attribute-ablation arm A4; per-slice metrics on payload vs prose rows; §10.3 purity scan |
| 5 | **Over-redaction eats prose** | Effective content ratio drops; short tickets become placeholder soup | Placeholder-density outlier gate (§4.7); block-detector precision target 0.95 |
| 6 | **Silent MinHash degradation** | Under-deduplication, inflated metrics | [measured] the int64 overflow case; regression test against the 99-row clique; assert 560 pairs/168 rows on this fixture |
| 7 | **Split leakage via templates and semantics** | Test precision far above shadow precision | Group-aware splitting; report cross-split pairs removed; random-vs-temporal gap |
| 8 | **Burst domination of a split** | Metric is one incident measured 74 times | Burst slice in the card; per-burst metrics |
| 9 | **Empty-`services` rows treated as negatives** | Systematic under-prediction, recall collapse | `label_status` column; assert no `unlabelled` row enters a supervised batch |
| 10 | **Label-set drift (rename/merge)** | Silent index reuse corrupts every historical comparison | `labels.json` append-only + version check; unmapped label = build failure |
| 11 | **Generator artefacts learned as features** | Implausibly high metrics on this corpus | §10.3 purity scan; boilerplate slice |
| 12 | **Language slice mis-assignment** | The EN gate in the ship criterion is computed on the wrong rows | [measured] 455-row shift; compute `lang` on the prose channel, store confidence |
| 13 | **Snapshot itself is the leak** | PII in a training bucket, an error-analysis CSV, or a notebook output | Leak audit on the snapshot; retention policy; never export raw text into analysis artifacts without re-running the audit |
| 14 | **Ablation over-read** | A 1pp difference is declared a winner | §5.4 resolution limit stated before the run; paired bootstrap CIs; 5 seeds |

---

## 13. Open questions (with owner)

| # | Question | Owner | Why it blocks |
|---|---|---|---|
| **Q1** | Will the production pipeline run on the real `tickets` table, and does `ticket_messages` exist there? Cases A/B/C/D cannot be reconstructed from this file at all | `dataset-provider` / backend | Determines whether the snapshot has usable provenance or is Case-G-equivalent (§9.2). Blocks any quality claim |
| **Q2** | Who annotates the **500-ticket redaction eval set** (§4.7), and when? ~30 person-hours, distinct from the model gold set | Support lead + security | Blocks E1, therefore blocks shipping any redaction policy to production |
| **Q3** | Sign-off on the per-category recall/precision targets and the leak-rate gates in §4.7 | Legal/DPO + security | These are risk-appetite decisions, not ML decisions. Gates cannot be enforced until they are numbers |
| **Q4** | 152-FZ: may redacted ticket text leave the production perimeter for training? May a self-hosted LLM be used for stage-2 label triage? | Legal/DPO | Determines whether §9's stage 2 exists and where the snapshot may live |
| **Q5** | Closed allowlists for the `TYPE+ATTR` attributes: host roles, resource kinds, error strings | Product owner + me | An open attribute vocabulary is a re-identification channel; a too-small one throws away ×7–×51 cues |
| **Q6** | Is the 20-service taxonomy stable? Pending renames/merges? Who owns `labels.json`? | Product owner | Renames invalidate historical labels (model spec §4.8) |
| **Q7** | Preprocessing latency budget at serve time — does L3 NER fit? Presidio+Slovnet on CPU per ticket is unmeasured, and the architecture doc already assumes split online/batch tiers | `system-architect` + me | A different tier set at train and serve is a silent train/serve skew (§14.2). Must be resolved before implementation; my recommended resolution is option (b) in §14.2 |
| **Q8** | Retention: how long do snapshots, embeddings and error-analysis exports live, and where? | `system-architect` + legal | Snapshots are the main realistic leak surface (§12.13) |
| **Q9** | Is an abstention/"unactionable" option added to the annotation guideline, so the 391-row question (§6.3) can be answered properly? | Support lead + product | Otherwise abstention behaviour stays unmeasurable |
| **Q10** | Tail services: annotate more, merge, or serve by rule? At 15 positives `terraform-provider` cannot be learned or evaluated | Product owner | §8.4 — the pipeline can only make this visible, not fix it |

Q1, Q2, Q3 and Q7 block implementation start. The rest block the v1 declaration.

---

## 14. Implementation notes and inference requirements

These are **constraints for `system-architect`**. I am not specifying service boundaries, queues, storage topology
or schema.

### 14.1 Artifacts the pipeline must emit

| Artifact | Contents | Size [estimate] |
|---|---|---|
| `snapshot.parquet` (or equivalent) | One row per ticket: ids, `text_in`, multi-hot labels, `label_status`, `provenance`, `split`, `dedup_group_id`, `semantic_group_id`, `burst_id`, `lang`, `lang_confidence`, `boilerplate_ratio`, `preprocess_version`, `n_tokens` | ~5 MB for 5k rows; ~50 MB at 50k |
| `labels.json` | §6.2 | <5 KB |
| `preprocess.json` | Full ordered detector list with pattern versions, placeholder inventory, attribute allowlists, granularity level, block policy, template, truncation strategy, `max_len` | 50–200 KB |
| `dedup_manifest.json` | Group memberships, thresholds, calibration evidence | ~1 MB |
| `redaction_audit.json` | Leak-audit counts by category, quarantine list, placeholder density outliers | small |
| `embeddings.npy` | Frozen encoder embeddings (dedup Pass 2 + label QA) | 5,013 × 768 fp32 ≈ 15 MB; 50k ≈ 154 MB |
| `dataset_card.md` | §11.3 | small |
| `ablation_report.md` | §5 arm table, seeds, CIs, decision trace | small |

### 14.2 The serving-side contract (stages 2–8 only)

- **Input:** `{ticket_id, title, description, priority, created_at}` — identical to model spec §9.2.
- **Output:** the exact `text_in` string plus `{preprocess_version, placeholder_counts, lang, quarantined: bool}`.
- **Purity:** deterministic, no network, no clock, no locale dependence, no randomness. Same input → same bytes,
  forever, for a given `preprocess_version`.
- **Idempotence:** re-running on already-processed text must be a no-op (gate G4).
- **Failure mode:** on R6 timeout or a detector exception, return `quarantined: true` and **no text** — never raw
  text. The caller decides what to do with a quarantined ticket; the pipeline never leaks by falling open.
- **Versioning:** `preprocess_version` is compared against the model's; mismatch is a hard error, mirroring the
  `label_set_version` rule in model spec §9.2.
- **One tier set, not two — the reconciliation flag.** The architecture doc proposes distinct detector tiers for
  online and batch use (`redaction.tiers.online = [block, pattern, validator]` vs `...batch = [..., ner]`). From the
  ML side that is only admissible under one condition: **the training corpus must be built with the tier set that
  runs at serve time.** If NER runs in batch only, training text contains `<PERSON>` exactly where serving text will
  contain a real name — a train/serve skew that no offline metric can see and that the shadow gate (proposal §7)
  would surface only weeks later. Acceptable resolutions, in order of preference: (a) run the same tiers in both
  places; (b) build the training text with the **online** tier set and use the batch/NER tier only for the leak
  audit and for the stored `*_redacted` display columns; (c) train with feature dropout over the NER-only
  placeholders and measure the degradation explicitly. **(b) is my recommendation** — it keeps the model's input
  distribution identical in both worlds while still giving the audit its higher recall. Gate G14 is what catches a
  violation.

### 14.3 Performance requirements to design against

| Requirement | Value | Basis |
|---|---|---|
| Batch build throughput | ≥ 200 tickets/s single-core for stages 2–8 excluding L3 NER | [measured] my unoptimised L1+L2 prototype runs at **0.50 ms/doc, ~2,000 docs/s** single-core on this corpus — 10× the requirement, so the budget is not tight for the deterministic layers |
| **Serving-path budget for stages 2–8** | **≤ 15 ms p95 per ticket**, i.e. ≤ 6% of the 250 ms model-call SLO (model spec §9.3) | Preprocessing must not become the dominant cost of a 43 ms inference |
| L3 NER, if it must run at serve time | Unmeasured. **Blocking question Q7** | Slovnet reports ~25 docs/s CPU; that is ~40 ms/doc, which **exceeds the budget above** and must be resolved before implementation |
| Per-document hard timeout | 20 ms, then quarantine | §4.5 R6 |
| Memory | < 200 MB for L1+L2; L3 adds the NER model (~30 MB Slovnet, ~500 MB for a BERT NER) | [estimate] |
| Build reproducibility | Pinned `regex`/`re`, `tokenizers`, encoder checkpoint hash, NER model version, Python version, all recorded in the card | Model spec §9.5 |

### 14.4 Reproducibility

Immutable input hash → immutable snapshot hash. Every metrics report references the snapshot hash. Seeds fixed and
recorded for: dedup sampling, split tie-breaks, QA sampling, ablation training. Never build from a live query.

---

## 15. Scratch work backing this document

Throwaway, read-only analysis scripts (not deliverables, not to be imported, not to be turned into the pipeline —
the pipeline is specified above and is not written yet):

```
/tmp/claude-0/-home-user-bert-poc/0fb3a48d-e747-55b7-a9f7-f0fd9518d210/scratchpad/
  s1_basic.py       label hygiene, class balance, cardinality
  s2_tokens.py      XLM-R token-length distribution
  s3_spans.py       value-detector counts + token accounting
  s4_fp.py          detector false positives, Luhn/ИНН validators, self-redaction
  s5_moji.py        mojibake and line-numbered paste probes
  s6_blocks.py      structural block detectors + block-kind→label lift
  s7_policy.py      redaction policy variants, token-budget table
  s8_ph.py          placeholder tokenisation cost
  s9_splits.py      split feasibility (temporal / gapped / org-grouped)
  s10_dedup.py      MinHash, cross-split leakage, bursts, boilerplate
  s11_minhash2.py   overflow-safe MinHash, LSH recall
  s12_pilot.py      the §5.6 pilot ablation
  s13_tail.py       rare-tail cue dependence, empty-services characterisation
  s14_lift_time.py  value-span lift, regex timing
  xlmr_tokenizer.json   downloaded XLM-R tokenizer (fixture for measurement only)
```
