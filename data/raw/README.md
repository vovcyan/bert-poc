# `tickets_export.csv` — synthetic raw ticket export

A **fake dump of the production `tickets` table**, shaped like `COPY (SELECT ... FROM tickets ORDER BY id) TO STDOUT WITH CSV HEADER`.
It serves two purposes:

1. a fixture for the week-1 audit and partition queries in
   [`dataset-construction-runbook.md`](../../docs/specs/dataset-construction-runbook.md) §1–§4 and for the class-balance /
   cardinality / RU:EN reports required by [`ticket-services-classifier.md`](../../docs/specs/ticket-services-classifier.md) §2.8;
2. a **plausible-scale stand-in corpus for pipeline work**. At 5,013 rows it sits inside the
   "5,000–20,000 usable human-labelled tickets → fine-tuned encoder, the recommended path" band of spec §2.4 / runbook §5.1,
   so preprocessing, splitting, dedup, training-loop and threshold code can be exercised end to end at realistic volume.
   It also matches the runbook §5.2 arithmetic: ~5,000 labelled tickets at a 10% correction rate implies ~50,000 raw tickets
   scanned. The earlier 310-row version of this file was a query fixture only — any metric computed on it was meaningless.
   Metrics computed on *this* file are still meaningless (see "Known gaps"), but the shape and volume are right.

It is **not** a training set: no split column, no provenance column, no cleaned label column, and no `ticket_messages`
companion — reconstructing provenance (Cases A/B/C/D) requires that table and it does not exist here.

## Provenance of the text (read this before trusting the labels)

**100% synthetic. No real customer data.** Every name, e-mail (`*@example.ru` / `*@example.com`), phone (`+7 900 000-xx-xx`,
`+1 555 01xx`), organisation (ООО «Ромашка-Сервис», ИНН `77000000xx`), card-like run (`4111 11** **** 1111`), URL and token
(`https://hooks.example.ru/ingest?token=…`) is invented and uses reserved example domains and number ranges.

How the text was produced, precisely:

- **310 tickets are individually hand-written** and appear verbatim (the original version of this file).
- The other ~4,700 rows come from a **hand-authored scenario bank of 799 distinct root causes** (e.g. "presigned URL signature
  fails after an endpoint change", "autovacuum cannot keep up", "SCIM creates duplicate users on e-mail case"). Each scenario
  carries 1–3 hand-written phrasings per language plus its own technical evidence (log lines, status codes, error strings).
  A generator then realises each scenario ~6 times (median 6, p90 9, max 15) varying: persona (developer / SRE / ops /
  non-technical admin / accountant / manager / angry customer / newcomer), verbosity (one line to six sentences), politeness,
  which hand-written generic "moves" are attached (greeting, context, impact, ask, sign-off), whether evidence is pasted as a
  block, language, typing noise, and every concrete detail (ids, hosts, buckets, codes, quantities, dates, money, names).
- A **near-duplicate guard** (char-5-gram Jaccard) rejects any realisation that is ≥0.62 similar to an already-accepted row of
  the same scenario, and exact-duplicate texts are dropped globally. The bounded-duplicate property therefore holds by
  construction, not by luck — see the diversity numbers below.

So: the *substance* of every ticket is hand-authored; the *surface* is compositional. This is not 5,013 individually
hand-written tickets and should not be described as such.

## Columns

`id` text `TICKET-\d+`, ascending with gaps · `organization_id` text `org_0001`–`org_0210` · `title` text · `description` text ·
`priority` ∈ {low, normal, high, urgent} · `services` text, **multi-label target**, `,`-separated (RFC4180-quoted) ·
`labels` text, open vocabulary, `;`-separated · `flags` text, `;`-separated ∈ {escalated, vip, reopened, sla_breach, duplicate} ·
`created_at`, `updated_at` ISO-8601 with offset, `updated_at >= created_at`. UTF-8, LF, header row, minimal quoting.

## Service taxonomy (20, kebab-case)

`access-control` roles, permissions, membership, offboarding · `api-gateway` public API edge: status codes, limits, keys, SDKs ·
`auth` login, SSO/SAML, 2FA, sessions, tokens · `backups` snapshots, retention, restore/PITR · `billing` invoices, charges,
refunds, accounting documents · `cdn` edge cache, invalidation, custom domains · `compute` VMs: boot, resize, disks, OOM,
autoscaling · `console-ui` the web console itself · `dns` zones and records · `integrations` third-party connectors and outbound
webhooks · `logging` log ingest, search, retention · `managed-postgres` hosted PostgreSQL · `managed-redis` hosted Redis/cache ·
`message-queue` managed broker · `monitoring` metrics, dashboards, alert rules · `networking` VPC, firewall, load balancer, VPN,
inter-zone traffic · `notifications` platform-sent e-mail/SMS/messenger delivery · `object-storage` S3-compatible buckets and
objects · `subscriptions` plans, seats, quotas, trials · `terraform-provider` the IaC provider.

**Deliberate ambiguity:** `auth` vs `access-control` (authentication vs authorisation — annotators will argue over "403 for one
user", "revoke a leaver's key", "key scope rejected"), and secondarily `monitoring` vs `logging`, `notifications` vs
`integrations`, `cdn` vs `networking`, `console-ui` vs the component whose page is broken. §6.5 / the taxonomy-ambiguity risk
needs something to bite on.

## Shape

5,013 rows · 210 organisations, Zipf-ish (org_0001 364 tickets, org_0002 236, org_0003 159, median 15, every org has ≥3
tickets) · `created_at` 2025-06-10 → 2026-08-09 (15 calendar months, 84–428 tickets/month) · 90.9% of tickets 08:00–19:59
local, 232 on weekends · 11 timezone offsets · priority normal 3,177 / high 1,156 / urgent 325 / low 355.

Language (as authored): **RU 3,347 (66.8%), EN 1,429 (28.5%), code-switched 237 (4.7%)** — code-switched = Russian prose with
English error strings and log pastes. A Cyrillic-ratio heuristic classifies only 9 of those 237 as "mixed", which is itself a
finding: the §6.3 language slice cannot be built with a naive script-ratio rule.

Labels: 8,771 positives over 4,622 labelled rows, **mean cardinality 1.90** (|S|=1: 1,569, 2: 1,988, 3: 1,034, 4: 31), plus
**391 rows with empty `services`** (untriaged — §2.2's `cardinality(...) = 0` check). Mean is below the ~2.2 the runbook
assumes: labels were only assigned where the ticket text itself references the component, and no padding was applied.
Head: billing 851, console-ui 793, auth 762, api-gateway 751, compute 715, access-control 667, object-storage 645.

**The rare tail is deliberately NOT scaled proportionally.** Proportional scaling from the 310-row version would have put every
service above the 50-positive line in §2.8 and destroyed the "services that cannot be learned or evaluated reliably" property.
Instead: **terraform-provider 15, message-queue 31, cdn 38, dns 41** — four services below 50 absolute positives, one below 20 —
with `managed-redis` (98) just above. These are held down by weighting, not by lack of scenarios (7–19 distinct scenarios each).

Co-occurrence is structured, not independent: auth×api-gateway, billing×subscriptions, object-storage×cdn, monitoring×compute,
backups×managed-postgres, access-control×auth, console-ui×the broken component.

## Diversity (acceptance numbers)

| Metric | Value |
|---|---|
| distinct titles (exact) | 3,905 / 5,013 = **0.779** |
| distinct descriptions (exact) | 5,007 / 5,013 = **0.999** (6 duplicate descriptions under different titles; 0 duplicate title+description pairs) |
| char-5-gram Jaccard ≥ 0.80 | 560 pairs, **1 unintended pair** (2 rows) — the rest are the intentional alert-template clique and copy-paste clusters; largest intentional cluster 25 rows |
| char-5-gram Jaccard ≥ 0.60 | 5,074 pairs total; **136 unintended** (excluding intentional clusters and incident bursts), covering 254 rows (5.1%), **largest unintended cluster 5 rows** |
| RU + code-switched descriptions | 99,938 tokens, 5,397 distinct, TTR 0.054, 1,099 hapax |
| EN descriptions | 49,380 tokens, 2,320 distinct, TTR 0.047, 600 hapax |
| repeated boilerplate | 135 sentences occur ≥20× each and account for **47.9% of all sentence instances** — the generic "moves" (`Мешает работать, но не блокирует.`, `Do not reply to this message…`). Scenario-specific sentences carry the signal; this filler is the compositional seam, and it is visible in aggregate even though individual rows read naturally |
| distinct scenarios per service | billing 141, console-ui 137, auth 119, api-gateway 111, access-control 109, compute 107, object-storage 104, subscriptions 92, monitoring 80, networking 79, notifications 69, managed-postgres 68, logging 55, integrations 52, backups 43, managed-redis 19, dns 18, cdn 17, message-queue 12, terraform-provider 7 |

## Deliberate dirt (it is a raw export — this is intentional)

- **Service hygiene** (the reason `norm_services()` exists): 49 rows with a space around the comma (`"logging, subscriptions"`),
  38 with mixed case (`"Billing"`), 40 with an internal duplicate (`"logging,logging"`).
- **Text**: 374 descriptions with embedded newlines (log/traceback/e-mail-quote pastes), 246 rows with embedded `"` (escaped
  `""`), 341 with `;` and ~4,100 with `,` inside prose, 119 ALL-CAPS titles, 337 lowercase-initial titles, 207 rows with
  leading/trailing whitespace, plus character-level typos (doubled/transposed letters, dropped commas, `ё`→`е`) on ~15% of rows.
  Description length 22–793 chars (median 186); 262 tickets under 80 chars.
- **PII to pseudonymise (§2.9)**: 265 e-mail addresses, 179 phone numbers, 76 card-like digit runs, 210 URLs of which 108 carry
  a token or key, plus several hundred invented personal names, legal-entity names and INNs.
- **Near-duplicates (§4.3)**: 99 auto-generated monitoring-alert tickets sharing one template across 60 organisations (the
  ≥0.80 clique), 72 rows in 26 copy-paste refile clusters where one customer re-filed the same ticket with a "пишу повторно"
  line appended, plus the legacy `docs-org11` / `key403-org19` / `pgpool-org7` clusters. Semantic-but-not-lexical clusters
  (e.g. the Postgres connection-pool cluster) score 0.2–0.5 and MinHash will *miss* them — that is the point.
- **Incident bursts**: 8 bursts (358 tickets), the largest 109 on 2025-11-18 (API-gateway 503 + token-validation outage) across
  ~90 organisations in a few hours, then 2026-03-05 storage write failure (99), 2026-06-23 zone-b network degradation (74),
  2025-09-09 authentication outage (56), 2026-04-14 database failover (42), 2026-01-21 console outage (40), 2026-05-27 mail
  delay, 2025-12-03 DNS. Burst tickets are *not* near-duplicates of each other: different customers describe the same root cause in
  their own words. They exist so the ≥1-week split gap in §6 has something to protect against.
- **Abstention cases**: ~100 genuinely unactionable tickets (`не работает`, `help pls`, `тест`, a forwarded mail thread),
  most with empty `services`. Row sources overall: 310 legacy hand-written, 358 burst, 99 auto-alert, 72 copy-paste refile,
  ~4,170 scenario-bank.

## Known gaps before you rely on it

- No `ticket_messages` rows, so §1/§2.1 provenance reconstruction cannot be exercised end to end — only the `tickets` side.
- `labels`/`flags` are as-of-export with no creation-time history, so §2.7's target-leakage question is unanswerable here.
- Labels are author-assigned, not blind-annotated: there is no inter-annotator agreement, so no α, and the file cannot stand in
  for the gold set (2,000 test + 1,000–1,500 val, runbook §4.2). Any accuracy number measured on it describes the generator,
  not the world.
- Surface realisation is compositional (see "Provenance of the text"), so lexical statistics — TTR, n-gram entropy, vocabulary
  growth — are lower than a real corpus of this size would show. Structural statistics (label balance, cardinality, timing,
  duplication) are the ones that transfer.
