# `tickets_export.csv` — synthetic raw ticket export

A **fake dump of the production `tickets` table**, shaped like `COPY (SELECT ... FROM tickets ORDER BY id) TO STDOUT WITH CSV HEADER`.
It is a fixture for the week-1 audit and partition queries in
[`docs/specs/dataset-construction-runbook.md`](../../docs/specs/dataset-construction-runbook.md) §1–§4 and for the
class-balance / cardinality / RU:EN reports required by [`ticket-services-classifier.md`](../../docs/specs/ticket-services-classifier.md) §2.8.
It is **not** a training set: there is no split column, no provenance column, no cleaned label column, and no `ticket_messages`
companion file — reconstructing provenance (Cases A/B/C/D) needs that table and it does not exist here.

**100% synthetic.** Every ticket was written for this fixture. All names, e-mails (`*@example.ru` / `*@example.com`), phone numbers
(`+7 900 000-xx-xx`, `+1 555 0100`), organisations (ООО «Ромашка-Сервис», ИНН `7700000001`), card-like digit runs
(`4111 11** **** 1111`), URLs and tokens (`https://hooks.example.ru/ingest?token=wh_9f3ac2b1e4`) are invented and use
reserved example domains/number ranges. No real customer, person, company or credential appears. Source: hand-authored;
organisation ids, timestamps and ticket ids assigned by a seeded script (seed 20260816).

## Columns

`id` text `TICKET-\d+`, sequential with gaps · `organization_id` text `org_00\d\d` · `title` text · `description` text ·
`priority` text ∈ {low, normal, high, urgent} · `services` text, **multi-label target**, `,`-separated (RFC4180-quoted) ·
`labels` text, open vocabulary, `;`-separated · `flags` text, `;`-separated ∈ {escalated, vip, reopened, sla_breach, duplicate} ·
`created_at`, `updated_at` ISO-8601 with offset (`+03:00` mostly; 9 offsets in total), `updated_at >= created_at`.
UTF-8, LF, header row, minimal quoting.

## Service taxonomy (20, kebab-case)

`access-control` roles, permissions, membership, offboarding · `api-gateway` public API edge: status codes, rate limits, keys, SDKs ·
`auth` login, SSO/SAML, 2FA, sessions, tokens · `backups` snapshots, retention, restore/PITR · `billing` invoices, charges, refunds, documents ·
`cdn` edge cache, invalidation, custom domains · `compute` VMs: boot, resize, disks, OOM, autoscaling · `console-ui` the web console itself ·
`dns` zones and records · `integrations` third-party connectors and outbound webhooks · `logging` log ingest, search, retention ·
`managed-postgres` hosted PostgreSQL · `managed-redis` hosted Redis/cache · `message-queue` managed broker · `monitoring` metrics, dashboards, alert rules ·
`networking` VPC, firewall, load balancer, VPN, inter-zone traffic · `notifications` platform-sent e-mail/SMS/messenger delivery ·
`object-storage` S3-compatible buckets and objects · `subscriptions` plans, seats, quotas, trials · `terraform-provider` the IaC provider.

**Deliberate ambiguity:** `auth` vs `access-control` (authentication vs authorisation — annotators will argue over "403 for one user"
and "revoke a leaver's key"), and secondarily `monitoring` vs `logging`, `notifications` vs `integrations`, `cdn` vs `networking`.
This is what the §6.5 / taxonomy-ambiguity risk needs to bite on.

## Shape

310 rows · 41 organisations, Zipf-ish (org_0001 43 tickets, org_0002 31, tail of 4 orgs with a single ticket) ·
`created_at` 2025-06-10 → 2026-08-07 (15 calendar months), 86.8% of tickets 08:00–19:59 local, 15 of 310 on a weekend ·
priority normal 221 / high 54 / low 21 / urgent 14.

Language (as authored): **RU 205 (66.1%), EN 92 (29.7%), code-switched 13 (4.2%)** — code-switched = Russian prose with English
error strings/log pastes. A script-ratio heuristic classifies only 4 of those 13 as "mixed", which is itself useful: the §6.3
language slice cannot be built with a naive Cyrillic-ratio rule.

Labels: 591 positives, 288 labelled rows, **mean cardinality 2.05** (|S|=1: 45, 2: 191, 3: 44, 4: 8), plus **22 rows with empty
`services`** (untriaged — §2.2's `cardinality(...) = 0` check). Mean is a little below the ~2.2 the runbook assumes; every label
here is defensible from the ticket text, and I did not pad to hit the number. Head: console-ui 63, api-gateway 62, billing 60,
object-storage 53, auth 44, compute 41, access-control 40. Rare tail, deliberately unlearnable: **cdn 8, dns 8, managed-redis 7,
message-queue 6, terraform-provider 5** — all far below the "50 positives" line in §2.8; `backups` 16, `logging` 16 and
`integrations` 18 sit just above it. Co-occurrence is structured, not independent (auth×api-gateway, billing×subscriptions,
object-storage×cdn, monitoring×compute, backups×managed-postgres).

## Deliberate dirt (it is a raw export — this is intentional)

- **Service hygiene** (the reason `norm_services()` exists): 5 rows with a space after the comma (`"logging, subscriptions"`),
  2 rows with mixed case (`"Billing"`, `"Auth,console-ui"`), 3 rows with an internal duplicate (`"logging,logging"`).
- **Text**: 23 descriptions with embedded newlines (log/traceback/e-mail-quote pastes), 58 rows with embedded `"` (escaped `""`),
  16 with `;` and 272 with `,` inside prose, 5 ALL-CAPS titles, 11 rows with leading/trailing whitespace, 13 lowercase-initial
  titles with typos and no punctuation (`не работает`, `смс-уведомления не приходят на мтс`), plus `«Востановить»` /
  `резервая копия` misspellings inside the text.
  Description length 31–785 chars (median 223); 5 tickets under 80 chars.
- **PII to pseudonymise (§2.9)**: 21 e-mail addresses, 11 phone numbers, 3 card-like digit runs, 12 URLs, 4 of them carrying a token or key,
  ~15 personal names.
- **Near-duplicates (§4.3)**: 3 pairs at char-5-gram Jaccard ≥ 0.8 — two are one customer re-filing a copy-pasted ticket
  (org_0011 closing documents, org_0019 `403 key_scope`), one is cross-org. 7 auto-generated monitoring-alert tickets share an
  identical template across 7 different organisations and form a clique at Jaccard 0.70–0.85. Two further clusters
  (org_0007 Postgres connection-pool ×4, org_0011 documents ×4) are semantically near-identical but *lexically* varied and score
  0.2–0.5 — i.e. MinHash will miss them, which is the point.
- **Incident bursts**: 2025-11-18 (11 tickets, 9 orgs, API-gateway 503 + token-validation outage), 2026-03-05 (9 tickets, 8 orgs,
  storage write failures), 2026-06-23 (8 tickets, 8 orgs, zone-b network degradation). These straddle nothing by themselves —
  they exist so the ≥1-week split gap in §6 has something to protect against.
- **Abstention cases**: 8 tickets that are genuinely unactionable (`не работает`, `help pls`, `тест`, a forwarded mail thread).

## Known gaps before you rely on it

No `ticket_messages` rows, so §1/§2.1 provenance reconstruction cannot be exercised end-to-end — only the shape of `tickets`.
`labels`/`flags` are populated as of export, with no creation-time history (§2.7 target-leakage question is unanswerable here).
310 rows is a PR-reviewable fixture, not an annotation volume: it exercises every §1–§4 query, but any metric computed on it is
meaningless — the real gold set is 2,000 test + 1,000–1,500 val (§4.2).
