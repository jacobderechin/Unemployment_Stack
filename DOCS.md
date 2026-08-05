# Unemployment Stack — details

Setup and install live in [README.md](README.md). This is everything else.

## Board behaviour

`status` is one of `applied`, `interviewing`, `offer`, `rejected`. Moving into
`interviewing`/`offer`/`rejected` auto-stamps `interview_date`/`offer_date`/
`rejected_date` with today's date, only the first time. Each stamped date shows
as an editable field with the day-count from your application beside it; cards
still in **Applied** show how long they've been waiting.

The **✎** button on a card opens a dialog to edit company, title and applied
date, and to delete a milestone date that was stamped by mistake.

## HTTP API

JSON over HTTP, for the curious or for scripting.

| Method   | Path             | Body                             | Does                         |
|----------|------------------|----------------------------------|------------------------------|
| `GET`    | `/api/jobs`      | —                                | list all jobs                |
| `POST`   | `/api/jobs`      | `{company, title, applied_date}` | add a job (status `applied`) |
| `PATCH`  | `/api/jobs/{id}` | any subset of the job's fields   | move column / edit fields    |
| `DELETE` | `/api/jobs/{id}` | —                                | remove a job                 |

PATCH a date field with a `YYYY-MM-DD` string, or `""` to clear it.

Gmail and postings:

| Method | Path                     | Body                        | Does                                       |
|--------|--------------------------|-----------------------------|--------------------------------------------|
| `GET`  | `/api/sync`              | —                           | last sync timestamp                        |
| `POST` | `/api/sync`              | —                           | run `ingest.py --commit` since that mark    |
| `POST` | `/api/auth`              | —                           | refresh `token.json`, or open the consent tab |
| `POST` | `/api/research/scan`     | —                           | run `research.py` over the saved profile   |
| `POST` | `/api/research/check`    | `{ids, mode}`               | `enrich.py` over those ids (`full`/`alive`/`new`) |
| `GET`  | `/api/research/matches`  | —                           | postings matching the saved profile        |
| `GET`  | `/api/research/matches/all` | —                        | every live posting, profile ignored        |
| `DELETE` | `/api/research/matches/{id}` | —                     | drop a posting                             |
| `GET`/`PATCH` | `/api/research/profile` | titles / locations / levels | the saved search profile            |

Resume:

| Method | Path                  | Body                              | Does                                      |
|--------|-----------------------|-----------------------------------|-------------------------------------------|
| `POST` | `/api/resume/extract` | raw file bytes, `X-Filename:` hdr | file → text; stores nothing               |
| `PUT`  | `/api/resume`         | `{text, filename}`                | split, extract skills, embed, save        |
| `GET`  | `/api/resume`         | —                                 | the saved resume                          |
| `GET`  | `/api/resume/matches` | —                                 | every live posting, scored, with features |
| `POST` | `/api/embed/backfill` | —                                 | run `embed.py` over un-embedded postings  |

The upload is raw bytes plus an `X-Filename` header rather than a multipart
form: Python 3.13 removed the `cgi` module, so there is no stdlib multipart
parser, and a header needs no parsing at all. Only `.pdf`, `.txt` and `.md` are
accepted, capped at 2 MB, and the size is refused from `Content-Length` before
the body is read.

## Email ingestion

`ingest.py` scans Gmail for application confirmations, interview invites and
rejections and proposes tracker entries, classified and extracted by a local
LLM via Ollama. It previews by default and only writes with `--commit`.

```bash
python3 ingest.py                # preview proposed adds/updates (writes nothing)
python3 ingest.py --commit       # apply them
python3 ingest.py --since 60     # scan the last 60 days (default 30)
python3 ingest.py --workers 8    # more concurrent classify requests (default 4)
OLLAMA_MODEL=qwen3:14b python3 ingest.py   # a different local model
BODY_CHARS=4000 python3 ingest.py          # more of each body to the model (default 2000)
```

The header's **Sync** button runs `--commit` from the last sync watermark.

Matching is by company **and** job title, so several roles at one company stay
separate. Emails that don't name a role get the title `(role not specified)` —
fix it with the card's ✎ button. A status email it can't pin to a single role is
flagged for you instead of guessed. Processed emails are remembered (a
`seen_emails` table) and skipped on reruns.

**Speed:** classification runs `--workers` emails concurrently, but that only
helps if Ollama is started with `OLLAMA_NUM_PARALLEL>1` (default 1 just queues
them). Set it via a systemd drop-in (`Environment="OLLAMA_NUM_PARALLEL=8"`) and
restart Ollama. Lowering `BODY_CHARS` also speeds each request. Spam/Trash and
any senders listed in `IGNORE_SENDERS` (top of `ingest.py`) are skipped before
fetching.

### Gmail troubleshooting

| Symptom                                     | Fix                                                                        |
|---------------------------------------------|----------------------------------------------------------------------------|
| `Missing credentials.json — see the README` | Step 4: the download has a long generated name — rename it to exactly `credentials.json`, next to `app.py` |
| Consent screen ends in `access_denied`      | Step 3: your address isn't in **Test users**                                |
| Sync worked, then stopped after a week      | Expected while publishing status is **Testing** — click **Reconnect Gmail** |
| Headless / remote machine                   | The sign-in needs a local browser; do step 5 on a desktop and copy `token.json` over |
| Wrong Google account                        | `rm token.json`, run `python3 ingest.py`, pick the other account            |

The requested scope is `gmail.readonly` and nothing else — the app cannot send,
delete or modify mail.

## Scanning for postings

The Job search tab's **Scan** button runs `research.py`, which logs every
posting matching the profile you saved there (titles / locations / levels) into
`research.db`. Two sources, one table:

- the gzipped chunks of
  [Feashliaa/job-board-data](https://github.com/Feashliaa/job-board-aggregator/) — the
  open web, ~1.4M rows, downloaded chunk by chunk;
- the [a16z Speedrun Talent
  Network](https://speedrun-talent-network.com/developers) — all 10k open roles
  in the portfolio, pulled as one 2MB markdown export (`/jobs.md`).

A role listed in both is stored once, on `(company, title)`, keeping the chunk
feed's link — that one points at the employer's own board rather than at a
listing page. Speedrun quotes the employer's salary band, so those rows arrive
with a salary already filled in; the chunk feed's figure is an estimate, so it's
left blank there for `enrich.py` to read off the posting itself. Seniority isn't
in the export and is read off the job title, the same way the network's own API
derives it.

```bash
python3 research.py       # same scan from the shell
```

## Checking postings

**Check postings** visits every posting currently shown in the table — the title
filter and seniority checkboxes decide which — and asks: is this still open? A
dead posting is hidden from the tab (`alive = 0`; the row stays on disk, so a
re-check can bring it back). A live one gets its salary, minimum years of
experience, skills and full description pulled out by the local model.

Liveness is not a status-code check. Of the boards in the feed only Lever 404s a
dead posting: Ashby answers 200 with an empty JavaScript shell, and Greenhouse
and company-hosted boards 200-redirect you to their index. `enrich.py`'s
`parse()` documents the rule. Ashby and Lever embed a schema.org `JobPosting` in
the page, so a JavaScript-rendered posting is readable without a browser.

Two model calls per live posting — the field extractor and the core-span extractor (see
[Embeddings](#embeddings)) — so a full run takes roughly twenty seconds a job. Filter the
table down first. Two cheaper buttons sit next to it:

- **Check alive** — same liveness pass, no model calls at all. Only the `alive`
  flag is written, so existing salary/years/skills survive. One http request per
  posting, 32 at a time: hundreds of rows in the time a full check does a handful.
- **Check new** — visits every posting shown, exactly like a full check, but only
  sends the ones with no enrichment yet (`years_experience IS NULL`) to the model.
  Dead postings are still pruned; you just pay the model for what the last scan
  added. Rows a previous run failed on or never reached count as un-enriched.
  The marker is `years_experience`, not `checked_at` — the latter is stamped by
  every visit, including an alive-only one.

From the shell: `python3 enrich.py --ids 3,7`, `--mode alive`, `--mode new`, or
`python3 enrich.py` with no arguments to check every live row.


## Resume analysis

Upload a PDF (or paste text), confirm the years of experience and job titles it
extracted, and you get two things: which of the skills each job title asks for
you already have, and a ranked list of the postings that fit you best.

### How a posting is scored

Not a vector search. Five features, each normalized to 0–1, combined as a weighted mean:

| Feature    | Weight | What it measures                                              |
|------------|--------|---------------------------------------------------------------|
| `skills`   | 0.32   | share of the posting's required skills your resume has        |
| `semantic` | 0.27   | cosine similarity to your resume                              |
| `years`    | 0.17   | your experience against the posting's stated minimum          |
| `level`    | 0.12   | seniority against your profile                                |
| `recency`  | 0.12   | linear decay over 60 days                                     |


The weights above are the server's starting point. In the tab each feature is an
Off / Low / Normal / High chip (weights 0 / 1 / 2 / 4), stored in `localStorage` under
`jt.resume.importance`. The mean is normalized by the weight total, so only the *balance*
between features changes the ranking — all five on High is the same ranking as all five on
Normal, which is why these are four discrete steps and not five 0–100 sliders.

The chips re-rank in the browser with no refetch — every row arrives with its
feature values, so the server scores once per tab visit. The **Why** column shows each
row's breakdown, which is the reason for a weighted mean rather than reciprocal rank
fusion: RRF is scale-free and robust, but it converts scores to ranks and throws away
the magnitudes the column displays.

A feature shown as **—** was never extracted for that posting. It scores as the mean of
that feature across the result set, so a gap in the scraped data is neither rewarded nor
punished. Neither 0 nor 1 is right, and neither is dropping the term and renormalizing:
the features are not equally generous (`level` averages 0.99, `skills` 0.19),
so dropping the skills term for the postings that have none floated the rows we knew
*least* about into all of the top 50. Run **Check new** in the Job search tab to fill
more in.

Two toggles hard-filter instead of scoring: hide roles needing more years than you have,
and hide levels outside your profile. Both default off, because posted minimums are
routinely inflated. A posting whose requirement was never stated always survives them —
`years_experience = 0` means "not stated", not "needs zero years".

### Embeddings

`market_matches.description` does not hold the posting. It holds the part of the posting
worth matching a resume against — the responsibilities-and-requirements span — and
`embed.py` embeds that whole (`Qwen3-Embedding-0.6B`, 32k-token context) into packed
float32 in `market_matches.embedding`.

### Locations

The feed writes a location however the employer typed it — `Hybrid (NYC Metro)`, `Strava SF`,
`Bay Area`, `San Francisco, CA, US; Remote, US`. `locations.py` canonicalises each **distinct**
string with the local model into a comma-joined city list in `location_map`, and
`list_matches()` joins it in as `cities`. 

One posting can name several cities, and the Salary tab counts it in each — so those bucket
`n`s sum to more than the posting count, deliberately. `app.py` runs the mapper at the end of
every scan. With Ollama off it exits cleanly and the browser falls back to its own metro list
(`METROS` in `index.html`), which only knows the cities written into it.

A mapping you disagree with is one row: `UPDATE location_map SET cities = 'Buffalo' WHERE raw =
'Grand Island, New York, USA'` and every posting follows. Nothing is denormalised onto
`market_matches`. Use `python3 locations.py --all` only when the prompt itself changed.

### Known limits

Skills are matched as exact strings, so spelling variants miss — a resume saying
"Natural Language Processing" does not match a posting saying "NLP". The extraction
prompt asks for umbrella terms alongside specific tools ("Machine Learning" as well as
"PyTorch"), which covers the common case; the `semantic` feature covers the rest.

## Tests

```bash
python3 test_app.py       # prints "ok" — database logic + validation rules
python3 test_ingest.py    # prints "ok" — email reconcile logic (no network)
python3 test_resume.py    # prints "ok" — resume scoring, extraction guards, blob round-trip
pytest test_research.py   # profile-filter predicate
pytest test_locations.py  # location canonicalisation, no Ollama needed
pytest test_enrich.py     # liveness rule, against captured pages from each board
```

Open <http://127.0.0.1:8000/#selftest> and read the console for the front-end tests.
`resumeSelfTest()` checks that the browser's `resScore()` reproduces `resume.combine()`
exactly — if those two drift, changing a chip would re-sort into a different order
than the server sent.

None of them need Gmail credentials, Ollama or the network: `test_ingest.py`
exercises the add/update/skip/flag decisions and `test_enrich.py` the alive/dead
rule, both against in-memory databases and saved page fixtures.
