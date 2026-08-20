#!/usr/bin/env python3
"""Scan Gmail for job-application emails and propose tracker entries.

Reads recent mail (Gmail API, read-only), classifies/extracts each with a local
Ollama model, and reconciles against tracker.db. Previews by default; pass
--commit to write. Nothing leaves the machine except the read-only Gmail fetch.

Setup (one-time): see the README "Email ingestion" section. Needs a
credentials.json (OAuth desktop client) in this folder and Ollama running.
"""
import argparse
import base64
import html
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

from tqdm import tqdm

import app  # reuse DB helpers; importing app does not start the server

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:35b")
SINCE_DAYS = int(os.environ.get("SINCE_DAYS", "30"))
BODY_CHARS = int(os.environ.get("BODY_CHARS", "2000"))   # chars of body sent to the model
WORKERS = int(os.environ.get("WORKERS", "8"))            # concurrent classify requests; match OLLAMA_NUM_PARALLEL
BATCH_SIZE = int(os.environ.get("GMAIL_BATCH", "20"))    # gmail msg.get per batch; lower if 429s
DEFAULT_TITLE = "(role not specified)"

# Senders to always skip (newsletters etc. that trip the keyword gate below).
# Case-insensitive substring of the From header: a full address
# (jobs-noreply@linkedin.com) or just a domain (indeed.com). Edit this list directly.
IGNORE_SENDERS = ['linkedin.com','indeed.com']

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDS = app.HERE / "credentials.json"
TOKEN = app.HERE / "token.json"

# cheap gate so the LLM only sees plausible mail (also narrows the Gmail query)
KEYWORDS = ("apply", "interview", "unfortunately", "regret", "offer", "position",
            "candidate", "recruit", "hiring", "thank you for your")

KIND_STATUS = {
    "new_application": "applied",
    "interview": "interviewing",
    "offer": "offer",
    "rejection": "rejected",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "is_application": {"type": "boolean"},
        "kind": {"type": "string",
                 "enum": ["new_application", "interview", "offer", "rejection", "other"]},
        "company": {"type": "string"},
        "title": {"type": "string"},
        "event_date": {"type": "string"},
    },
    "required": ["is_application", "kind", "company"],
}

SYSTEM = """You classify job-search emails for a personal application tracker.
Given one email, decide whether it concerns the recipient's own job application
and extract details.

kind:
- new_application: confirmation that the recipient submitted/applied to a role.
- interview: an invitation to interview, or scheduling of one.
- offer: a job offer.
- rejection: a rejection ("not moving forward", "other candidates", etc.).
- other: anything else (job alerts, newsletters, recruiter cold outreach, account notices).

Rules:
- company: the hiring company, NOT the job board / ATS (Greenhouse, Lever, Workday, etc.).
- title: the specific role title if the email states one; if it does NOT name a role, leave it "".
- event_date: the date the email refers to (else the email's own date) as YYYY-MM-DD.
- If it is not about the recipient's own application, set is_application=false and kind="other".
Return only the JSON object."""


# --- LLM -------------------------------------------------------------------

def classify(email, model=None):
    """Ask the local model to classify and extract one email.

    Inputs:  email — a parsed message dict from _parse_message.
             model — the Ollama model; None uses OLLAMA_MODEL.
    Returns: the parsed dict: is_application, kind, company, and optionally title
             and event_date.
    Raises:  urllib.error.URLError if Ollama is unreachable, json.JSONDecodeError if
             the reply is not the object the schema asked for.
    Notes:   The email's date is passed as a local YYYY-MM-DD rather than the raw
             header. Any stray <think> block is stripped before parsing.
    Used by: main(), inside the thread pool.
    """
    user = (f"From: {email['from']}\nDate: {_email_date_iso(email)}\n"
            f"Subject: {email['subject']}\n\n{email['body']}")
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "format": SCHEMA,
        "think": False,
        "stream": False,
        # ponytail: num_gpu=99 pins every layer to the GPU. Ollama's auto-fit
        # undercounts this APU's 96G VRAM, offloads ~20/66 layers and spills the
        # rest (plus an 11G KV cache) into 31G of host RAM, OOM-killing the service.
        # Drop it if a model ever genuinely exceeds VRAM — 99 fails rather than spills.
        "options": {"temperature": 0.6, "top_p": 0.95, "num_gpu": 99},  # Qwen3 thinking-mode defaults
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    content = resp["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return json.loads(content)


# --- reconcile (pure logic; unit-tested without network) -------------------

def _norm(s):
    """Fold a company or title for comparison.

    Inputs:  s — any string, or None.
    Returns: it stripped and lowercased; "" for None.
    Used by: reconcile and dedupe, wherever two names are matched.
    """
    return (s or "").strip().lower()


def _email_date_iso(email):
    """The email's own date, as a local YYYY-MM-DD.

    Inputs:  email — a parsed message dict, whose "date" is the raw header.
    Returns: the date in the local timezone as an ISO string; today's date if the
             header is missing or unparseable.
    Used by: classify (in the prompt) and _pick_date (as the fallback).
    """
    try:
        return parsedate_to_datetime(email["date"]).astimezone().date().isoformat()
    except (TypeError, ValueError):
        return date.today().isoformat()


def _pick_date(x, email):
    """Which date to stamp on the action.

    Inputs:  x — the model's extraction, which may carry an event_date.
             email — the parsed message, for the fallback.
    Returns: the model's event_date when it is a valid YYYY-MM-DD, else the email's
             own date.
    Used by: reconcile.
    """
    ev = (x.get("event_date") or "").strip()
    try:
        date.fromisoformat(ev)
        return ev
    except ValueError:
        return _email_date_iso(email)


def reconcile(x, email, conn):
    """Decide what to do with one extraction.

    Inputs:  x — the model's extraction for this email.
             email — the parsed message it came from.
             conn — an open tracker.db connection, read to find existing cards.
    Returns: an action dict with an 'op' of add | update | skip | flag, carrying
             company/title/status/date where those apply and a 'reason' where one
             is worth printing.

    Matching is by (company, title) so distinct roles at the same company never
    collapse. An email that names no role and matches exactly one card at that company
    updates it; one that matches several is flagged for the user to resolve by hand.

    Used by: main(), on the main thread — sqlite is single-threaded.
    """
    if not x.get("is_application") or x.get("kind") == "other":
        return {"op": "skip", "reason": "not an application"}
    status = KIND_STATUS.get(x.get("kind"))
    if not status:
        return {"op": "skip", "reason": f"unknown kind {x.get('kind')!r}"}
    company = (x.get("company") or "").strip()
    if not company:
        return {"op": "skip", "reason": "no company"}
    title = (x.get("title") or "").strip() or DEFAULT_TITLE
    has_title = title != DEFAULT_TITLE
    when = _pick_date(x, email)

    jobs = app.list_jobs(conn)
    same_company = [j for j in jobs if _norm(j["company"]) == _norm(company)]
    exact = [j for j in same_company if _norm(j["title"]) == _norm(title)]

    if x["kind"] == "new_application":
        if exact:
            return {"op": "skip", "reason": "already tracked",
                    "company": company, "title": title}
        return {"op": "add", "company": company, "title": title,
                "status": "applied", "date": when}

    # interview / offer / rejection
    if has_title:
        if exact:
            return {"op": "update", "job_id": exact[0]["id"], "company": company,
                    "title": title, "status": status, "date": when}
        return {"op": "add", "company": company, "title": title, "status": status,
                "date": when, "reason": "no prior match; creating"}
    if len(same_company) == 1:
        j = same_company[0]
        return {"op": "update", "job_id": j["id"], "company": company,
                "title": j["title"], "status": status, "date": when}
    if len(same_company) > 1:
        return {"op": "flag", "company": company, "title": title, "status": status,
                "date": when, "reason": f"{len(same_company)} roles at {company}; which one?"}
    return {"op": "add", "company": company, "title": title, "status": status,
            "date": when, "reason": "no prior match; creating"}


# status precedence for collapsing duplicates: a role only moves forward
_RANK = {"applied": 0, "interviewing": 1, "offer": 2, "rejected": 2}


def dedupe(actions):
    """Collapse one run's actions down to one card per (company, title).

    Inputs:  actions — the action dicts reconcile produced, in the order they were
             classified.
    Returns: the same list, with the kept action carrying the most-advanced status,
             the earliest applied_date and the advancing email's milestone date. The
             extra emails become 'skip' so they still get marked seen. skip/flag
             actions pass through untouched.

    reconcile() reads the committed DB, so two emails in one run about the same *new*
    role both come back as 'add'; this is what stops them becoming two cards.

    Used by: main(), before the preview is printed.
    """
    kept, out = {}, []
    for a in actions:
        if a["op"] not in ("add", "update"):
            out.append(a)
            continue
        key = (_norm(a["company"]), _norm(a["title"]))
        prev = kept.get(key)
        if prev is None:
            a.setdefault("applied_date", a["date"])
            kept[key] = a
            out.append(a)
            continue
        prev["applied_date"] = min(prev["applied_date"], a["date"])      # earliest application
        if (_RANK[a["status"]], a["date"]) > (_RANK[prev["status"]], prev["date"]):
            prev["status"], prev["date"] = a["status"], a["date"]        # advance / latest wins
        a["op"], a["reason"] = "skip", f"duplicate of {a['company']} — {a['title']}"
        out.append(a)
    return out


def apply_action(a, conn):
    """Execute one add/update action against the tracker.

    Inputs:  a — an action dict from reconcile/dedupe.
             conn — an open tracker.db connection.
    Returns: the affected job id, or None for a skip/flag (which are no-ops).
    Used by: main(), only under --commit.
    """
    if a["op"] == "add":
        job = app.create_job(conn, a["company"], a["title"], a.get("applied_date", a["date"]))
        if a["status"] != "applied":
            _set_status(conn, job["id"], a["status"], a["date"])
        return job["id"]
    if a["op"] == "update":
        _set_status(conn, a["job_id"], a["status"], a["date"])
        return a["job_id"]
    return None


def _set_status(conn, job_id, status, when):
    """Move one card to a status, stamping the email's date rather than today's.

    Inputs:  conn — an open tracker.db connection.
             job_id — the card to move.
             status — one of app.STATUSES.
             when — the date to write into that status's milestone column.
    Returns: None.
    Used by: apply_action.
    """
    fields = {"status": status}
    col = app.DATE_FIELDS.get(status)
    if col:
        fields[col] = when
    app.update_job(conn, job_id, fields)


# --- gmail -----------------------------------------------------------------

def gmail_service():
    """An authorized Gmail API client, refreshing or obtaining consent as needed.

    Inputs:  none — reads token.json and credentials.json beside this file.
    Returns: the built googleapiclient service, read-only scope.
    Raises:  SystemExit if credentials.json is missing.
    Side effect: writes token.json, and opens a browser consent tab when the refresh
             token is dead or absent.
    Notes:   The google libraries are imported lazily so reconcile/tests run without
             them installed.
    Used by: fetch_recent, and POST /api/auth to re-consent from the UI.
    """
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES) if TOKEN.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = None            # refresh token revoked/expired -> full consent below
        if not creds or not creds.valid:
            if not CREDS.exists():
                raise SystemExit(f"Missing {CREDS.name} — see the README 'Email ingestion' setup.")
            creds = InstalledAppFlow.from_client_secrets_file(str(CREDS), SCOPES).run_local_server(port=0)
        TOKEN.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def _retry_batches(pending, run_batch, sleep=time.sleep, attempts=6):
    """Drive a batch fetch until nothing is left to retry.

    Inputs:  pending — the message ids to fetch.
             run_batch — a callable taking the pending ids and returning the ones
             that hit a retryable error.
             sleep — injected for the tests.
             attempts — how many passes before giving up.
    Returns: the ids still unfetched, backing off exponentially (capped at 30s)
             between passes.
    Used by: fetch_recent.
    """
    delay = 1.0
    for _ in range(attempts):
        pending = run_batch(pending)
        if not pending:
            break
        print(f"  rate-limited on {len(pending)} message(s); retrying in {delay:.0f}s …")
        sleep(delay)
        delay = min(delay * 2, 30)
    return pending


def fetch_recent(since_days=SINCE_DAYS, after=None, limit=None):
    """Fetch recent job-shaped mail from Gmail.

    Inputs:  since_days — the rolling window to search, when `after` is not given.
             after — scan from a fixed YYYY-MM-DD instead (Gmail wants YYYY/MM/DD,
             so it is rewritten here).
             limit — stop after this many messages; None for no cap.
    Returns: a list of parsed message dicts from _parse_message.

    The Gmail query does the first pass of filtering: a keyword OR-group, spam/trash/
    sent/promotions excluded, and IGNORE_SENDERS excluded before anything is fetched.
    Bodies come back in batches; ids that hit a 429 or 5xx are retried with backoff and
    anything still failing is reported and skipped.

    Used by: main().
    """
    service = gmail_service()
    kw = " OR ".join(["apply","application", "applied", "interview", "unfortunately",
                      "regret", "offer", "candidate", "recruiter", "position"])
    window = f"after:{after.replace('-', '/')}" if after else f"newer_than:{since_days}d"
    q = f"{window} -in:spam -in:trash -in:sent -category:promotions ({kw})"
    if IGNORE_SENDERS:
        q += " " + " ".join(f"-from:{s}" for s in IGNORE_SENDERS)
    ids, token = [], None
    while True:
        res = service.users().messages().list(
            userId="me", q=q, pageToken=token, maxResults=100).execute()
        ids.extend(m["id"] for m in res.get("messages", []))
        token = res.get("nextPageToken")
        if not token or (limit and len(ids) >= limit):
            break
    ids = ids[:limit] if limit else ids
    out = []

    def run_batch(id_list):
        """Fetch each id in chunks, appending the parsed messages to `out`.

        Inputs:  id_list — the message ids to fetch.
        Returns: the ids that hit a retryable error (429 'too many concurrent', or
                 5xx) so the caller can back off and retry. Anything else is printed
                 and dropped.
        Notes:   Smaller chunks stay under Gmail's per-user concurrency cap; mid is the
                 batch request_id so the callback can name what to retry. Order is
                 irrelevant.
        """
        retry = []

        def _cb(mid, resp, exc):
            """Batch callback: keep the message, or note it for a retry."""
            if exc is None:
                out.append(_parse_message(resp))
                return
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status in (429, 500, 502, 503):
                retry.append(mid)
            else:
                print(f"  ! fetch failed: {exc}")

        for i in range(0, len(id_list), BATCH_SIZE):
            batch = service.new_batch_http_request(callback=_cb)
            for mid in id_list[i:i + BATCH_SIZE]:
                batch.add(service.users().messages().get(userId="me", id=mid, format="full"),
                          request_id=mid)
            batch.execute()
        return retry

    leftover = _retry_batches(ids, run_batch)
    if leftover:
        print(f"  ! {len(leftover)} message(s) still rate-limited after retries; skipped")
    return out


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")  # keep \t \n \r


def _sanitize(text):
    """Strip control and ANSI-escape bytes from untrusted mail.

    Inputs:  text — a header or body string, or None.
    Returns: the same text without control characters, keeping tab/newline/return.
    Notes:   Stops mail driving the terminal (on print) or smuggling escapes into the
             model.
    Used by: _parse_message, on every field except the raw date header.
    """
    return _CTRL.sub("", text or "")


def _parse_message(full):
    """Reshape one Gmail message resource into the dict the rest of this file uses.

    Inputs:  full — a message resource in "full" format.
    Returns: {id, from, subject, date, body}. Everything but `date` is sanitized;
             the date is left raw for parsedate_to_datetime. The body is truncated
             to BODY_CHARS.
    Used by: fetch_recent's batch callback.
    """
    headers = {h["name"].lower(): h["value"] for h in full["payload"].get("headers", [])}
    return {
        "id": full["id"],
        "from": _sanitize(headers.get("from", "")),
        "subject": _sanitize(headers.get("subject", "")),
        "date": headers.get("date", ""),
        "body": _sanitize(_extract_body(full["payload"]))[:BODY_CHARS],
    }


def _extract_body(payload):
    """The message's readable body.

    Inputs:  payload — a Gmail message payload, possibly multipart.
    Returns: the text/plain part if there is one, else the text/html part converted
             to text, else "".
    Used by: _parse_message.
    """
    plain = _collect(payload, "text/plain")
    if not plain:
        body_html = _collect(payload, "text/html")
        plain = _html_to_text(body_html) if body_html else ""
    return plain


def _html_to_text(s):
    """Flatten an HTML mail body to text.

    Inputs:  s — the HTML source.
    Returns: the text with script/style bodies dropped, tags stripped and entities
             unescaped.
    Used by: _extract_body.
    """
    s = re.sub(r"<(script|style)\b.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(s)


def _collect(part, mime):
    """Find the first part of a given MIME type and decode it.

    Inputs:  part — a message payload or sub-part.
             mime — the type to look for, e.g. "text/plain".
    Returns: the decoded body as a string, or "" if no such part exists. Recurses
             into sub-parts.
    Used by: _extract_body.
    """
    if part.get("mimeType") == mime and part.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(part["body"]["data"].encode()).decode("utf-8", "replace")
    for sub in part.get("parts", []) or []:
        got = _collect(sub, mime)
        if got:
            return got
    return ""


def looks_relevant(subject, body):
    """The cheap gate that keeps obviously irrelevant mail away from the model.

    Inputs:  subject, body — the message's text.
    Returns: True if any of KEYWORDS appears in either, case-insensitively.
    Used by: main(), when picking what to classify.
    """
    text = f"{subject}\n{body}".lower()
    return any(k in text for k in KEYWORDS)


def _ignored(sender):
    """Is this sender on the always-skip list?

    Inputs:  sender — the raw From header.
    Returns: True if any IGNORE_SENDERS entry appears in it, case-insensitively.
    Used by: main(), as a second pass behind the Gmail query's own -from: terms.
    """
    s = (sender or "").lower()
    return any(bad and bad.lower() in s for bad in IGNORE_SENDERS)


# --- seen-email bookkeeping ------------------------------------------------

def ensure_seen(conn):
    """Create the seen_emails table if it doesn't exist.

    Inputs:  conn — an open tracker.db connection.
    Returns: None. Commits.
    Used by: main(), at startup.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS seen_emails "
                 "(message_id TEXT PRIMARY KEY, processed_at TEXT)")
    conn.commit()


def is_seen(conn, mid):
    """Has this message already been processed?

    Inputs:  conn — an open tracker.db connection.
             mid — the Gmail message id.
    Returns: True if it is recorded in seen_emails.
    Used by: main(), unless --all is passed.
    """
    return conn.execute("SELECT 1 FROM seen_emails WHERE message_id = ?", (mid,)).fetchone() is not None


def mark_seen(conn, mid):
    """Record that a message has been processed, so a later run skips it.

    Inputs:  conn — an open tracker.db connection.
             mid — the Gmail message id.
    Returns: None. Commits. Re-marking an already-seen id is a no-op.
    Used by: main(), under --commit, for everything except flagged emails.
    """
    conn.execute("INSERT OR IGNORE INTO seen_emails (message_id, processed_at) VALUES (?, ?)",
                 (mid, datetime.now(timezone.utc).isoformat()))
    conn.commit()


# --- cli -------------------------------------------------------------------

def _print_preview(actions):
    """Print the run's proposed changes, grouped by operation.

    Inputs:  actions — the action dicts, each carrying its source email.
    Returns: None; prints ADD, UPDATE, FLAG and skip sections in that order,
             omitting the empty ones.
    Used by: main(), on every run — preview or commit.
    """
    buckets, labels = {}, {"add": "ADD", "update": "UPDATE", "flag": "FLAG (manual)", "skip": "skip"}
    for a in actions:
        buckets.setdefault(a["op"], []).append(a)
    for op in ("add", "update", "flag", "skip"):
        items = buckets.get(op, [])
        if not items:
            continue
        print(f"{labels[op]} ({len(items)}):")
        for a in items:
            if op == "skip":
                print(f"  - {a['email']['subject'][:60]}  [{a.get('reason', '')}]")
            else:
                extra = f"  [{a['reason']}]" if a.get("reason") else ""
                print(f"  - {a['company']} — {a['title']} → {a['status']} ({a['date']}){extra}")
        print()


def main():
    """Command-line entry point: fetch, classify, reconcile, preview, optionally commit.

    Inputs:  none directly — reads sys.argv for --commit, --since, --after, --limit,
             --model, --all and --workers.
    Returns: None. Without --commit nothing is written. With it, every add/update is
             applied, every non-flagged email is marked seen (flagged ones are left
             unseen for manual handling), and meta.last_synced advances to today.

    Classification runs in a thread pool. That only speeds things up if Ollama runs
    with OLLAMA_NUM_PARALLEL>1 — otherwise the requests queue server-side, harmlessly.
    A single email's failure is reported and skipped rather than ending the run.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", action="store_true", help="apply changes (default: preview only)")
    ap.add_argument("--since", type=int, default=SINCE_DAYS, help="days of mail to scan")
    ap.add_argument("--after", help="scan mail on/after this date (YYYY-MM-DD); overrides --since")
    ap.add_argument("--limit", type=int, default=0, help="max emails to fetch (0 = no cap)")
    ap.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model to use")
    ap.add_argument("--all", action="store_true", help="reprocess emails already seen")
    ap.add_argument("--workers", type=int, default=WORKERS, help="concurrent classify requests")
    args = ap.parse_args()

    with closing(app.connect()) as conn:
        app.init_db(conn)
        ensure_seen(conn)

        scope = f"since {args.after}" if args.after else f"from the last {args.since} days"
        print(f"Fetching Gmail {scope} …")
        emails = fetch_recent(args.since, args.after, args.limit or None)
        todo = [em for em in emails
                if (args.all or not is_seen(conn, em["id"]))
                and not _ignored(em["from"])
                and looks_relevant(em["subject"], em["body"])]
        print(f"  {len(emails)} fetched, {len(todo)} to classify with {args.model} …\n")

        actions = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(classify, em, args.model) for em in todo]
            for em, fut in tqdm(list(zip(todo, futures)), desc="Classifying", unit="email"):
                try:
                    x = fut.result()
                except Exception as e:                   # noqa: BLE001 - report and keep going
                    tqdm.write(f"  ! classify failed for {em['subject'][:50]!r}: {e}")
                    continue
                a = reconcile(x, em, conn)               # main thread only: sqlite is single-threaded
                a["email"] = em
                actions.append(a)

        actions = dedupe(actions)
        if actions:
            _print_preview(actions)
        else:
            print("No new job-related emails found.")

        if not args.commit:
            print("Preview only. Re-run with --commit to apply.")
            return

        n = 0
        for a in actions:
            if a["op"] in ("add", "update"):
                apply_action(a, conn)
                n += 1
            if a["op"] != "flag":          # leave flagged emails unseen for manual handling
                mark_seen(conn, a["email"]["id"])
        app.set_meta(conn, "last_synced", date.today().isoformat())  # scanned through today
        print(f"Committed {n} change(s); flagged items left for you to resolve.")


if __name__ == "__main__":
    main()
