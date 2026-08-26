#!/usr/bin/env python3
"""Local job-application tracker: stdlib HTTP helper + SQLite, localhost only.

Run:  python3 app.py   (or double-click run-tracker.sh)
Nothing is exposed to the web — the server binds to 127.0.0.1.

The server is a threaded http.server: long runs live on their own thread
(start_job), so GETs still get answered while one is going.
"""
import json
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import closing
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import enrich   # posting_meta reads title/company/location off a pasted url (one Ollama call)
import research  # market-research feed scan + research.db helpers (self-contained)
import resume    # resume text extraction + the posting scorer (no torch: embed.py is a subprocess)
from enrich import MODES  # enrich still runs as a subprocess; this is just the mode names

HERE = Path(__file__).parent
DB_PATH = HERE / "tracker.db"
INDEX = HERE / "index.html"
HOST, PORT = "127.0.0.1", 8000
STATUSES = ("applied", "interviewing", "offer", "rejected")
# status -> column stamped with the date a card first reaches that status
DATE_FIELDS = {"interviewing": "interview_date", "offer": "offer_date", "rejected": "rejected_date"}
EDITABLE_DATES = set(DATE_FIELDS.values())


# --- database -------------------------------------------------------------

def connect():
    """Open a connection to tracker.db.

    Inputs:  none (DB_PATH is module state).
    Returns: an sqlite3.Connection with a Row factory, a 5s busy timeout to wait
             out the ingest subprocess's writes, and WAL journalling so the app
             can answer GETs while a subprocess writes. WAL sticks in the file
             header, so setting it is a no-op after the first time.
    Used by: every request handler and every run_* helper.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn):
    """Create the tracker schema, and migrate a database made before it grew.

    Inputs:  conn — an open tracker.db connection.
    Returns: None. Creates the jobs and meta tables if absent, and adds any
             milestone date column missing from an older database.
    Used by: the __main__ startup block, and ingest.py before it writes.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
               id             INTEGER PRIMARY KEY AUTOINCREMENT,
               company        TEXT NOT NULL,
               title          TEXT NOT NULL,
               applied_date   TEXT NOT NULL,
               status         TEXT NOT NULL DEFAULT 'applied',
               interview_date TEXT,
               offer_date     TEXT,
               rejected_date  TEXT
           )"""
    )
    have = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    for col in EDITABLE_DATES - have:
        conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()


def get_meta(conn, key, default=None):
    """Read one value out of the meta key/value table.

    Inputs:  conn — an open tracker.db connection.
             key — the meta key, e.g. "last_synced".
             default — what to return when the key is not set.
    Returns: the stored string, or `default`.
    Used by: run_sync and GET /api/sync, to read the ingest watermark.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn, key, value):
    """Write one value into the meta key/value table, inserting or overwriting.

    Inputs:  conn — an open tracker.db connection.
             key, value — the pair to store.
    Returns: None. Commits.
    Used by: ingest.py, to advance the last_synced watermark.
    """
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    conn.commit()


def list_jobs(conn):
    """Every tracked application, newest application first.

    Inputs:  conn — an open tracker.db connection.
    Returns: a list of plain dicts, one per row, ordered by applied_date then id
             descending, so it can go straight to json.dumps.
    Used by: GET /api/jobs, and ingest.reconcile when matching an email.
    """
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY applied_date DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def create_job(conn, company, title, applied_date):
    """Add one application card.

    Inputs:  conn — an open tracker.db connection.
             company, title, applied_date — strings; each is stripped and all
             three are required.
    Returns: the inserted row as a dict.
    Raises:  ValueError if any of the three is blank after stripping.
    Used by: POST /api/jobs, and ingest.apply_action for an "add".
    """
    company = (company or "").strip()
    title = (title or "").strip()
    applied_date = (applied_date or "").strip()
    if not (company and title and applied_date):
        raise ValueError("company, title and applied_date are all required")
    cur = conn.execute(
        "INSERT INTO jobs (company, title, applied_date) VALUES (?, ?, ?)",
        (company, title, applied_date),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_job(conn, job_id, fields):
    """Apply edits from a request body to one card.

    Inputs:  conn — an open tracker.db connection.
             job_id — the row to edit.
             fields — a dict that may carry any of: status, company, title,
             applied_date, and the milestone dates in EDITABLE_DATES. An empty
             string clears a milestone date. Only whitelisted keys are read, so
             the column names interpolated into SQL are never attacker-controlled.
    Returns: True if a row was updated, False if job_id matched nothing.
    Raises:  ValueError on an unknown status, a blank required field, an
             unparseable date, or a body with nothing to update in it.
    Side effect: moving to a status in DATE_FIELDS stamps today's date on that
             milestone the first time it is reached, unless the body sets it.
    Used by: PATCH /api/jobs/<id>, and ingest._set_status.
    """
    sets, params = [], []

    status = fields.get("status")
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        sets.append("status = ?")
        params.append(status)
        col = DATE_FIELDS.get(status)
        if col and col not in fields:
            sets.append(f"{col} = COALESCE({col}, ?)")
            params.append(date.today().isoformat())

    for col in ("company", "title"):
        if col in fields:
            val = (fields[col] or "").strip()
            if not val:
                raise ValueError(f"{col} is required")
            sets.append(f"{col} = ?")
            params.append(val)

    if "applied_date" in fields:
        applied = (fields["applied_date"] or "").strip()
        date.fromisoformat(applied)
        sets.append("applied_date = ?")
        params.append(applied)

    for col in EDITABLE_DATES & fields.keys():
        val = fields[col]
        if val:
            date.fromisoformat(val)
        sets.append(f"{col} = ?")
        params.append(val or None)

    if not sets:
        raise ValueError("nothing to update")
    params.append(job_id)
    cur = conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def delete_job(conn, job_id):
    """Remove one application card for good.

    Inputs:  conn — an open tracker.db connection.
             job_id — the row to delete.
    Returns: True if a row was deleted, False if job_id matched nothing.
    Used by: DELETE /api/jobs/<id>.
    """
    cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    return cur.rowcount > 0


# --- sync -----------------------------------------------------------------

def run_sync():
    """Run ingest.py --commit over mail since the last sync watermark (blocking).

    Inputs:  none. The watermark is read from meta.last_synced; with none set,
             ingest falls back to its own --since default (30 days).
    Returns: (ok, last_synced) — whether the subprocess exited 0, and the
             watermark read back afterwards (ingest.py advances it itself).
    Notes:   stdout/stderr are inherited, so tqdm and progress stream live to the
             app's terminal.
    Used by: POST /api/sync, as a background job.
    """
    with closing(connect()) as conn:
        since = get_meta(conn, "last_synced")
    cmd = [sys.executable, "-u", str(HERE / "ingest.py"), "--commit"]
    if since:
        cmd += ["--after", since]
    proc = subprocess.run(cmd, cwd=str(HERE))
    with closing(connect()) as conn:
        return proc.returncode == 0, get_meta(conn, "last_synced")


def run_scan():
    """Run research.py, then locations.py, over the saved profile (blocking).

    Inputs:  none. research.py reads the profile out of research.db itself.
    Returns: (ok, count, last_scanned) — whether research.py exited 0, how many
             matches are now visible (counted the way the tab counts them, so the
             two agree), and the watermark research.py wrote.
    Notes:   New locations only ever arrive from a scan, so locations.py is
             canonicalised here rather than behind its own button. It gets its own
             subprocess for the same reason run_embed does: the model work stays
             out of the server process. Its failure is not fatal — unmapped rows
             fall back to the browser's own bucketing and the next scan retries.
    Used by: POST /api/research/scan, as a background job.
    """
    proc = subprocess.run([sys.executable, "-u", str(HERE / "research.py")], cwd=str(HERE))
    subprocess.run([sys.executable, "-u", str(HERE / "locations.py")], cwd=str(HERE))
    with closing(research.connect()) as conn:
        count = len(research.list_matches(conn))
        last = research.get_setting(conn, "last_scanned")
    return proc.returncode == 0, count, last


def embed_targets(conn, ids, mode):
    """Which of the requested rows a check should hand to embed.py.

    Inputs:  conn — an open research.db connection.
             ids — the match ids the check will visit.
             mode — one of enrich.MODES.
    Returns: the subset of `ids` to re-embed afterwards, as a list.

    Decided *before* enrich.py runs, because 'new' is defined by the state enrich
    is about to overwrite.

      full  — every requested row. enrich rewrites them all and NULLs the vector of
              each one it changes, and embed.py only takes rows whose vector is NULL,
              so the re-embed lands on exactly what was rewritten.
      new   — only the rows enrich will actually send to the model: the ones it counts
              as unenriched (years_experience IS NULL, enrich.rows_to_check). Afterwards
              they are indistinguishable from rows enriched weeks ago, and passing the
              whole list would drag every never-embedded row in with them.
      alive — nothing. No model calls, so no doc_text() input moved.

    Used by: run_check.
    """
    if mode == "alive" or not ids:
        return []
    if mode == "full":
        return list(ids)
    return [r["id"] for r in conn.execute(
        f"SELECT id FROM market_matches WHERE years_experience IS NULL "
        f"AND id IN ({','.join('?' * len(ids))})", ids)]


def check_args(ids, mode):
    """Validate a check request before anything acts on it.

    Inputs:  ids — the requested match ids, in any form int() accepts.
             mode — the requested enrich mode.
    Returns: (ids, mode) with the ids coerced to ints, so no raw request value
             ever reaches argv.
    Raises:  ValueError on an empty id list or an unknown mode; TypeError if ids
             is not iterable.

    Split out so the request thread can reject a bad request with a 400 before
    starting a job for it. run_check calls it too, because these values reach argv
    and the boundary check belongs next to the subprocess.

    Used by: POST /api/research/check and run_check.
    """
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError("no match ids given")
    if mode not in MODES:
        raise ValueError(f"invalid mode: {mode!r}")
    return ids, mode


def run_check(ids, mode="full"):
    """Run enrich.py over the given postings, then re-embed what changed (blocking).

    Inputs:  ids — match ids to visit.
             mode — 'full' visits and enriches every one, 'alive' only updates the
             alive flag (no model calls), 'new' enriches only never-enriched rows.
    Returns: (ok, alive, dead) — whether enrich.py exited 0, how many of the
             requested postings are still open, and how many were buried.
    Raises:  ValueError / TypeError via check_args on a malformed request.

    Each posting's url is visited: dead ones are hidden, live ones get salary,
    skills, description and years filled in. Anything rewritten is then re-embedded
    so the Resume Analysis tab never trails the corpus.

    Dropping the dead rows from the embed list is load-bearing: embed.py's --ids
    replaces its own alive filter rather than narrowing it, so a posting enrich just
    buried would otherwise get a vector. A failed embed is not fatal, same as
    locations.py in run_scan — the rows stay NULL and the backfill button retries.

    Used by: POST /api/research/check, as a background job.
    """
    ids, mode = check_args(ids, mode)
    with closing(research.connect()) as conn:
        todo = embed_targets(conn, ids, mode)
    cmd = [sys.executable, "-u", str(HERE / "enrich.py"),
           "--mode", mode, "--ids", ",".join(map(str, ids))]
    proc = subprocess.run(cmd, cwd=str(HERE))
    with closing(research.connect()) as conn:
        alive = {r["id"] for r in conn.execute(
            f"SELECT id FROM market_matches WHERE alive = 1 AND id IN "
            f"({','.join('?' * len(ids))})", ids)}
    todo = [i for i in todo if i in alive]
    if todo:
        subprocess.run([sys.executable, "-u", str(HERE / "embed.py"),
                        "--ids", ",".join(map(str, todo))], cwd=str(HERE))
    return proc.returncode == 0, len(alive), len(ids) - len(alive)


def embed_counts(conn):
    """How much of the live corpus is embedded.

    Inputs:  conn — an open research.db connection.
    Returns: (done, pending) — alive postings that carry a vector, and alive
             postings that still don't.
    Used by: run_embed, to report where a backfill got to.
    """
    row = conn.execute(
        "SELECT COUNT(embedding) AS done, COUNT(*) - COUNT(embedding) AS pending "
        "FROM market_matches WHERE alive = 1").fetchone()
    return row["done"], row["pending"]


def run_embed(resume_only=False):
    """Run embed.py to fill in missing vectors (blocking).

    Inputs:  resume_only — embed the saved resume's sections instead of the
             postings.
    Returns: (ok, done, pending) — whether embed.py exited 0, and the corpus
             counts from embed_counts afterwards.
    Notes:   The subprocess boundary is load-bearing: it keeps torch out of the
             server process, so starting the app stays instant whether or not
             sentence-transformers is installed.
    Used by: POST /api/embed/backfill as a background job, and PUT /api/resume.
    """
    cmd = [sys.executable, "-u", str(HERE / "embed.py")]
    if resume_only:
        cmd.append("--resume")
    proc = subprocess.run(cmd, cwd=str(HERE))
    with closing(research.connect()) as conn:
        done, pending = embed_counts(conn)
    return proc.returncode == 0, done, pending


# --- background jobs ------------------------------------------------------
# The run_* helpers above take anywhere from 20s to several minutes. Running one
# inside a request handler means the browser holds an idle connection the whole
# time, gives up, and the result is lost even though the subprocess wrote it to
# the DB. So they run on a thread instead: POST starts a job and returns 202, the
# browser polls GET /api/job for the result.

JOB_LOCK = threading.Lock()
JOB = None      # {name, started, done, result, error} — the current or last-finished run


def start_job(name, fn):
    """Run fn() on a background thread, one job at a time.

    Inputs:  name — a label the browser polls back, e.g. "scan".
             fn — a zero-argument callable; its return value becomes the job's
             result and any exception becomes the job's error string.
    Returns: True if the job started, False if one is already running.

 

    Used by: Handler._start, behind every long-running POST/PUT.
    """
    global JOB
    with JOB_LOCK:
        if JOB and not JOB["done"]:
            return False
        JOB = job = {"name": name, "started": time.time(),
                     "done": False, "result": None, "error": None}

    def work():
        """Thread body: run fn, record its result or error, then mark the job done."""
        try:
            job["result"] = fn()
        except Exception as e:               # noqa: BLE001 - surfaced by GET /api/job
            job["error"] = str(e)
        finally:
            job["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return True


def job_status():
    """The current or last-finished job, for the browser's poll loop.

    Inputs:  none (reads the module's JOB slot under the lock).
    Returns: a dict of {name, started, done, result, error, elapsed}, where
             elapsed is seconds since the job started. With no job ever started,
             an idle-looking placeholder with name None and done True.
    Used by: GET /api/job, and Handler._start when reporting a 409.
    """
    with JOB_LOCK:
        job = JOB
    if job is None:
        return {"name": None, "done": True, "result": None, "error": None, "elapsed": 0}
    return {**job, "elapsed": round(time.time() - job["started"], 1)}


# --- http -----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """The whole HTTP surface: index.html plus the /api/* JSON endpoints."""

    def _send(self, code, payload=None, ctype="application/json"):
        """Write one response and finish the exchange.

        Inputs:  code — the HTTP status.
                 payload — bytes sent as-is, None for an empty body, or any
                 JSON-serializable object.
                 ctype — the Content-Type header value.
        Returns: None.
        Notes:   A browser that gave up on a slow POST (check/scan/embed) before we
                 answered raises BrokenPipeError here. Those endpoints write their
                 results to the DB and only return counts, so nothing is lost — a
                 reload shows the real state. It is swallowed rather than letting
                 socketserver print a traceback per abandoned request.
        """
        body = b"" if payload is None else (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        try:
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _body(self):
        """The request body, parsed as JSON.

        Inputs:  none (reads Content-Length and rfile).
        Returns: the decoded object, or {} for an empty body.
        Raises:  json.JSONDecodeError on a malformed body.
        """
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _raw(self):
        """The request body as bytes, for the one endpoint that takes a file.

        Inputs:  none (reads Content-Length and rfile).
        Returns: the raw bytes, or b"" for an empty body.
        Raises:  ValueError if Content-Length exceeds resume.MAX_BYTES — refusing
                 on the header alone means an oversized upload is rejected before
                 it is read into memory.

        """
        length = int(self.headers.get("Content-Length", 0))
        if length > resume.MAX_BYTES:
            raise ValueError(f"file too large ({length} bytes, max {resume.MAX_BYTES})")
        return self.rfile.read(length) if length else b""

    def _id(self):
        """The trailing path segment of the request url as an int.

        Inputs:  none (reads self.path).
        Returns: e.g. 7 for /api/jobs/7.
        Raises:  ValueError if that segment is not a number.
        """
        return int(self.path.rstrip("/").rsplit("/", 1)[-1])

    def _start(self, name, fn):
        """Hand a long-running callable to start_job and answer accordingly.

        Inputs:  name, fn — as start_job takes them.
        Returns: None, having sent 202 if the job started or 409 if one is
                 already running.
        """
        if start_job(name, fn):
            return self._send(202, {"started": True, "job": name})
        return self._send(409, {"error": f"{job_status()['name']} is already running",
                                "job": job_status()["name"]})

    def do_GET(self):
        """Serve index.html and every read-only endpoint.

        Inputs:  none (routes on self.path).
        Returns: None, having sent the response. Unknown paths get a 404.

        Routes: / and /index.html (the app), /api/jobs, /api/sync, /api/job,
        /api/research/profile, /api/research/matches, /api/research/matches/all,
        /api/resume, /api/resume/matches.
        """
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/jobs":
            with closing(connect()) as conn:
                self._send(200, list_jobs(conn))
        elif self.path == "/api/sync":
            with closing(connect()) as conn:
                self._send(200, {"last_synced": get_meta(conn, "last_synced")})
        elif self.path == "/api/job":
            self._send(200, job_status())
        elif self.path == "/api/research/profile":
            with closing(research.connect()) as conn:
                prof = research.get_profile(conn)
                prof["last_scanned"] = research.get_setting(conn, "last_scanned")
                self._send(200, prof)
        elif self.path == "/api/research/matches":
            with closing(research.connect()) as conn:
                self._send(200, research.list_matches(conn))
        elif self.path == "/api/research/matches/all":
            # every alive posting, ignoring the saved profile — the Salary tab does its own
            # filtering. An empty-titles profile is list_matches' "return everything" path.
            with closing(research.connect()) as conn:
                self._send(200, research.list_matches(conn, {"titles": []}))
        elif self.path == "/api/resume":
            with closing(research.connect()) as conn:
                self._send(200, resume.get_resume(conn) or {})
        elif self.path == "/api/resume/matches":
            # Every alive embedded posting, scored, unfiltered: the tab's importance chips
            # and hard-filter toggles re-rank client-side, so this is fetched once.
            with closing(research.connect()) as conn:
                self._send(200, resume.rank(conn, research.get_profile(conn)))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        """Start the long-running jobs, extract an uploaded resume, add a card.

        Inputs:  none (routes on self.path, reads the body where relevant).
        Returns: None, having sent the response. Unknown paths get a 404.

        /api/sync, /api/research/scan, /api/research/check and /api/embed/backfill
        run as background jobs: 202 now, result via GET /api/job, output still
        streaming to the app's terminal. /api/resume/extract is stateless — it turns
        an uploaded file into text and hands it straight back for the user to eyeball
        and fix; nothing is stored until they PUT it. /api/auth stays synchronous
        because gmail_service() can sit on a browser consent screen forever, which is
        a person to wait on, not a batch job to poll. /api/jobs creates a card.
        """
        if self.path == "/api/sync":
            return self._start("sync", lambda: dict(
                zip(("ok", "last_synced"), run_sync())))
        if self.path == "/api/research/scan":
            return self._start("scan", lambda: dict(
                zip(("ok", "count", "last_scanned"), run_scan())))
        if self.path == "/api/research/check":
            # Validated here, on the request thread, so a malformed request is still a
            # 400 the caller can act on rather than a job that starts and then fails.
            try:
                d = self._body()
                ids, mode = check_args(d.get("ids") or [], d.get("mode") or "full")
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            return self._start("check", lambda: dict(
                zip(("ok", "alive", "dead"), run_check(ids, mode))))
        if self.path == "/api/resume/extract":
            try:
                text = resume.extract_text(self._raw(), self.headers.get("X-Filename", ""))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, {"text": text})
        if self.path == "/api/embed/backfill":
            return self._start("backfill", lambda: dict(
                zip(("ok", "embedded", "pending"), run_embed())))
        if self.path == "/api/auth":
            try:
                import ingest                 # refreshes token.json, or opens a browser
                ingest.gmail_service()
                return self._send(200, {"ok": True})
            except Exception as e:            # noqa: BLE001 - surface to the UI
                return self._send(500, {"error": str(e)})
        if self.path == "/api/research/parse":
            # The Add-by-URL "Fetch" button: one Ollama call to read title/company/location
            # off the pasted page (~seconds), so it stays synchronous like /api/auth rather
            # than joining the scan/check job queue. A fetch or model failure is a 502.
            try:
                url = (self._body().get("url") or "").strip()
            except json.JSONDecodeError as e:
                return self._send(400, {"error": str(e)})
            if not url:
                return self._send(400, {"error": "url is required"})
            try:
                return self._send(200, enrich.posting_meta(url))
            except Exception as e:            # noqa: BLE001 - fetch/model failure, surface to the UI
                return self._send(502, {"error": str(e)})
        if self.path == "/api/research/add":
            # Manually log one posting by url. Synchronous (a single INSERT, no model
            # call), unlike scan/check.
            try:
                d = self._body()
                with closing(research.connect()) as conn:
                    mid, created = research.add_match(
                        conn, d.get("url"), d.get("title"), d.get("company"), d.get("location"))
                return self._send(201 if created else 200,
                                  {"id": mid, "created": created})
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
        if self.path != "/api/jobs":
            return self._send(404, {"error": "not found"})
        try:
            d = self._body()
            with closing(connect()) as conn:
                job = create_job(conn, d.get("company"), d.get("title"), d.get("applied_date"))
            self._send(201, job)
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": str(e)})

    def do_PUT(self):
        """Save the confirmed resume text (PUT /api/resume only).

        Inputs:  a JSON body of {text, filename}; text is required.
        Returns: None, having sent 202 (the work runs as a background job), 400
                 on a blank or malformed body, or 404 on any other path.

        The work splits the text into sections, asks the local model for skills and
        total years, then embeds the sections. That is slow — an Ollama call plus a
        model load — so it runs as a background job like scan/check: the body is
        validated here, the work happens on a thread, the result arrives via /api/job.
        """
        if self.path != "/api/resume":
            return self._send(404, {"error": "not found"})
        try:
            d = self._body()
            text = (d.get("text") or "").strip()
            if not text:
                raise ValueError("resume text is required")
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        filename = d.get("filename")

        def save():
            """Job body: extract skills, store the resume, then embed its sections.

            Returns {ok, resume, profile, pending} for the browser's poll.
            Raises RuntimeError if Ollama is unreachable.
            """
            sections = resume.split_sections(text)
            try:
                x = resume.extract_skills(text)      # local Ollama over http
            except OSError as e:                     # urllib.error.URLError subclasses this
                raise RuntimeError(f"could not reach Ollama at "
                                   f"{resume.enrich.OLLAMA_URL}: {e}") from e
            with closing(research.connect()) as conn:
                # vectors land in a second step; embed.py owns them so torch stays out
                resume.save_resume(conn, filename, text, sections,
                                   x["skills"], x["years_experience"], None)
            ok, _, pending = run_embed(resume_only=True)
            with closing(research.connect()) as conn:
                return {"ok": ok, "resume": resume.get_resume(conn),
                        "profile": research.get_profile(conn), "pending": pending}

        return self._start("resume", save)

    def do_PATCH(self):
        """Edit the resume skill list, the search profile, or one job card.

        Inputs:  none (routes on self.path, reads a JSON body).
        Returns: None, having sent the response. A bad body is a 400, an unknown
                 path or a missing row is a 404.

        Routes: /api/resume/skills (replace the skill list by hand),
        /api/research/profile (merge fields into the saved profile),
        /api/jobs/<id> (apply edits via update_job).
        """
        if self.path == "/api/resume/skills":
            try:
                d = self._body()
                with closing(research.connect()) as conn:
                    skills = resume.set_skills(conn, d.get("skills"))
                if skills is None:
                    return self._send(404, {"error": "no resume saved yet"})
                return self._send(200, {"skills": skills})
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
        if self.path == "/api/research/profile":
            try:
                d = self._body()
                with closing(research.connect()) as conn:
                    prof = research.set_profile(conn, d)
                return self._send(200, prof)
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
        if not self.path.startswith("/api/jobs/"):
            return self._send(404, {"error": "not found"})
        try:
            d = self._body()
            with closing(connect()) as conn:
                ok = update_job(conn, self._id(), d)
            self._send(200 if ok else 404, {"ok": ok})
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": str(e)})

    def do_DELETE(self):
        """Remove one logged market match, or one job card.

        Inputs:  none (routes on self.path; the id is the last path segment).
        Returns: None, having sent {"ok": bool} with 200 or 404. A non-numeric id
                 is a 400, an unknown path a 404.

        Routes: /api/research/matches/<id>, /api/jobs/<id>.
        """
        if self.path.startswith("/api/research/matches/"):
            try:
                with closing(research.connect()) as conn:
                    ok = research.delete_match(conn, self._id())
                return self._send(200 if ok else 404, {"ok": ok})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
        if not self.path.startswith("/api/jobs/"):
            return self._send(404, {"error": "not found"})
        try:
            with closing(connect()) as conn:
                ok = delete_job(conn, self._id())
            self._send(200 if ok else 404, {"ok": ok})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *args):
        """Silence BaseHTTPRequestHandler's per-request stderr line.

        Inputs:  ignored. Returns: None. Delete this method to see request logs.
        """
        pass


if __name__ == "__main__":
    with closing(connect()) as c:
        init_db(c)
    with closing(research.connect()) as c:
        research.init_db(c)
    url = f"http://{HOST}:{PORT}"
    print(f"Job tracker running at {url}   (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
