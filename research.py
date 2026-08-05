#!/usr/bin/env python3
"""Scan the public job-board feeds and log postings that match a saved profile.

Two sources, same profile filter and same table:

  * the gzipped chunks published by Feashliaa/job-board-data (GitHub Pages), and
  * the a16z Speedrun Talent Network's markdown export (no auth, no key).

Keeps the postings matching your search profile (titles / locations / seniority) and
upserts them into research.db — kept separate from the application tracker so the
matches never mingle with jobs you've actually applied to (and leaves room for a
future embeddings column).

Run:  python3 research.py        (uses the profile saved via the app's Market tab)
The Market tab's "Scan" button runs this same script as a subprocess.

The chunk feed's salary is an *estimated* market range, so the scan leaves it blank
there; Speedrun quotes the employer's own band, so that one is kept. Either way skills/
description are filled in later by enrich.py, which visits each posting's url. enrich.py
also owns `alive` — list_matches() hides dead postings, so a scan never has to prune them.
"""
import gzip
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date
from itertools import repeat
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).parent
RESEARCH_DB = HERE / "research.db"
# {i} -> chunk index; probed 0,1,2,… until a 404 says we've run past the last one.
CHUNK_URL = os.environ.get(
    "JOBFEED_URL",
    "https://feashliaa.github.io/job-board-data/data/chunks/jobs_chunk_{i}.json.gz",
)
CHUNK_WORKERS = int(os.environ.get("CHUNK_WORKERS", "8"))    # concurrent chunk downloads
LEVELS = ("intern", "entry", "mid", "senior")

# a16z Speedrun Talent Network — https://speedrun-talent-network.com/developers
# Their bulk markdown export: every open role in the portfolio, ~2MB, one request. The
# REST API at /api/v1/jobs is the same data paginated 50 at a time, which is the slower
# way to end up here. ?source= identifies us, as their docs ask.
SPEEDRUN_MD = "https://speedrun-talent-network.com/jobs.md"

# columns enrich.py fills in; added to databases created before that feature existed
ENRICHED_COLS = {
    "alive": "INTEGER NOT NULL DEFAULT 1",
    "years_experience": "INTEGER",
    "checked_at": "TEXT",
}

# columns embed.py fills in. embedded_at is the staleness marker: enrich.py rewrites
# description/skills, which invalidates the vector built from them, so a row with
# embedded_at < checked_at needs re-embedding.
EMBED_COLS = {
    "embedding": "BLOB",
    "embedded_at": "TEXT",
}


# --- database -------------------------------------------------------------

def connect():
    """Open a connection to research.db.

    Inputs:  none (RESEARCH_DB is module state).
    Returns: an sqlite3.Connection with a Row factory, a 5s busy timeout to wait
             out the scan subprocess's writes, and WAL journalling so the app can
             answer GETs while a scan or check writes. WAL sticks in the file
             header, so setting it is a no-op after the first time.
    Used by: app.py, enrich.py, embed.py, locations.py and resume.py — every
             module that touches the market data.
    """
    conn = sqlite3.connect(RESEARCH_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn):
    """Create the research schema, and migrate a database made before it grew.

    Inputs:  conn — an open research.db connection.
    Returns: None. Creates market_matches, settings, location_map and resume if
             absent, and adds any ENRICHED_COLS/EMBED_COLS column an older
             database is missing. SQLite allows ADD COLUMN with a constant
             default, so existing rows become alive = 1.
    Used by: every entry point, before it reads or writes.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS market_matches (
               id               INTEGER PRIMARY KEY AUTOINCREMENT,
               url              TEXT UNIQUE,
               title            TEXT NOT NULL,
               company          TEXT NOT NULL,
               location         TEXT,
               skill_level      TEXT,
               salary           TEXT DEFAULT '',   -- Speedrun's band, else enrich.py's
               skills           TEXT DEFAULT '',   -- comma-joined
               description      TEXT DEFAULT '',   -- enrich.py's extracted span, not the page
               years_experience INTEGER,           -- minimum stated; 0 = unstated
               alive            INTEGER NOT NULL DEFAULT 1,
               checked_at       TEXT,              -- last time enrich.py visited the url
               found_at         TEXT
           )"""
    )
    have = {r["name"] for r in conn.execute("PRAGMA table_info(market_matches)")}
    for col, decl in {**ENRICHED_COLS, **EMBED_COLS}.items():
        if col not in have:
            conn.execute(f"ALTER TABLE market_matches ADD COLUMN {col} {decl}")
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS location_map (
               raw    TEXT PRIMARY KEY,   -- exactly as the feed wrote it
               cities TEXT NOT NULL       -- comma-joined, same shape as market_matches.skills
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS resume (
               id         INTEGER PRIMARY KEY CHECK (id = 1),
               filename   TEXT,
               text       TEXT NOT NULL,
               sections   TEXT DEFAULT '[]',   -- JSON list of strings
               skills     TEXT DEFAULT '',     -- comma-joined, same shape as market_matches.skills
               vectors    BLOB,                -- packed float32, len(sections) x embed.DIM
               updated_at TEXT
           )"""
    )
    conn.commit()


def get_setting(conn, key, default=None):
    """Read one value out of the settings key/value table.

    Inputs:  conn — an open research.db connection.
             key — the setting name, e.g. "last_scanned" or "profile".
             default — what to return when the key is not set.
    Returns: the stored string, or `default`.
    Used by: get_profile, app.run_scan, and GET /api/research/profile.
    """
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn, key, value):
    """Write one value into the settings key/value table, inserting or overwriting.

    Inputs:  conn — an open research.db connection.
             key, value — the pair to store.
    Returns: None. Commits.
    Used by: set_profile, and scan() for the last_scanned watermark.
    """
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    conn.commit()


# --- profile --------------------------------------------------------------

def _clean_terms(terms):
    """Normalize a list of profile search terms.

    Inputs:  terms — a list of strings, or a newline-separated string, or None.
    Returns: a list of stripped, non-blank strings.
    Used by: _PROFILE_FIELDS, for titles and locations.
    """
    if isinstance(terms, str):
        terms = terms.splitlines()
    return [t.strip() for t in (terms or []) if t and t.strip()]


def _clean_years(v):
    """Normalize a years-of-experience value.

    Inputs:  v — anything int() might accept, or None.
    Returns: an int clamped to 0-60; anything unparseable becomes 0. 0 means
             "not stated" here exactly as it does in market_matches.years_experience.
    Used by: get_profile and _PROFILE_FIELDS.
    """
    try:
        return max(0, min(60, int(v or 0)))
    except (TypeError, ValueError):
        return 0


def get_profile(conn):
    """The saved search profile, with every field guaranteed present.

    Inputs:  conn — an open research.db connection.
    Returns: {"titles": [...], "locations": [...], "levels": [...],
             "years_experience": int}. An unset profile comes back as empty lists
             and 0 rather than as None.
    Used by: matches()/list_matches, the scan, resume.rank, and GET
             /api/research/profile.
    """
    raw = get_setting(conn, "profile")
    prof = json.loads(raw) if raw else {}
    return {
        "titles": prof.get("titles", []),
        "locations": prof.get("locations", []),
        "levels": prof.get("levels", []),
        "years_experience": _clean_years(prof.get("years_experience")),
    }


# Which keys set_profile knows how to write, and how to clean each one.
_PROFILE_FIELDS = {
    "titles": _clean_terms,
    "locations": _clean_terms,
    "levels": lambda v: [x for x in (v or []) if x in LEVELS],
    "years_experience": _clean_years,
}


def set_profile(conn, d):
    """Merge the given keys into the saved profile.

    Inputs:  conn — an open research.db connection.
             d — a dict that may carry any of _PROFILE_FIELDS' keys; each value is
             passed through that field's cleaner. Unknown keys are ignored, so
             handing back a whole GET response (which carries last_scanned) is safe.
    Returns: the whole profile as get_profile would give it.

    Used by: PATCH /api/research/profile, and resume.save_resume for the year count.
    """
    prof = get_profile(conn)
    for key, clean in _PROFILE_FIELDS.items():
        if key in d:
            prof[key] = clean(d[key])
    set_setting(conn, "profile", json.dumps(prof))
    return prof


# --- matching (pure; unit-tested without network) -------------------------

# Short forms the boards write a city as, so "San Francisco" in a profile still finds
# "SF Office" and "New York" finds "NYC, NY". Matched on word boundaries, unlike the
# profile term itself: a bare "sf" as a substring would fire inside "Sfax" and "ny"
# inside "Albany". Add a city by adding a line — nothing else reads this.
LOCATION_ALIASES = {
    "san francisco": ("sf", "san fran"),
    "new york": ("nyc", "ny"),
}


def _location_hit(term, loc):
    """Does one posting location satisfy one profile location term?

    Inputs:  term — a lowercased profile location term.
             loc — a lowercased posting location string.
    Returns: True on a plain substring hit, or on any of the term's
             LOCATION_ALIASES matching at a word boundary.
    Used by: matches().
    """
    if term in loc:
        return True
    return any(re.search(rf"\b{re.escape(a)}\b", loc) for a in LOCATION_ALIASES.get(term, ()))


def matches(job, profile):
    """Does one feed record satisfy the search profile?

    Inputs:  job — a feed record (or any row) with title, location, skill_level
             and optionally is_recruiter.
             profile — titles / locations / levels as get_profile returns them.
    Returns: True to keep the posting, False to drop it.

    Used by: _scan_chunk (off the main thread — this is pure and only reads the
             profile), scan_speedrun, and list_matches.
    """
    if job.get("is_recruiter"):
        return False
    titles = profile.get("titles") or []
    if not titles:
        return False
    title = (job.get("title") or "").lower()
    if not any(t.lower() in title for t in titles):
        return False
    locations = profile.get("locations") or []
    if locations:
        loc = (job.get("location") or "").lower()
        if not any(_location_hit(l.lower(), loc) for l in locations):
            return False
    levels = profile.get("levels") or []
    if levels and job.get("skill_level") not in levels:
        return False
    return True


# --- match storage --------------------------------------------------------

def upsert_match(conn, job, salary=""):
    """Insert a match, or refresh its feed-derived fields if the url is logged.

    Inputs:  conn — an open research.db connection.
             job — a record with url, title, company, location, skill_level.
             salary — written on first insert only. Deliberately a parameter
             rather than a field of `job`: the chunk feed carries an estimated band
             under that name and must not leak it in. Leaving it out of the DO
             UPDATE keeps enrich.py's employer-read value authoritative once it exists.
    Returns: None. Does not commit — the caller batches that.

    Used by: scan() and scan_speedrun().
    """
    conn.execute(
        """INSERT INTO market_matches (url, title, company, location, skill_level, salary, found_at)
           VALUES (:url, :title, :company, :location, :skill_level, :salary, :found_at)
           ON CONFLICT(url) DO UPDATE SET
               title       = excluded.title,
               company     = excluded.company,
               location    = excluded.location,
               skill_level = excluded.skill_level""",
        {
            "url": job.get("url"),
            "title": (job.get("title") or "").strip(),
            "company": (job.get("company") or "").strip(),
            "location": (job.get("location") or "").strip() or None,
            "skill_level": job.get("skill_level"),
            "salary": salary,
            "found_at": date.today().isoformat(),
        },
    )


def list_matches(conn, profile=None):
    """Live logged matches whose title/location still fit the profile.

    Inputs:  conn — an open research.db connection.
             profile — the filter to apply; None reads the saved one. Pass
             {"titles": []} to get every alive row back.
    Returns: a list of dicts, newest first. `embedding` is dropped (it is a BLOB
             and this goes straight to json.dumps) and `cities` rides along from
             location_map — NULL until locations.py has seen that raw string, and
             the browser falls back to its own bucketing for those.

    Used by: GET /api/research/matches and /matches/all, and app.run_scan's count.
    """
    rows = [{k: v for k, v in dict(r).items() if k != "embedding"} for r in conn.execute(
        "SELECT m.*, lm.cities FROM market_matches m "
        "LEFT JOIN location_map lm ON lm.raw = m.location "
        "WHERE m.alive = 1 ORDER BY m.found_at DESC, m.id DESC")]
    if profile is None:
        profile = get_profile(conn)
    if not profile.get("titles"):
        return rows
    return [r for r in rows if matches(r, {**profile, "levels": []})]


def delete_match(conn, match_id):
    """Remove one logged posting for good.

    Inputs:  conn — an open research.db connection.
             match_id — the market_matches row to delete.
    Returns: True if a row was deleted, False if match_id matched nothing.
    Used by: DELETE /api/research/matches/<id>.
    """
    cur = conn.execute("DELETE FROM market_matches WHERE id = ?", (match_id,))
    conn.commit()
    return cur.rowcount > 0


# --- chunk feed -----------------------------------------------------------

def _fetch_chunk(i, retries=4):
    """Download and decode one chunk of the job feed.

    Inputs:  i — the chunk index, substituted into CHUNK_URL.
             retries — how many times to back off on a transient failure.
    Returns: the decoded job list, or None once we've run past the last chunk (404).
    Raises:  urllib.error.HTTPError for anything but 404/429/5xx, and for a
             transient code that outlives the retries — a real failure should stop
             the scan loudly rather than silently truncating the feed.
    Used by: _scan_chunk.
    """
    url = CHUNK_URL.format(i=i)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(gzip.decompress(r.read()))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in (429, 500, 502, 503) or attempt == retries:
                raise
            time.sleep(2 ** attempt)  # 1, 2, 4, 8s


def _scan_chunk(i, profile):
    """Fetch chunk i and return only its matching jobs.

    Inputs:  i — the chunk index.
             profile — the search profile to filter with.
    Returns: the matching records as a list, or None once past the last chunk.

    Used by: scan(), inside the thread pool.
    """
    jobs = _fetch_chunk(i)
    return None if jobs is None else [j for j in jobs if matches(j, profile)]


# --- a16z Speedrun Talent Network ----------------------------------------

# "- [TITLE at COMPANY](URL) - LOCATION · Remote · Full Time · $150k - $250k",
# where everything after the location is optional and any of it may be absent.
SPEEDRUN_LINE = re.compile(r"^- \[(.+)\]\((\S+)\) - (.+)$")


def _speedrun_level(title):
    """Read seniority off a job title.

    Inputs:  title — the posting's title.
    Returns: "intern", "entry", "senior", or None for a title that names no
             seniority (unknown level, not "mid").

    Used by: speedrun_record.
    """
    t = title.lower()
    for pattern, level in ((r"\b(intern|internship)\b", "intern"),
                           (r"\b(junior|jr|new grad|entry[- ]level)\b", "entry"),
                           (r"\b(senior|sr|staff|principal|lead|head|director|vp|chief|"
                            r"founding|president|founders?)\b", "senior")):
        if re.search(pattern, t):
            return level
    return None


def _speedrun_meta(meta):
    """Split a Speedrun line's trailing metadata run into location and salary.

    Inputs:  meta — the "LOCATION · Remote · Full Time · $150k - $250k" tail, where
             everything after the location is optional.
    Returns: (location, salary). Only a plain dollar band is kept: parseSalary() in
             index.html reads bare numbers, so "PLN 6,800 - PLN 8,500/mo" would be
             charted as US dollars a year. A "/mo" or "/hr" suffix is rejected for
             the same reason, leaving salary "".
    Used by: speedrun_record.
    """
    parts = [p.strip() for p in meta.split("·")]
    salary = next((p for p in reversed(parts)
                   if p.startswith("$") and "/" not in p), "")
    return parts[0], salary


def speedrun_record(line):
    """Reshape one line of the markdown export into the chunk feed's field names.

    Inputs:  line — one line of the export.
    Returns: {url, title, company, location, skill_level, salary}, so matches() and
             upsert_match() take it without knowing where it came from. None for a
             line that isn't a posting (headings, the preamble, blanks) or one whose
             label carries no " at " separator to split on.

    Used by: scan_speedrun.
    """
    m = SPEEDRUN_LINE.match(line)
    if not m:
        return None
    label, url, meta = m.groups()
    title, _, company = label.rpartition(" at ")
    if not title:                      # no separator at all — can't tell the two apart
        return None
    location, salary = _speedrun_meta(meta)
    return {"url": url, "title": title.strip(), "company": company.strip(),
            "location": location, "skill_level": _speedrun_level(title), "salary": salary}


def _speedrun_export(retries=4):
    """Download the whole Speedrun network as markdown, in one request.

    Inputs:  retries — how many times to back off on a transient failure.
    Returns: the export's lines as a list of strings.
    Raises:  urllib.error.HTTPError for anything but 429/5xx, and for a transient
             code that outlives the retries, so a real failure stops the scan loudly.
    Notes:   Cloudflare 403s (error 1010) on the default Python-urllib agent, hence
             the browser User-Agent.
    Used by: scan_speedrun.
    """
    req = urllib.request.Request(f"{SPEEDRUN_MD}?source=unemployment_stack",
                                 headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", "replace").splitlines()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == retries:
                raise
            time.sleep(2 ** attempt)


def scan_speedrun(conn, profile, progress=True):
    """Log the postings in the Speedrun export that match the profile.

    Inputs:  conn — an open research.db connection.
             profile — the search profile.
             progress — show the tqdm bar.
    Returns: how many new postings were logged. Commits.

    The export is every open role in the portfolio in one 2MB download, so the
    profile is applied here rather than through the API's search — same "contains"
    rule the chunk feed gets, and no relevance ranking deciding what we never see.
    (The REST search is the wrong shape for this: `q` is ranked, not exhaustive, and
    its 200-page ceiling and per-page latency put a complete walk out of reach.)

    Levels are dropped from the check on purpose: the export states no seniority, so
    rejecting on it here would keep postings out of the database entirely instead of
    letting the Market tab's checkboxes decide.

    Duplicates are suppressed on (company, title) — a role reachable through both
    sources is one posting with two urls, and the chunk feed's is the employer's own
    board. Two genuinely distinct reqs with the same title at one company collapse
    into one; add location to the key if that starts costing you real postings. Only
    live rows dedupe: a posting enrich.py buried is hidden from the tab, so letting it
    suppress a listing Speedrun still carries would lose the role from both sources
    at once.

    Used by: scan().
    """
    seen = {(r["company"].strip().lower(), r["title"].strip().lower())
            for r in conn.execute("SELECT company, title FROM market_matches WHERE alive = 1")}
    lines = _speedrun_export()
    new = 0
    for line in tqdm(lines, desc="Speedrun export", unit="line", disable=not progress):
        rec = speedrun_record(line)
        if rec is None or not matches(rec, {**profile, "levels": []}):
            continue
        key = (rec["company"].lower(), rec["title"].lower())
        if key in seen:
            continue
        seen.add(key)
        upsert_match(conn, rec, salary=rec["salary"])
        new += 1
    conn.commit()
    return new


# --- scan -----------------------------------------------------------------

def scan(profile=None, progress=True):
    """Download every chunk, then search Speedrun, upserting the matches.

    Inputs:  profile — the search profile; None reads the saved one. A profile with
             no titles logs nothing and says so.
             progress — show the tqdm bars.
    Returns: the running total of matches in the DB. Stamps last_scanned.

    Used by: __main__, and app.run_scan as a subprocess.
    """
    with closing(connect()) as conn:
        init_db(conn)
        if profile is None:
            profile = get_profile(conn)
        if not profile.get("titles"):
            print("No job titles in your profile — nothing to match. "
                  "Add titles in the Market tab and scan again.")
            return conn.execute("SELECT COUNT(*) AS n FROM market_matches").fetchone()["n"]

        bar = tqdm(desc="Scanning chunks", unit="chunk", disable=not progress)
        i = new = 0
        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            done = False
            while not done:
                wave = pool.map(_scan_chunk, range(i, i + CHUNK_WORKERS),
                                repeat(profile))
                for hits in wave:
                    if hits is None:
                        done = True
                        break
                    for job in hits:            # writes stay on this thread, like enrich.py
                        upsert_match(conn, job)
                        new += 1
                    conn.commit()               # commit per chunk; cheap and crash-safe
                    bar.update(1)
                    bar.set_postfix(matches=new)
                    i += 1
        bar.close()

        speedrun = scan_speedrun(conn, profile, progress=progress)

        set_setting(conn, "last_scanned", date.today().isoformat())
        total = conn.execute("SELECT COUNT(*) AS n FROM market_matches").fetchone()["n"]
    print(f"Scanned {i} chunk(s) + Speedrun; {new} posting(s) from the feed and "
          f"{speedrun} from Speedrun logged this run ({total} total).")
    return total


if __name__ == "__main__":
    scan()
