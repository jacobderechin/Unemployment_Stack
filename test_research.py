"""Unit tests for research.matches() — the pure profile-filter predicate —
and research.prune_matches(), which applies it to already-logged rows.
Runs without network:  pytest test_research.py
"""
import sqlite3
from contextlib import closing

import pytest

import research


def job(**kw):
    base = {"title": "Senior Software Engineer", "location": "Remote (US)",
            "skill_level": "senior", "is_recruiter": False}
    base.update(kw)
    return base


def prof(titles=(), locations=(), levels=()):
    return {"titles": list(titles), "locations": list(locations), "levels": list(levels)}


def test_title_contains_case_insensitive():
    assert research.matches(job(), prof(titles=["engineer"]))          # substring, lowercased
    assert research.matches(job(), prof(titles=["SOFTWARE"]))
    assert not research.matches(job(), prof(titles=["designer"]))


def test_empty_titles_match_nothing():
    # refuse to log the whole feed when no titles are set
    assert not research.matches(job(), prof())


def test_empty_locations_match_any():
    assert research.matches(job(location="Kraków, Poland"), prof(titles=["engineer"]))


def test_location_contains():
    p = prof(titles=["engineer"], locations=["remote"])
    assert research.matches(job(location="Remote (US)"), p)
    assert not research.matches(job(location="New York, NY"), p)


def test_location_aliases_widen_a_city_to_its_short_forms():
    sf = prof(titles=["engineer"], locations=["San Francisco"])
    for loc in ("SF Office", "sf bay", "San Fran", "San Francisco, CA"):
        assert research.matches(job(location=loc), sf), loc
    ny = prof(titles=["engineer"], locations=["New York"])
    for loc in ("NYC", "New York City", "Brooklyn, NY"):
        assert research.matches(job(location=loc), ny), loc


def test_location_aliases_respect_word_boundaries():
    # a bare substring alias would fire inside these; the profile term itself still
    # matches as a substring, which is why "sf"/"ny" alone need the tighter rule
    assert not research.matches(job(location="Sfax, Tunisia"),
                                prof(titles=["engineer"], locations=["San Francisco"]))
    for loc in ("Albany, Oregon", "Germany", "Sydney"):
        assert not research.matches(job(location=loc),
                                    prof(titles=["engineer"], locations=["New York"])), loc


def test_empty_levels_match_any():
    assert research.matches(job(skill_level="entry"), prof(titles=["engineer"]))


def test_level_filter():
    p = prof(titles=["engineer"], levels=["mid", "senior"])
    assert research.matches(job(skill_level="senior"), p)
    assert not research.matches(job(skill_level="intern"), p)


def test_recruiter_excluded():
    assert not research.matches(job(is_recruiter=True), prof(titles=["engineer"]))


def logged_db(*levels):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    research.init_db(conn)
    for i, lvl in enumerate(levels):
        research.upsert_match(conn, job(url=f"u{i}", company="Acme", skill_level=lvl))
    conn.commit()
    return conn


def test_list_matches_sends_every_level_regardless_of_profile():
    # the level checkboxes filter in the browser; they can only re-show a level the
    # server actually sent, so list_matches must not pre-filter on levels
    conn = logged_db("intern", "mid", "senior")
    p = prof(titles=["engineer"], levels=["senior"])
    assert {m["skill_level"] for m in research.list_matches(conn, p)} == {"intern", "mid", "senior"}


def test_list_matches_hides_rows_the_narrowed_title_excludes():
    conn = logged_db("mid", "senior")            # both titled "Senior Software Engineer"
    assert research.list_matches(conn, prof(titles=["designer"])) == []
    # hidden, not deleted: widening the profile brings them back
    assert len(research.list_matches(conn, prof(titles=["engineer"]))) == 2
    assert conn.execute("SELECT COUNT(*) FROM market_matches").fetchone()[0] == 2


def test_list_matches_shows_all_when_profile_has_no_titles():
    conn = logged_db("intern", "senior")
    assert len(research.list_matches(conn, prof())) == 2


import gzip
import json
import urllib.error


def _fake_urlopen(codes):
    """urlopen that raises HTTPError for each code in `codes`, then returns an
    empty-job-list chunk. Tracks how many times it was called."""
    calls = {"n": 0}

    class _Resp:
        def read(self): return gzip.compress(json.dumps([]).encode())
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def open_(url, timeout=None):
        i = calls["n"]; calls["n"] += 1
        if i < len(codes):
            raise urllib.error.HTTPError(url, codes[i], "err", {}, None)
        return _Resp()
    open_.calls = calls
    return open_


def test_fetch_chunk_retries_transient_503(monkeypatch):
    monkeypatch.setattr(research.time, "sleep", lambda s: None)  # no real backoff
    fake = _fake_urlopen([503, 503])                             # two 503s then OK
    monkeypatch.setattr(research.urllib.request, "urlopen", fake)
    assert research._fetch_chunk(0) == []
    assert fake.calls["n"] == 3


def test_fetch_chunk_404_is_end_of_feed(monkeypatch):
    fake = _fake_urlopen([404])
    monkeypatch.setattr(research.urllib.request, "urlopen", fake)
    assert research._fetch_chunk(0) is None


def test_fetch_chunk_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(research.time, "sleep", lambda s: None)
    fake = _fake_urlopen([503] * 99)                            # never recovers
    monkeypatch.setattr(research.urllib.request, "urlopen", fake)
    try:
        research._fetch_chunk(0, retries=2)
        assert False, "expected HTTPError to propagate"
    except urllib.error.HTTPError:
        pass


# --- a16z Speedrun Talent Network ----------------------------------------

LINE = ("- [Senior AI Engineer at Acme](https://speedrun-talent-network.com/jobs/x)"
        " - San Francisco \u00b7 Remote \u00b7 Full Time \u00b7 $180k - $230k")


def test_speedrun_parses_an_export_line():
    r = research.speedrun_record(LINE)
    assert r == {"url": "https://speedrun-talent-network.com/jobs/x",
                 "title": "Senior AI Engineer", "company": "Acme",
                 "location": "San Francisco", "skill_level": "senior",
                 "salary": "$180k - $230k"}


def test_speedrun_ignores_non_posting_lines():
    for line in ("## Engineering", "", "10331 open roles at a16z speedrun portfolio companies.",
                 "- [No separator](https://s/x) - San Francisco"):
        assert research.speedrun_record(line) is None


def test_speedrun_splits_on_the_last_at():
    # " at " inside the job title must not be mistaken for the company separator
    r = research.speedrun_record("- [Engineer at Scale at Acme](https://s/x) - Remote")
    assert (r["title"], r["company"]) == ("Engineer at Scale", "Acme")


def test_speedrun_optional_metadata_is_optional():
    r = research.speedrun_record("- [Data Scientist at Acme](https://s/x) - Remote")
    assert (r["location"], r["salary"], r["skill_level"]) == ("Remote", "", None)


def test_speedrun_level_reads_the_title_most_junior_first():
    lvl = lambda t: research._speedrun_level(t)
    assert lvl("Machine Learning Engineering Intern") == "intern"   # not "senior" via "engineering"
    assert lvl("Junior Data Scientist") == "entry"
    assert [lvl(t) for t in ("Staff Engineer", "Head of AI", "Founding Engineer")] == ["senior"] * 3
    assert lvl("Internal Communications Manager") is None           # "intern" is not a word here
    assert lvl("Data Scientist") is None                            # unstated, not "mid"


def test_speedrun_salary_only_when_plain_dollars_a_year():
    sal = lambda meta: research.speedrun_record(f"- [X at Y](https://s/x) - {meta}")["salary"]
    assert sal("SF \u00b7 $180k - $230k") == "$180k - $230k"
    assert sal("Krak\u00f3w \u00b7 PLN 6,800 - PLN 8,500/mo") == ""   # charted as USD otherwise
    assert sal("SF \u00b7 $50/hr") == ""                            # hourly, same reason
    assert sal("SF \u00b7 Full Time") == ""


def test_speedrun_salary_never_overwrites_an_enriched_one():
    conn = logged_db("senior")                                   # url u0, no salary
    conn.execute("UPDATE market_matches SET salary = '$1 - $2' WHERE url = 'u0'")
    research.upsert_match(conn, job(url="u0", company="Acme"), salary="$9 - $9")
    assert conn.execute("SELECT salary FROM market_matches").fetchone()[0] == "$1 - $2"


def _export(monkeypatch, *lines):
    monkeypatch.setattr(research, "_speedrun_export", lambda retries=4: list(lines))


def test_speedrun_skips_a_role_the_chunk_feed_already_logged(monkeypatch):
    conn = logged_db("senior")        # "Senior Software Engineer" at Acme, url u0
    _export(monkeypatch, "- [senior software engineer at ACME](https://s/x) - SF")
    assert research.scan_speedrun(conn, prof(titles=["engineer"]), progress=False) == 0
    assert conn.execute("SELECT COUNT(*) FROM market_matches").fetchone()[0] == 1


def test_speedrun_dedupes_against_live_rows_only(monkeypatch):
    # the chunk feed's copy is dead, so the tab hides it — suppressing Speedrun's live
    # listing too would lose the role from both sources at once
    conn = logged_db("senior")                                   # url u0, alive
    conn.execute("UPDATE market_matches SET alive = 0 WHERE url = 'u0'")
    _export(monkeypatch, "- [Senior Software Engineer at Acme](https://s/x) - SF")
    assert research.scan_speedrun(conn, prof(titles=["engineer"]), progress=False) == 1
    assert len(research.list_matches(conn, prof(titles=["engineer"]))) == 1


def test_speedrun_deduplicates_within_its_own_export(monkeypatch):
    conn = logged_db()                                            # empty
    _export(monkeypatch,
            "- [AI Engineer at Acme](https://s/a) - SF",
            "- [AI Engineer at Acme](https://s/b) - New York",     # same role, two listings
            "- [AI Engineer at Other Co](https://s/c) - SF")
    assert research.scan_speedrun(conn, prof(titles=["engineer"]), progress=False) == 2
    assert {r[0] for r in conn.execute("SELECT company FROM market_matches")} == {"Acme", "Other Co"}


def test_speedrun_keeps_unstated_levels_that_the_profile_would_reject(monkeypatch):
    # levels are the Market tab's job; filtering them at scan time would keep every
    # posting whose title states no seniority out of the database entirely
    conn = logged_db()
    _export(monkeypatch, "- [AI Engineer at Acme](https://s/x) - SF")
    assert research.scan_speedrun(conn, prof(titles=["engineer"], levels=["senior"]),
                                  progress=False) == 1


def test_speedrun_still_applies_the_title_and_location_filters(monkeypatch):
    conn = logged_db()
    _export(monkeypatch, "- [AI Engineer at Acme](https://s/x) - Austin, TX",
                         "- [Product Manager at Acme](https://s/y) - New York")
    assert research.scan_speedrun(conn, prof(titles=["engineer"], locations=["new york"]),
                                  progress=False) == 0


# --- parallel chunk scan --------------------------------------------------

CHUNKS = 5      # chunks 0..4 exist, 5 and beyond 404


def _canned_chunks(seen):
    """A _fetch_chunk that serves CHUNKS chunks of one matching job each, then 404s.
    Records every index it was asked for, so the test can prove nothing past the end
    was mistaken for real data."""
    def fake(i, retries=4):
        seen.append(i)
        if i >= CHUNKS:
            return None
        return [job(title="AI Engineer", company=f"C{i}", url=f"https://x/{i}")]
    return fake


def test_scan_stops_at_the_first_missing_chunk(monkeypatch, tmp_path):
    """The feed's length is discovered by running off the end, and fetches now happen
    CHUNK_WORKERS at a time — so the wave that finds the 404 has to stop exactly there
    no matter how many chunks past it that same wave already downloaded. Must hold for
    every pool width, including one that straddles the boundary."""
    monkeypatch.setattr(research, "_speedrun_export", lambda: [])
    for workers in (1, 2, 3, 8):
        seen = []
        monkeypatch.setattr(research, "CHUNK_WORKERS", workers)
        monkeypatch.setattr(research, "RESEARCH_DB", tmp_path / f"w{workers}.db")
        monkeypatch.setattr(research, "_fetch_chunk", _canned_chunks(seen))
        total = research.scan(prof(titles=["engineer"]), progress=False)
        assert total == CHUNKS, f"{workers} workers: logged {total}, expected {CHUNKS}"
        # every real chunk was asked for, and the run stopped at the first 404
        assert set(range(CHUNKS)) <= set(seen)
        with closing(research.connect()) as conn:
            urls = {r[0] for r in conn.execute("SELECT url FROM market_matches")}
        assert urls == {f"https://x/{i}" for i in range(CHUNKS)}


def test_scan_without_titles_fetches_nothing(monkeypatch, tmp_path):
    # an empty profile matches nothing, so downloading 59 chunks to discard them is waste
    monkeypatch.setattr(research, "RESEARCH_DB", tmp_path / "empty.db")
    monkeypatch.setattr(research, "_fetch_chunk",
                        lambda i, retries=4: pytest.fail("fetched with no titles"))
    assert research.scan(prof(), progress=False) == 0
