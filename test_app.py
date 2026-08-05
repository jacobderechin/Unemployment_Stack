#!/usr/bin/env python3
"""Self-check for app.py's DB logic. Run: python3 test_app.py"""
import sqlite3
import threading
import time
from datetime import date

import app


def main():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    app.init_db(conn)
    assert app.list_jobs(conn) == []

    job = app.create_job(conn, "Acme", "Backend Eng", "2026-06-10")
    assert job["status"] == "applied" and job["company"] == "Acme"
    assert [j["id"] for j in app.list_jobs(conn)] == [job["id"]]

    # required-field validation at the boundary
    for bad in [("", "t", "2026-01-01"), ("c", "", "2026-01-01"), ("c", "t", "  ")]:
        try:
            app.create_job(conn, *bad)
            assert False, "expected ValueError for empty field"
        except ValueError:
            pass

    # moving to a milestone stamps its date (today, on first arrival)
    assert app.update_job(conn, job["id"], {"status": "interviewing"}) is True
    assert app.list_jobs(conn)[0]["status"] == "interviewing"
    assert app.list_jobs(conn)[0]["interview_date"] == date.today().isoformat()

    # an existing stamp survives leaving and re-entering the column
    app.update_job(conn, job["id"], {"interview_date": "2026-06-01"})
    app.update_job(conn, job["id"], {"status": "applied"})
    app.update_job(conn, job["id"], {"status": "interviewing"})
    assert app.list_jobs(conn)[0]["interview_date"] == "2026-06-01"

    # milestone dates are editable, and a bad date is rejected
    app.update_job(conn, job["id"], {"rejected_date": "2026-06-20"})
    assert app.list_jobs(conn)[0]["rejected_date"] == "2026-06-20"
    try:
        app.update_job(conn, job["id"], {"interview_date": "not-a-date"})
        assert False, "expected ValueError for bad date"
    except ValueError:
        pass

    assert app.update_job(conn, 9999, {"status": "offer"}) is False  # unknown id

    # invalid status rejected
    try:
        app.update_job(conn, job["id"], {"status": "hired"})
        assert False, "expected ValueError for bad status"
    except ValueError:
        pass

    # edit company, title and applied date
    app.update_job(conn, job["id"], {"company": "Globex", "title": "SRE", "applied_date": "2026-05-01"})
    row = app.list_jobs(conn)[0]
    assert (row["company"], row["title"], row["applied_date"]) == ("Globex", "SRE", "2026-05-01")

    # required fields stay required; applied_date must be a real date
    for bad in [{"company": "  "}, {"title": ""}, {"applied_date": "nope"}]:
        try:
            app.update_job(conn, job["id"], bad)
            assert False, "expected ValueError"
        except ValueError:
            pass

    # a milestone date stamped by mistake can be cleared
    app.update_job(conn, job["id"], {"interview_date": ""})
    assert app.list_jobs(conn)[0]["interview_date"] is None

    # delete
    assert app.delete_job(conn, job["id"]) is True
    assert app.list_jobs(conn) == []

    check_embed_targets()
    check_jobs()
    print("ok")


def check_embed_targets():
    """Which rows a check hands to embed.py, per mode."""
    import research
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    research.init_db(conn)
    conn.executemany(
        "INSERT INTO market_matches (id, url, title, company, years_experience) "
        "VALUES (?, ?, 'T', 'C', ?)",
        [(1, "u1", 5), (2, "u2", None), (3, "u3", 0)],   # 2 is unenriched; 0 != NULL
    )

    assert app.embed_targets(conn, [1, 2, 3], "full") == [1, 2, 3]
    assert app.embed_targets(conn, [1, 2, 3], "new") == [2]     # only the never-enriched row
    assert app.embed_targets(conn, [1, 2, 3], "alive") == []    # no model calls, nothing to redo
    assert app.embed_targets(conn, [], "full") == []


def check_jobs():
    """The background job runner: one at a time, and a result that is never half-visible."""
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return {"ok": True}

    app.JOB = None
    assert app.start_job("slow", slow) is True
    assert started.wait(5), "job never started"
    # a second job is refused while the first runs — they would fight over the same DBs
    assert app.start_job("other", lambda: {"ok": True}) is False
    s = app.job_status()
    assert s["name"] == "slow" and s["done"] is False and s["result"] is None

    release.set()
    wait_done()
    assert app.job_status()["result"] == {"ok": True}
    assert app.job_status()["error"] is None

    # with the first finished, the slot is free again — and a raised error is reported,
    # not swallowed, or the browser would poll a job that never resolves
    assert app.start_job("boom", lambda: 1 / 0) is True
    wait_done()
    s = app.job_status()
    assert s["done"] is True and s["result"] is None and "division by zero" in s["error"]

    app.JOB = None
    assert app.job_status() == {"name": None, "done": True, "result": None,
                                "error": None, "elapsed": 0}


def wait_done(timeout=5):
    """Poll job_status the way the browser does; `done` implies the result is there."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = app.job_status()
        if s["done"]:
            return s
        time.sleep(0.01)
    raise AssertionError("job never finished")


if __name__ == "__main__":
    main()
