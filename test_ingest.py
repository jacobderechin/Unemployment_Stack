#!/usr/bin/env python3
"""Self-check for ingest.reconcile / apply_action (no network).
Run: python3 test_ingest.py"""
import sqlite3

import app
import ingest

EMAIL = {"from": "Jobs <jobs@acme.com>", "subject": "x",
         "date": "Mon, 15 Jun 2026 10:00:00 +0000", "body": "x", "id": "m1"}


def conn_with(*jobs):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    app.init_db(c)
    for company, title, applied in jobs:
        app.create_job(c, company, title, applied)
    return c


def x(**kw):
    base = {"is_application": True, "kind": "new_application",
            "company": "Acme", "title": "", "event_date": "2026-06-10"}
    base.update(kw)
    return base


def main():
    # new application -> ADD (applied)
    a = ingest.reconcile(x(title="Backend Eng"), EMAIL, conn_with())
    assert a["op"] == "add" and a["status"] == "applied" and a["title"] == "Backend Eng", a

    # duplicate (company, title) -> SKIP
    c = conn_with(("Acme", "Backend Eng", "2026-06-10"))
    assert ingest.reconcile(x(title="Backend Eng"), EMAIL, c)["op"] == "skip"

    # same company, different title -> ADD (no false dedup)
    assert ingest.reconcile(x(title="Frontend Eng"), EMAIL, c)["op"] == "add"

    # rejection matching (company, title) -> UPDATE with rejected + email's date
    c = conn_with(("Acme", "Backend Eng", "2026-06-10"))
    a = ingest.reconcile(x(kind="rejection", title="Backend Eng", event_date="2026-06-18"), EMAIL, c)
    assert a["op"] == "update" and a["status"] == "rejected" and a["date"] == "2026-06-18", a

    # title-less status email, one company match -> UPDATE
    c = conn_with(("Acme", "Backend Eng", "2026-06-10"))
    a = ingest.reconcile(x(kind="interview", title=""), EMAIL, c)
    assert a["op"] == "update" and a["status"] == "interviewing", a

    # title-less status email, multiple matches -> FLAG (no change)
    c = conn_with(("Acme", "Backend Eng", "2026-06-10"), ("Acme", "Frontend Eng", "2026-06-11"))
    assert ingest.reconcile(x(kind="rejection", title=""), EMAIL, c)["op"] == "flag"

    # title-less new application -> ADD with default title
    a = ingest.reconcile(x(title=""), EMAIL, conn_with())
    assert a["op"] == "add" and a["title"] == ingest.DEFAULT_TITLE, a

    # status email with no prior match -> ADD already in that status
    a = ingest.reconcile(x(kind="rejection", title="Backend Eng"), EMAIL, conn_with())
    assert a["op"] == "add" and a["status"] == "rejected", a

    # not an application -> SKIP
    assert ingest.reconcile(x(is_application=False, kind="other"), EMAIL, conn_with())["op"] == "skip"

    # apply_action writes through: ADD-in-rejected -> create then stamp status+date
    c = conn_with()
    a = ingest.reconcile(x(kind="rejection", title="SRE", event_date="2026-06-20"), EMAIL, c)
    ingest.apply_action(a, c)
    row = app.list_jobs(c)[0]
    assert row["status"] == "rejected" and row["rejected_date"] == "2026-06-20", dict(row)

    # apply_action UPDATE path stamps the interview date
    c = conn_with(("Globex", "SRE", "2026-06-01"))
    a = ingest.reconcile(x(company="Globex", kind="interview", title="SRE", event_date="2026-06-09"), EMAIL, c)
    ingest.apply_action(a, c)
    row = app.list_jobs(c)[0]
    assert row["status"] == "interviewing" and row["interview_date"] == "2026-06-09", dict(row)

    # sanitize removes the escape/control bytes (neutralizing the sequence),
    # keeps printable text and tab/newline
    out = ingest._sanitize("a\x1b[31mb\x07c\tx\ny")
    assert "\x1b" not in out and "\x07" not in out and out == "a[31mbc\tx\ny", repr(out)

    # html path drops script/style bodies and unescapes entities
    t = ingest._html_to_text("<script>bad()</script><p>Hi&amp;bye</p>")
    assert "bad()" not in t and "Hi&bye" in t, t

    # ignore-list matches by domain and by full address, not unrelated senders
    ingest.IGNORE_SENDERS = ["indeed.com", "news@foo.com"]
    assert ingest._ignored("Jobs <alerts@indeed.com>")
    assert ingest._ignored("News <news@foo.com>")
    assert not ingest._ignored("Recruiter <r@acme.com>")

    # dedupe: two same-run emails for one new role collapse to a single card,
    # most-advanced status wins, applied_date stays the earliest
    e1, e2 = {**EMAIL, "id": "a1"}, {**EMAIL, "id": "a2"}
    a1 = ingest.reconcile(x(kind="new_application", title="Data Eng", event_date="2026-06-01"), e1, conn_with())
    a2 = ingest.reconcile(x(kind="interview", title="Data Eng", event_date="2026-06-09"), e2, conn_with())
    a1["email"], a2["email"] = e1, e2
    merged = ingest.dedupe([a1, a2])
    adds = [m for m in merged if m["op"] == "add"]
    assert len(adds) == 1 and adds[0]["status"] == "interviewing", merged
    assert adds[0]["applied_date"] == "2026-06-01" and adds[0]["date"] == "2026-06-09", adds[0]
    # the loser becomes a skip so its email is still marked seen on commit
    assert any(m["op"] == "skip" and m["email"]["id"] == "a2" for m in merged), merged
    # and it commits to one row with the right dates
    c = conn_with()
    ingest.apply_action(adds[0], c)
    row = app.list_jobs(c)[0]
    assert (row["applied_date"] == "2026-06-01" and row["status"] == "interviewing"
            and row["interview_date"] == "2026-06-09"), dict(row)

    # _retry_batches: retries leftovers with backoff until none remain
    calls = []
    def shrink(ids):
        calls.append(list(ids))
        return ids[1:]                      # one fewer needs retry each round
    assert ingest._retry_batches([1, 2, 3], shrink, sleep=lambda _: None) == []
    assert len(calls) == 3, calls
    # gives up (returns leftovers) once attempts run out
    left = ingest._retry_batches([1, 2], lambda ids: ids, sleep=lambda _: None, attempts=4)
    assert left == [1, 2], left

    print("ok")


if __name__ == "__main__":
    main()
