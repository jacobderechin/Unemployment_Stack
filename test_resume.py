#!/usr/bin/env python3
"""Tests for resume.py's pure logic: extraction guards, section splitting, and the
six-feature scorer. No network, no Ollama, no model — resume.py keeps torch behind
embed.load(), which nothing here calls.

Run:  python3 test_resume.py
"""
import sqlite3

import numpy as np

import embed
import research
import resume
from datetime import date


def eq(got, want, msg):
    assert got == want, f"{msg}: got {got!r}, want {want!r}"


# --- years_fit: the sentinel is the whole point ----------------------------

def test_years_fit():
    # 0 means "the posting didn't say", not "needs zero years". If this ever returns a
    # number, every un-enriched posting becomes a perfect match for everyone.
    eq(resume.years_fit(0, 5), None, "unstated years abstains")
    eq(resume.years_fit(None, 5), None, "null years abstains")

    eq(resume.years_fit(3, 5), 1.0, "you exceed the minimum")
    eq(resume.years_fit(5, 5), 1.0, "you meet it exactly")
    eq(resume.years_fit(7, 5), 0.7, "2 years short is a stretch")
    eq(resume.years_fit(10, 5), 0.3, "5 years short")
    eq(resume.years_fit(15, 5), 0.0, "10 years short")
    # an unset candidate years counts as 0, so the gap is scored rather than skipped
    eq(resume.years_fit(3, None), 0.3, "unknown candidate years treated as 0")
    eq(resume.years_fit(3, 0), 0.3, "0 candidate years still scores the gap")


# --- coverage: their asks, not Jaccard ------------------------------------

def test_coverage():
    r = resume.skill_set("Python, Kubernetes, React")
    eq(resume.coverage("", r), None, "posting with no skills abstains")
    eq(resume.coverage(None, r), None, "null skills abstains")
    eq(resume.coverage("Python, Kubernetes", r), 1.0, "you have both asks")
    eq(resume.coverage("Python, Go", r), 0.5, "you have one of two")
    eq(resume.coverage("Go, Rust", r), 0.0, "you have neither")

    # extras must not dilute: a posting asking for one thing you have is a full match
    # however long your resume is
    wide = resume.skill_set("Python, Go, Rust, C, Haskell, SQL, Terraform")
    eq(resume.coverage("Python", wide), 1.0, "extra resume skills do not penalize")

    eq(resume.coverage("python", resume.skill_set("PYTHON")), 1.0, "case-insensitive")
    eq(resume.coverage("Python ,  Go ", resume.skill_set(" python,go")), 1.0, "whitespace tolerant")


def test_skill_set():
    eq(resume.skill_set(""), set(), "empty string")
    eq(resume.skill_set(None), set(), "None")
    eq(resume.skill_set("A, , B,"), {"a", "b"}, "blank entries dropped")


# --- the other features ---------------------------------------------------

def test_level_fit():
    eq(resume.level_fit(None, ["senior"]), None, "unlabelled posting abstains")
    eq(resume.level_fit("senior", []), None, "no stated preference abstains")
    eq(resume.level_fit("senior", ["senior", "mid"]), 1.0, "level wanted")
    eq(resume.level_fit("intern", ["senior"]), 0.3, "level not wanted")


def test_no_title_feature():
    """research.matches() already drops postings whose title misses the target titles,
    so a title term could only ever return 1.0. It must not be scored."""
    assert "title" not in resume.WEIGHTS, "title is tautological, not a feature"
    assert not hasattr(resume, "title_fit"), "title_fit should be gone, not just unweighted"
    eq(round(sum(resume.WEIGHTS.values()), 6), 1.0, "weights sum to 1")


def test_recency():
    today = date(2026, 7, 25)
    eq(resume.recency(None, today), None, "missing date abstains")
    eq(resume.recency("not-a-date", today), None, "unparseable date abstains")
    eq(resume.recency("2026-07-25", today), 1.0, "found today")
    eq(resume.recency("2026-05-01", today), 0.0, "older than the window floors at 0")
    assert 0.4 < resume.recency("2026-06-25", today) < 0.6, "30 days of a 60-day window"
    eq(resume.recency("2026-08-01", today), 1.0, "a future date clamps to 1, not >1")


# --- combine: abstention must not become a penalty ------------------------

def test_combine():
    eq(resume.combine({}), 0.0, "no features at all does not divide by zero")
    eq(resume.combine({k: None for k in resume.WEIGHTS}), 0.0, "all-None does not divide by zero")
    eq(resume.combine({"skills": 1.0}), 1.0, "a single present feature is its own mean")
    eq(resume.combine({"skills": 0.5}), 0.5, "single feature passes through")

    full = {"semantic": 0.8, "skills": 1.0, "years": 1.0, "level": 1.0, "recency": 1.0}
    assert resume.combine(dict(full, years=None, level=None)) > \
        resume.combine(dict(full, years=0.0, level=0.0)), "missing must beat scoring zero"

    # custom weights (what the UI's importance chips send) are honoured
    eq(resume.combine({"skills": 1.0, "semantic": 0.0}, {"skills": 1, "semantic": 0}), 1.0,
       "zero-weighted feature drops out")


def test_combine_imputes_missing():
    """The regression that mattered: dropping a missing term and renormalizing rewards
    postings with no data, because the features are not equally generous."""
    fills = {"semantic": 0.5, "skills": 0.17, "years": 0.75, "level": 1.0,
             "recency": 0.79}

    # a missing feature scores exactly as if it held the corpus mean — no bonus, no penalty
    miss = {"semantic": 0.6, "skills": None, "years": 0.7, "level": 1.0, "recency": 0.8}
    same = dict(miss, skills=fills["skills"])
    eq(round(resume.combine(miss, fills=fills), 9), round(resume.combine(same, fills=fills), 9),
       "missing == average, exactly")

    # and the actual bug: a posting with no skills data must NOT outrank one that has
    # above-average coverage, all else equal
    good = dict(miss, skills=0.60)
    assert resume.combine(good, fills=fills) > resume.combine(miss, fills=fills), \
        "real above-average skills must beat missing skills"
    # nor should it be punished below a genuinely poor match
    bad = dict(miss, skills=0.02)
    assert resume.combine(miss, fills=fills) > resume.combine(bad, fills=fills), \
        "missing skills must beat genuinely poor coverage"

    # with no fill available anywhere, the term drops out rather than poisoning the score
    eq(resume.combine({"skills": None, "recency": 1.0}, fills={}), 1.0,
       "no fill available -> renormalize over what is left")


def test_feature_means():
    fs = [{"skills": 0.2, "years": None}, {"skills": 0.4, "years": 1.0}, {"skills": None, "years": None}]
    m = resume.feature_means(fs)
    eq(round(m["skills"], 6), 0.3, "mean over present values only")
    eq(m["years"], 1.0, "single present value")
    assert "level" not in m, "a feature absent everywhere gets no fill"
    eq(resume.feature_means([]), {}, "empty result set")


def test_features_for():
    prof = {"years_experience": 5, "levels": ["senior"], "titles": ["Engineer"]}
    row = {"skills": "Python, Go", "years_experience": 0, "skill_level": None,
           "title": "Staff Engineer", "found_at": None}
    f = resume.features_for(row, resume.skill_set("Python"), prof, semantic=0.5,
                            today=date(2026, 7, 25))
    eq(f["semantic"], 0.5, "semantic passed through")
    eq(f["skills"], 0.5, "one of two skills")
    eq(f["years"], None, "years sentinel abstains through features_for")
    eq(f["level"], None, "null level abstains")
    eq(f["recency"], None, "missing found_at abstains")
    eq(set(f), set(resume.WEIGHTS), "features_for emits exactly the weighted features")
    # and the whole thing still scores, on the two features that are present
    assert 0 < resume.combine(f) < 1, "sparse row still produces a usable score"


# --- section splitting ----------------------------------------------------

def test_split_sections():
    eq(resume.split_sections(""), [], "empty text")
    eq(resume.split_sections("   \n\n  "), [], "whitespace only")
    eq(resume.split_sections(None), [], "None is not a crash")

    # one vector, whatever the blank-line structure: splitting was measured to change
    # the ranking by r=0.990 while leaving 2 of 4 chunks winning nothing at all
    many = "\n\n".join(f"{'x' * 250}{i}" for i in range(20))
    eq(resume.split_sections(many), [many], "blank lines no longer split")
    eq(resume.split_sections("  padded  "), ["padded"], "stripped, not split")


# --- extract_text guards (the trust boundary) -----------------------------

def test_extract_text_guards():
    for name in ("resume.exe", "resume.docx", "resume", "resume.pdf.sh"):
        try:
            resume.extract_text(b"data", name)
            assert False, f"{name} should be rejected"
        except ValueError as e:
            assert "unsupported" in str(e), f"{name}: wrong error: {e}"

    try:
        resume.extract_text(b"x" * (resume.MAX_BYTES + 1), "r.txt")
        assert False, "oversized file should be rejected"
    except ValueError as e:
        assert "too large" in str(e), f"wrong error: {e}"

    try:
        resume.extract_text(b"", "r.txt")
        assert False, "empty file should be rejected"
    except ValueError as e:
        assert "empty" in str(e), f"wrong error: {e}"

    eq(resume.extract_text(b"hello", "r.txt"), "hello", "plain text passes through")
    eq(resume.extract_text(b"# hi", "R.MD"), "# hi", "extension check is case-insensitive")
    # undecodable bytes must not raise — a resume with a stray byte is still readable
    assert resume.extract_text(b"caf\xff", "r.txt"), "bad bytes replaced, not fatal"


# --- sanitize -------------------------------------------------------------

def test_sanitize():
    eq(resume.sanitize({"years_experience": 7, "skills": ["Python", "Go"]}),
       {"years_experience": 7, "skills": "Python, Go"}, "happy path")
    eq(resume.sanitize({})["years_experience"], 0, "missing years -> 0")
    eq(resume.sanitize({"years_experience": 999, "skills": []})["years_experience"], 0,
       "absurd years rejected")
    eq(resume.sanitize({"years_experience": "five", "skills": []})["years_experience"], 0,
       "non-int years rejected")
    eq(resume.sanitize({"skills": ["Python", "python", "PYTHON"]})["skills"], "Python",
       "deduped case-insensitively, first spelling kept")
    eq(resume.sanitize({"skills": [" Go ", "", None, 3]})["skills"], "Go",
       "blanks and non-strings dropped")
    n = len(resume.sanitize({"skills": [f"s{i}" for i in range(200)]})["skills"].split(", "))
    eq(n, resume.MAX_SKILLS, "skills capped")
    eq(len(resume.sanitize({"skills": ["x" * 200]})["skills"]), 50, "each skill clipped to 50")


# --- blob round-trip ------------------------------------------------------

def test_pack_unpack():
    v = np.random.rand(1, 8).astype(np.float32)
    got = embed.unpack(embed.pack(v), n=1)
    eq(got.shape, (1, 8), "single vector shape")
    assert np.allclose(got, v), "single vector round-trips"

    many = np.random.rand(5, 8).astype(np.float32)
    got = embed.unpack(embed.pack(many), n=5)
    eq(got.shape, (5, 8), "multi-vector shape")
    assert np.allclose(got, many), "multi-vector round-trips"
    eq(got.dtype, np.dtype("float32"), "stays float32")

    # a 1-D vector packs as one row, so callers need not pre-shape
    eq(embed.unpack(embed.pack(np.zeros(8, dtype=np.float32))).shape, (1, 8), "1-D input")

    # dim is derived, so an inconsistent n is caught rather than silently reshaped
    for bad_n in (3, 0, -1):
        try:
            embed.unpack(embed.pack(many), n=bad_n)
            assert False, f"n={bad_n} should raise"
        except ValueError:
            pass


def test_doc_text():
    row = {"title": "ML Engineer", "skill_level": "senior", "years_experience": 5,
           "skills": "Python, PyTorch", "description": "d" * 5000}
    t = embed.doc_text(row)
    assert t.startswith("ML Engineer"), "title leads"
    assert "senior" in t and "5+ years" in t and "PyTorch" in t, "distilled fields present"
    # uncut: enrich.py already narrowed description to the span worth embedding, so
    # truncating here would clip the very text that was extracted
    eq(t.count("d"), 5000, "description embedded whole")

    # every optional field absent, including the years sentinel, must not emit noise
    bare = embed.doc_text({"title": "Analyst", "skill_level": None, "years_experience": 0,
                           "skills": "", "description": ""})
    eq(bare, "Analyst", "bare row is just the title")


def test_set_skills():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    research.init_db(conn)
    # rollback() before reading: a no-op after a real commit, but if set_skills ever
    # stops committing this throws the write away and the assert below catches it —
    # app.py closes its connection after every request, so an uncommitted write is lost.
    def read():
        conn.rollback()
        return conn.execute("SELECT skills FROM resume WHERE id = 1").fetchone()["skills"]

    eq(resume.set_skills(conn, ["Python"]), None, "no resume row -> None, not a silent no-op")
    conn.execute("INSERT INTO resume (id, text, skills) VALUES (1, 'cv', 'Python, Go')")
    conn.commit()
    eq(resume.set_skills(conn, ["Go", "go", " Rust "]), "Go, Rust", "clamped like the extractor's output")
    eq(read(), "Go, Rust", "written, replacing the old list")
    eq(resume.set_skills(conn, ["B.S., Computer Science"]), "B.S., Computer Science",
       "a typed comma splits into two skills, not one that splits itself on the next read")
    eq(read().split(", "), ["B.S.", "Computer Science"], "stored as two entries")
    eq(resume.set_skills(conn, []), "", "removing the last skill is allowed")
    try:
        resume.set_skills(conn, "Python")     # a bare string would iterate per character
        assert False, "a non-list must be rejected"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
