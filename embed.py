#!/usr/bin/env python3
"""Embed each posting so the Resume Analysis tab can score it against your resume.

Run:  python3 embed.py                 (every alive row missing a vector)
      python3 embed.py --ids 3,7,12    (just those rows)
      python3 embed.py --all           (re-embed everything — use after changing EMBED_MODEL)

"""
import argparse
import json
import os
from contextlib import closing
from datetime import datetime

import numpy as np
from tqdm import tqdm

import research  # reuse research.db helpers; importing it does not touch the network

MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# 16 OOMs the GPU on a batch of long postings; measured peak is 45 GiB at 8, 23 GiB
# at 4. Cost scales as batch x seq_len^2, so raise EMBED_BATCH only if your longest
# description is well under the ~7.5k tokens this was measured against.
BATCH = int(os.environ.get("EMBED_BATCH", "4"))

# The instruction the resume is encoded with. Qwen3 ships one under the name "query",
# but it reads "Given a web search query, retrieve relevant passages that answer the
# query" — a task this app is not doing. Naming the real task widens the similarity
# spread 36% (std .0448 -> .0609 over 715 postings). This is Qwen's own template with
# the task slot rewritten, so it is coupled to that "Instruct: …\nQuery:" format —
# set EMBED_QUERY_PROMPT="" to fall back to whatever the model registers.
QUERY_PROMPT = os.environ.get(
    "EMBED_QUERY_PROMPT",
    "Instruct: Given a candidate's resume, retrieve job postings whose required skills, "
    "seniority and day-to-day responsibilities this candidate is a strong fit for.\n"
    "Query:")

_model = None


def load():
    """The SentenceTransformer, loaded once per process.

    Inputs:  none (MODEL is module state).
    Returns: the shared SentenceTransformer instance, constructing it on the first call.
    Notes:   sentence_transformers is imported in here, not at module scope, so app.py
             can `import embed` for pack/unpack without paying torch's multi-second
             import.
    Used by: embed().
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import, hence lazy
        _model = SentenceTransformer(MODEL)
    return _model


def embed(texts, is_query=False, progress=False):
    """Encode texts to L2-normalized vectors, so cosine similarity is a plain dot product.

    Inputs:  texts — an iterable of strings.
             is_query — encode these as the query side rather than the document side.
             progress — show sentence-transformers' own progress bar.
    Returns: a (len(texts), dim) numpy array of normalized vectors.

    Used by: run() for postings, run_resume() for the resume's sections.
    """
    m = load()
    kw = {}
    if is_query:
        if QUERY_PROMPT:
            kw["prompt"] = QUERY_PROMPT
        elif "query" in (getattr(m, "prompts", None) or {}):
            kw["prompt_name"] = "query"
    return m.encode(list(texts), normalize_embeddings=True, batch_size=BATCH,
                    show_progress_bar=progress, **kw)


# --- blob storage (pure; unit-tested without the model) --------------------

def pack(vecs):
    """One or more vectors -> bytes for a BLOB column.

    Inputs:  vecs — a 1-D vector or a 2-D array of them.
    Returns: contiguous float32 bytes, ready for sqlite.
    Used by: run(), run_resume().
    """
    return np.ascontiguousarray(np.atleast_2d(vecs), dtype=np.float32).tobytes()


def unpack(blob, n=1):
    """bytes -> an (n, dim) array.

    Inputs:  blob — bytes written by pack().
             n — how many vectors the blob holds.
    Returns: an (n, dim) float32 array.
    Raises:  ValueError if the blob does not divide evenly into n vectors.

    Used by: resume.rank, for both the resume's sections and the postings.
    """
    a = np.frombuffer(blob, dtype=np.float32)
    if n <= 0 or a.size % n:
        raise ValueError(f"blob of {a.size} floats does not divide into {n} vectors")
    return a.reshape(n, -1)


def doc_text(row):
    """What actually gets embedded for a posting.

    Inputs:  row — a market_matches row with title, skill_level, years_experience,
             skills and description.
    Returns: one string: the distilled fields first, then the description, with the
             absent fields left out.

    Used by: run().
    """
    parts = [row["title"] or ""]
    if row["skill_level"]:
        parts.append(f"Level: {row['skill_level']}")
    if row["years_experience"]:
        parts.append(f"Requires {row['years_experience']}+ years of experience")
    if row["skills"]:
        parts.append(f"Skills: {row['skills']}")
    if row["description"]:
        parts.append(row["description"])
    return "\n".join(p for p in parts if p)


# --- work queue -----------------------------------------------------------

COLS = "id, title, skill_level, years_experience, skills, description"


def rows_to_embed(conn, ids=None, redo=False):
    """The rows a run will embed.

    Inputs:  conn — an open research.db connection.
             ids — an explicit id list, which replaces the alive filter so you can
             re-embed a row you just enriched by hand. None means every alive row.
             redo — include rows that already carry a vector.
    Returns: a list of dicts holding COLS, ordered by id.
    Notes:   Mirrors enrich.rows_to_check.
    Used by: run().
    """
    where = []
    params = []
    if ids:
        where.append(f"id IN ({','.join('?' * len(ids))})")
        params += list(ids)
    else:
        where.append("alive = 1")
    if not redo:
        where.append("embedding IS NULL")
    q = f"SELECT {COLS} FROM market_matches WHERE {' AND '.join(where)} ORDER BY id"
    return [dict(r) for r in conn.execute(q, params)]


def run(ids=None, redo=False, progress=True):
    """Embed the postings that need it and store their vectors.

    Inputs:  ids — match ids to embed; None means every alive row missing a vector.
             redo — re-embed rows that already have one.
             progress — show the tqdm bar.
    Returns: how many postings were embedded. Stamps embedded_at.
    Used by: main(), and app.run_embed as a subprocess.
    """
    with closing(research.connect()) as conn:
        research.init_db(conn)
        todo = rows_to_embed(conn, ids, redo)
        if not todo:
            print("Nothing to embed.")
            return 0
        print(f"Embedding {len(todo)} posting(s) with {MODEL} …")
        now = datetime.now().isoformat(timespec="seconds")
        done = 0
        for i in tqdm(range(0, len(todo), BATCH), desc="Embedding", unit="batch",
                      disable=not progress):
            chunk = todo[i:i + BATCH]
            vecs = embed([doc_text(r) for r in chunk])
            conn.executemany(
                "UPDATE market_matches SET embedding = ?, embedded_at = ? WHERE id = ?",
                [(pack(v), now, r["id"]) for v, r in zip(vecs, chunk)],
            )
            conn.commit()
            done += len(chunk)
    print(f"Embedded {done} posting(s).")
    return done


def run_resume(progress=True):
    """Embed the saved resume's sections in place.

    Inputs:  progress — show the model's progress bar.
    Returns: how many sections were embedded; 0 if no resume is saved or it has
             no sections.

    Used by: main() via --resume, and app.run_embed(resume_only=True).
    """
    with closing(research.connect()) as conn:
        research.init_db(conn)
        row = conn.execute("SELECT sections FROM resume WHERE id = 1").fetchone()
        if not row:
            print("No resume saved.")
            return 0
        sections = json.loads(row["sections"] or "[]")
        if not sections:
            print("Resume has no sections to embed.")
            return 0
        print(f"Embedding {len(sections)} resume section(s) with {MODEL} …")
        vecs = embed(sections, is_query=True, progress=progress)
        conn.execute("UPDATE resume SET vectors = ? WHERE id = 1", (pack(vecs),))
        conn.commit()
    print(f"Embedded {len(sections)} resume section(s).")
    return len(sections)


def main():
    """Command-line entry point.

    Inputs:  none directly — reads sys.argv for --ids, --all and --resume.
    Returns: run_resume()'s count for --resume, otherwise None after running the
             posting backfill.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", default="", help="comma-separated match ids (default: every alive row)")
    ap.add_argument("--all", action="store_true", dest="redo",
                    help="re-embed rows that already have a vector (after changing EMBED_MODEL)")
    ap.add_argument("--resume", action="store_true",
                    help="embed the saved resume's sections instead of the postings")
    args = ap.parse_args()
    if args.resume:
        return run_resume()
    ids = [int(i) for i in args.ids.split(",") if i.strip()]
    run(ids or None, args.redo)


if __name__ == "__main__":
    main()
