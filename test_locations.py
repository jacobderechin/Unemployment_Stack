"""Unit tests for locations.parse() — the pure "trust nothing the model returned"
step — and the location_map join in research.list_matches().
Runs without Ollama and without network:  pytest test_locations.py
"""
import sqlite3

import locations
import research


def place(i, *cities):
    return {"i": i, "cities": list(cities)}


def test_maps_each_line_to_comma_joined_cities():
    batch = ["Hybrid (NYC Metro)", "SF | NYC"]
    got = locations.parse([place(0, "New York"), place(1, "San Francisco", "New York")], batch)
    assert got == {"Hybrid (NYC Metro)": "New York", "SF | NYC": "San Francisco,New York"}


def test_answers_may_arrive_out_of_order_or_incomplete():
    batch = ["NYC", "Berlin, DE", "Remote - US"]
    got = locations.parse([place(2, "Remote"), place(0, "New York")], batch)
    assert got == {"Remote - US": "Remote", "NYC": "New York"}   # line 1 simply stays pending


def test_discards_an_index_outside_the_batch():
    # a hallucinated line number must not write a mapping onto the wrong raw string
    batch = ["Hybrid (NYC Metro)"]
    assert locations.parse([place(7, "New York")], batch) == {}
    assert locations.parse([place(-1, "New York")], batch) == {}


def test_dedupes_repeated_cities_keeping_order():
    got = locations.parse([place(0, "San Francisco", "San Francisco")],
                          ["SF Bay Area, San Francisco, CA"])
    assert got == {"SF Bay Area, San Francisco, CA": "San Francisco"}


def test_survives_a_junk_response():
    batch = ["Remote - US"]
    assert locations.parse(None, batch) == {}
    assert locations.parse([], batch) == {}
    assert locations.parse(["New York"], batch) == {}          # list of strings, not objects
    assert locations.parse([{"i": 0}], batch) == {"Remote - US": ""}
    assert locations.parse([place(0, "", "  ")], batch) == {"Remote - US": ""}


def test_places_of_finds_the_array_under_any_wrapper_key():
    # Ollama honours the inner object shape but not the top-level property name
    want = [{"i": 0, "cities": ["New York"]}]
    assert locations.places_of({"places": want}) == want
    assert locations.places_of({"results": want}) == want
    assert locations.places_of({"entries": want}) == want
    assert locations.places_of(want) == want                   # bare array
    assert locations.places_of({"note": "none found"}) == []
    assert locations.places_of(None) == []


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    research.init_db(conn)
    return conn


def test_list_matches_joins_the_mapping_in():
    conn = _db()
    research.upsert_match(conn, {"url": "u1", "title": "Data Engineer", "company": "Acme",
                                 "location": "Hybrid (NYC Metro)", "skill_level": "senior"})
    conn.execute("INSERT INTO location_map (raw, cities) VALUES (?, ?)",
                 ("Hybrid (NYC Metro)", "New York"))
    row = research.list_matches(conn, {"titles": []})[0]
    assert row["cities"] == "New York"


def test_unmapped_location_yields_null_cities():
    conn = _db()
    research.upsert_match(conn, {"url": "u1", "title": "Data Engineer", "company": "Acme",
                                 "location": "Kraków, Poland", "skill_level": "senior"})
    row = research.list_matches(conn, {"titles": []})[0]
    assert row["cities"] is None          # the browser falls back to its own bucketing


def test_upsert_strips_the_location():
    # the feed writes "New York Office " and "New York Office" as separate values; unstripped
    # they enter location_map as two keys and cost two model calls for one answer
    conn = _db()
    research.upsert_match(conn, {"url": "u1", "title": "Data Engineer", "company": "Acme",
                                 "location": "  New York Office  ", "skill_level": "senior"})
    assert research.list_matches(conn, {"titles": []})[0]["location"] == "New York Office"
