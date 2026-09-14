"""Embed entity names so graph seeds can be resolved semantically.

Usage:
    python embed_entities.py            # embed entities without a vector yet
    python embed_entities.py --stats

Lexical seed matching only fires when the query reuses the extractor's exact
wording. "majority voting" never matches `ensemble_voting_decision`, so the graph
side of retrieval silently returns nothing. These vectors fix that.

An entity is embedded as "<name> (<kind>)" -- the kind disambiguates same-name
entities from unrelated contexts (Chrome the browser vs chrome.storage.local).
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import embed

HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_vecs (
  entity_id INTEGER PRIMARY KEY, vec BLOB, model TEXT);
"""


def run(db, embedder, limit=None, workers=10):
    db.executescript(SCHEMA)
    print(f"api check: unrelated-text cosine = {embed.verify_api(embedder):.4f}",
          flush=True)
    rows = db.execute(
        "SELECT id, name, kind FROM entities"
        " WHERE id NOT IN (SELECT entity_id FROM entity_vecs)"
        f" ORDER BY id {'LIMIT ' + str(limit) if limit else ''}").fetchall()
    if not rows:
        print("nothing to embed")
        return
    print(f"embedding {len(rows)} entity names with {workers} workers...", flush=True)

    def work(row):
        eid, name, kind = row
        try:
            return eid, embedder(f"{name} ({kind})"), None
        except (RuntimeError, TypeError) as e:
            return eid, None, str(e)

    done = n_err = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in rows]):
            eid, vec, err = fut.result()
            done += 1
            if vec is None:
                n_err += 1
            else:
                db.execute("INSERT OR REPLACE INTO entity_vecs VALUES (?,?,?)",
                           (eid, embed.to_blob(vec), embed.MODEL))
                db.commit()
            if done % 100 == 0 or done == len(rows):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(rows)}  {n_err} err  "
                      f"eta {(len(rows)-done)/max(rate,1e-9)/60:.1f}m", flush=True)
    print(f"done: {done - n_err} embedded, {n_err} errors "
          f"in {(time.time()-started)/60:.1f}m", flush=True)


def load(db):
    return db.execute("SELECT entity_id, vec FROM entity_vecs").fetchall()


def stats(db):
    n = db.execute("SELECT COUNT(*) FROM entity_vecs").fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"entity vectors: {n}/{total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    if args.stats:
        stats(con)
        sys.exit()
    key = embed.api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")
    run(con, embed.Embedder(key), limit=args.limit, workers=args.workers)
    print()
    stats(con)
