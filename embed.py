"""Embed chunks with Segmind text-embedding-3-small and search them.

Usage:
    python embed.py                  # embed everything not yet embedded
    python embed.py --query "..."    # vector search
    python embed.py --stats

Why small over large: 1536-d at 0.9s/call vs 3072-d at 7.2s. At 411 chunks the
extra dimensions are not measurable; the 8x latency is.

Segmind's embedding endpoint takes ONE string. Passing a list silently returns
only the first element's vector -- batching here would corrupt the index.
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "text-embedding-3-small"
DIM = 1536

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id INTEGER PRIMARY KEY, dim INTEGER, vec BLOB, model TEXT);
"""


def api_key():
    key = os.environ.get("SEGMIND_API_KEY")
    if not key:
        path = os.path.join(HERE, ".env")
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.startswith("SEGMIND_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


class Embedder:
    def __init__(self, key, qps=8.0):
        self.key = key
        self.lock = threading.Lock()
        self.min_gap = 1.0 / qps
        self.last = 0.0

    def _throttle(self):
        with self.lock:
            gap = time.time() - self.last
            if gap < self.min_gap:
                time.sleep(self.min_gap - gap)
            self.last = time.time()

    def __call__(self, text, retries=3):
        if not isinstance(text, str):
            raise TypeError("one string per call; a list returns only the first vector")
        # MUST be "prompt". Segmind silently ignores an unknown "input" field and
        # returns a constant vector -- every chunk embeds identically and cosine
        # ranking degenerates to 1.0 everywhere. verify_api() guards this.
        body = json.dumps({"prompt": text}).encode()
        for attempt in range(retries):
            self._throttle()
            req = urllib.request.Request(
                f"https://api.segmind.com/v1/{MODEL}", data=body, method="POST",
                headers={"Content-Type": "application/json", "x-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    vec = json.loads(resp.read().decode())["embedding"]
                if len(vec) != DIM:
                    raise RuntimeError(f"expected {DIM} dims, got {len(vec)}")
                return vec
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"{type(e).__name__}: {e}") from None
        raise RuntimeError("retries exhausted")


def normalize(vec):
    """Store unit vectors so cosine similarity is a plain dot product at query time."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return array("f", vec)
    return array("f", [x / norm for x in vec])


def to_blob(vec):
    return normalize(vec).tobytes()


def from_blob(blob):
    out = array("f")
    out.frombytes(blob)
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def verify_api(embedder):
    """Two calls that prove the endpoint reads the text at all.

    A wrong parameter name gets silently ignored and returns a constant vector,
    which looks entirely plausible until every cosine in the index is 1.0.
    """
    a = normalize(embedder("the cat sat on the mat and purred softly"))
    b = normalize(embedder("quantum chromodynamics describes the strong nuclear force"))
    sim = dot(a, b)
    if sim > 0.9:
        raise RuntimeError(
            f"embedding endpoint ignores the input text (cos={sim:.6f} for unrelated "
            f"strings). Check the request parameter name -- refusing to build a "
            f"useless index.")
    return sim


def run(db, embedder, limit=None, workers=8):
    db.executescript(SCHEMA)
    print(f"api check: unrelated-text cosine = {verify_api(embedder):.4f} (want << 0.9)",
          flush=True)
    rows = db.execute(
        "SELECT id, text FROM chunks WHERE id NOT IN (SELECT chunk_id FROM embeddings)"
        f" ORDER BY id {'LIMIT ' + str(limit) if limit else ''}").fetchall()
    if not rows:
        print("nothing to embed")
        return
    print(f"embedding {len(rows)} chunks with {workers} workers...", flush=True)

    def work(row):
        cid, text = row[0], row[1]
        try:
            return cid, embedder(text), None
        except (RuntimeError, TypeError) as e:
            return cid, None, str(e)

    done = n_err = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in rows]):
            cid, vec, err = fut.result()
            done += 1
            if vec is None:
                n_err += 1
                print(f"  ! chunk {cid}: {err}", flush=True)
            else:
                db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                           (cid, DIM, to_blob(vec), MODEL))
                db.commit()  # per result: a kill must not discard work
            if done % 25 == 0 or done == len(rows):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(rows)}  {n_err} err  "
                      f"eta {(len(rows)-done)/max(rate,1e-9)/60:.1f}m", flush=True)
    print(f"done: {done - n_err} embedded, {n_err} errors "
          f"in {(time.time()-started)/60:.1f}m", flush=True)


def load_index(db):
    """All vectors in memory. 411 x 1536 floats is ~2.5 MB -- no ANN structure needed."""
    return [(r[0], from_blob(r[1]))
            for r in db.execute("SELECT chunk_id, vec FROM embeddings")]


def search(db, embedder, query, top_k=10, index=None):
    qvec = normalize(embedder(query))
    index = index if index is not None else load_index(db)
    scored = sorted(((dot(qvec, v), cid) for cid, v in index), reverse=True)[:top_k]
    if not scored:
        return []
    ids = ",".join(str(c) for _, c in scored)
    meta = {r[0]: r for r in db.execute(
        f"SELECT c.id, c.text, c.project, c.session_id, c.ts, s.title"
        f" FROM chunks c JOIN sessions s ON s.id=c.session_id WHERE c.id IN ({ids})")}
    out = []
    for score, cid in scored:
        if cid in meta:
            _, text, project, sid, ts, title = meta[cid]
            out.append({"chunk_id": cid, "score": score, "text": text,
                        "project": project, "session_id": sid, "ts": ts, "title": title})
    return out


def stats(db):
    n = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"embedded: {n}/{total} chunks  model={MODEL} dim={DIM}")
    if n:
        size = db.execute("SELECT SUM(LENGTH(vec)) FROM embeddings").fetchone()[0]
        print(f"index size: {size/1e6:.1f} MB in-db")


def selfcheck():
    v = [3.0, 4.0] + [0.0] * 10
    unit = normalize(v)
    assert abs(math.sqrt(sum(x * x for x in unit)) - 1.0) < 1e-6
    assert abs(unit[0] - 0.6) < 1e-6 and abs(unit[1] - 0.8) < 1e-6
    assert normalize([0.0] * 5).tolist() == [0.0] * 5, "zero vector must not divide by zero"

    round_trip = from_blob(to_blob([1.0, 2.0, 3.0]))
    assert len(round_trip) == 3
    assert abs(dot(round_trip, round_trip) - 1.0) < 1e-6, "blob round-trip keeps unit norm"

    a, b = normalize([1.0, 0.0]), normalize([0.0, 1.0])
    assert abs(dot(a, a) - 1.0) < 1e-6, "self similarity is 1"
    assert abs(dot(a, b)) < 1e-6, "orthogonal vectors score 0"
    assert dot(a, normalize([1.0, 1.0])) > dot(a, b), "closer vector must rank higher"

    # a list input must raise, never silently embed only the first element
    try:
        Embedder("fake")(["a", "b"])
    except TypeError:
        pass
    else:
        raise AssertionError("list input must be rejected")
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--query")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit()

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    key = api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")

    if args.query:
        t = time.time()
        hits = search(con, Embedder(key), args.query, top_k=args.top_k)
        print(f"{len(hits)} hits in {time.time()-t:.2f}s\n")
        for h in hits:
            head = " ".join(h["text"].split())[:150]
            print(f"  {h['score']:.3f}  [{h['project'][:26]}] {h['title'] or ''}")
            print(f"         {head}\n")
    elif args.stats:
        stats(con)
    else:
        run(con, Embedder(key), limit=args.limit, workers=args.workers)
        print()
        stats(con)
