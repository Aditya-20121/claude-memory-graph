"""Extract (entity, relation, entity) triples from chunks via Segmind gpt-5-nano.

Usage:
    python extract.py --limit 20     # dry run on 20 chunks
    python extract.py                # everything not yet extracted
    python extract.py --stats        # graph stats only
    python extract.py --retry        # re-attempt chunks that errored

Resumable: a chunk is skipped once it has a row in `extracted`.
One call per chunk -- gpt-5-nano costs ~0.000075 credits/call, so batching would
trade exact provenance for a saving too small to measure.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from schema import ENTITY_KINDS, PREDICATES, SCHEMA_SQL

ENDPOINT = "https://api.segmind.com/v1/gpt-5-nano"
HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT = """Extract a knowledge graph from one exchange between a developer and an AI assistant.

Return ONLY this JSON, no prose:
{{"triples": [{{"s": "<subject>", "sk": "<subject kind>", "p": "<predicate>", "o": "<object>", "ok": "<object kind>"}}]}}

"p" MUST be one of: {preds}
"sk" and "ok" MUST be one of: {kinds}

An ENTITY is a NAMED THING that could be mentioned again in a different
conversation six months from now. Both "s" and "o" must be entities.

NEVER emit as an entity:
- a measurement or quantity ("464 MB", "19 GB", "3.7 GB of stale leftovers", "40%")
- a status or outcome ("SAFE TO DELETE", "cleaned", "Completed so far", "done", "fixed")
- a date, a time, or a version number on its own
- a vague adjective or summary phrase ("redundant", "the main issue", "best practice")
- the mechanics of chatting ("assistant", "user", "the request", "code", "file")

Use predicates for their real meaning, never as value assignment:
- WRONG: (disk_free_space) -[evaluates]-> (20.7 GB)     <- quantity as object
- WRONG: (npm-cache) -[decided]-> (SAFE TO DELETE)      <- status as object
- WRONG: (HuggingFace cache) -[causes]-> (464 MB)       <- quantity as object
- RIGHT: (VAD) -[uses]-> (VisDrone)
- RIGHT: (384px downscale) -[causes]-> (detection flicker)
- RIGHT: (two-stage cascade) -[fixes]-> (inference latency blocker)
- RIGHT: (Aditya) -[decided]-> (two-stage cascade)
- RIGHT: (capacity_probe.py) -[evaluates]-> (false alarm rate)

"decided" takes a person as subject and a decision as object.
Prefer the specific and reusable: model names, library names, metrics, file
names, techniques, blockers, decisions.

0 to 8 triples. Return {{"triples": []}} if the exchange states no durable facts.
Quality over quantity -- an empty result is better than a vague one.

EXCHANGE (project: {project}):
{text}"""

# Deterministic backstop for what the prompt still lets through.
STOPWORDS = {
    "assistant", "user", "the user", "claude", "request", "response", "code", "file",
    "files", "output", "input", "result", "results", "error", "issue", "problem",
    "task", "project", "thing", "stuff", "it", "this", "that", "them", "data",
    "done", "fixed", "cleaned", "complete", "completed", "completed so far",
    "safe to delete", "success", "failed", "pending", "n/a", "none", "ok", "yes", "no",
    "redundant", "unnecessary", "best practice", "the main issue", "current status",
}
QUANTITY = re.compile(
    r"^[\d.,\s]*\d[\d.,\s]*\s*"
    r"(%|[kmgt]?b|bytes?|ms|s|sec(onds?)?|min(utes?)?|h(ours?)?|days?|px|fps|x|"
    r"tokens?|gb/s|mb/s|credits?)?$", re.I)


SLUG = re.compile(r"^[a-z]--|---", re.I)


def project_name(slug, cwd):
    """Human-readable project label. The raw slug must never reach the prompt."""
    if cwd:
        base = os.path.basename(cwd.rstrip("\\/"))
        if base and not base.endswith(":"):
            return base
    # "D--Instagram-Reels" -> "Instagram Reels": drop the drive prefix, then dashes
    bare = re.sub(r"^[a-zA-Z]--", "", slug)
    return re.sub(r"-+", " ", bare).strip() or slug


def looks_like_entity(name):
    """Reject quantities, statuses and filler that the prompt still lets slip through."""
    clean = " ".join(name.lower().split()).strip(" .:;\"'`")
    if not clean or clean in STOPWORDS:
        return False
    if SLUG.search(name):          # "D--Projects---AI-ML-LLMs-VAD" is a path slug, not a thing
        return False
    if QUANTITY.match(clean):                      # "464 MB", "40%", "20.7"
        return False
    if sum(c.isdigit() for c in clean) > len(clean) / 2:
        return False
    if len(clean) < 2 or len(name.split()) > 6:    # too short, or a sentence fragment
        return False
    return True


class Segmind:
    def __init__(self, key, qps=4.0):
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

    def chat(self, prompt, retries=3):
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode()
        for attempt in range(retries):
            self._throttle()
            req = urllib.request.Request(
                ENDPOINT, data=body, method="POST",
                headers={"Content-Type": "application/json", "x-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"{type(e).__name__}: {e}") from None
        raise RuntimeError("retries exhausted")


def parse_triples(raw):
    """Validate model output against the closed vocabulary. Junk is dropped, not stored."""
    text = raw.strip()
    if text.startswith("```"):  # models fence JSON even when told not to
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None  # caller records this as an error

    out = []
    for t in (data or {}).get("triples", []):
        if not isinstance(t, dict):
            continue
        s, p, o = (str(t.get(k, "")).strip() for k in ("s", "p", "o"))
        sk, ok = (str(t.get(k, "")).strip().lower() for k in ("sk", "ok"))
        p = p.strip().lower().replace(" ", "_").replace("-", "_")
        if not (s and o) or s.lower() == o.lower():
            continue
        if p not in PREDICATES or sk not in ENTITY_KINDS or ok not in ENTITY_KINDS:
            continue
        if len(s) > 80 or len(o) > 80:
            continue
        if not (looks_like_entity(s) and looks_like_entity(o)):
            continue
        if p == "decided" and sk != "person":   # decided is person -> decision
            continue
        out.append((s, sk, p, o, ok))
    return out


def entity_id(db, name, kind, cache):
    norm = " ".join(name.lower().split())
    key = (norm, kind)
    if key in cache:
        return cache[key]
    db.execute("INSERT OR IGNORE INTO entities (name, kind, norm) VALUES (?,?,?)",
               (name, kind, norm))
    eid = db.execute("SELECT id FROM entities WHERE norm=? AND kind=?", (norm, kind)).fetchone()[0]
    cache[key] = eid
    return eid


def run(db, api, limit=None, retry=False, workers=8):
    db.executescript(SCHEMA_SQL)
    where = ("WHERE c.id IN (SELECT chunk_id FROM extracted WHERE error IS NOT NULL)"
             if retry else
             "WHERE c.id NOT IN (SELECT chunk_id FROM extracted)")
    rows = db.execute(
        "SELECT c.id, c.text, c.project, c.session_id, c.ts, s.cwd FROM chunks c"
        " JOIN sessions s ON s.id = c.session_id " + where +
        f" ORDER BY {'RANDOM()' if limit else 'c.id'}"
        f" {'LIMIT ' + str(limit) if limit else ''}").fetchall()
    if not rows:
        print("nothing to extract")
        return
    print(f"extracting {len(rows)} chunks with {workers} workers...", flush=True)

    def work(row):
        cid, text, project, _, _, cwd = row
        prompt = PROMPT.format(preds=", ".join(sorted(PREDICATES)),
                               kinds=", ".join(sorted(ENTITY_KINDS)),
                               project=project_name(project, cwd), text=text[:12000])
        try:
            return row, parse_triples(api.chat(prompt)), None
        except RuntimeError as e:
            return row, None, str(e)

    cache, done, n_triples, n_err = {}, 0, 0, 0
    started = time.time()
    now = datetime.now(timezone.utc).isoformat()
    # as_completed, not map: one slow call must not head-of-line block every commit
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for fut in as_completed(futures):
            row, triples, err = fut.result()
            cid, _, project, session_id, ts, _ = row
            done += 1
            if triples is None:
                err = err or "unparseable JSON"
                n_err += 1
            else:
                for s, sk, p, o, ok in triples:
                    db.execute(
                        "INSERT OR IGNORE INTO triples"
                        " (subj, pred, obj, chunk_id, session_id, ts, project)"
                        " VALUES (?,?,?,?,?,?,?)",
                        (entity_id(db, s, sk, cache), p, entity_id(db, o, ok, cache),
                         cid, session_id, ts, project))
                n_triples += len(triples)
            db.execute(
                "INSERT OR REPLACE INTO extracted VALUES (?,?,?,?)",
                (cid, 0 if triples is None else len(triples), err, now))
            db.commit()  # per result: a kill must never discard paid-for work
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-9)
            print(f"  {done}/{len(rows)}  {n_triples} triples  {n_err} err"
                  f"  {elapsed/done:.1f}s/call  eta {(len(rows)-done)/max(rate,1e-9)/60:.1f}m",
                  flush=True)
    print(f"done: {n_triples} triples from {done} chunks ({n_err} errors) "
          f"in {(time.time()-started)/60:.1f}m", flush=True)


def stats(db):
    one = lambda sql: (db.execute(sql).fetchone() or [0])[0]
    print(f"entities : {one('SELECT COUNT(*) FROM entities')}")
    print(f"triples  : {one('SELECT COUNT(*) FROM triples')}")
    print(f"chunks   : {one('SELECT COUNT(*) FROM extracted')} extracted, "
          f"{one('SELECT COUNT(*) FROM extracted WHERE error IS NOT NULL')} errored")

    print("\nentity kinds:")
    for kind, n in db.execute("SELECT kind, COUNT(*) FROM entities GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {kind:12s} {n:5d}")
    print("\npredicates:")
    for pred, n in db.execute("SELECT pred, COUNT(*) FROM triples GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {pred:12s} {n:5d}")

    print("\nmost connected entities:")
    for name, kind, deg, projects in db.execute("""
            SELECT e.name, e.kind, COUNT(*) d, COUNT(DISTINCT t.project)
            FROM entities e JOIN triples t ON t.subj=e.id OR t.obj=e.id
            GROUP BY e.id ORDER BY d DESC LIMIT 15"""):
        span = f"  [{projects} projects]" if projects > 1 else ""
        print(f"  {deg:4d}  {kind:11s} {name}{span}")

    print("\nsample triples:")
    for s, p, o, proj in db.execute("""
            SELECT a.name, t.pred, b.name, t.project FROM triples t
            JOIN entities a ON a.id=t.subj JOIN entities b ON b.id=t.obj
            ORDER BY RANDOM() LIMIT 12"""):
        print(f"  ({s}) -[{p}]-> ({o})   ~{proj[:28]}")


def selfcheck():
    good = '```json\n{"triples":[{"s":"VAD","sk":"project","p":"uses","o":"PyTorch","ok":"technology"}]}\n```'
    assert parse_triples(good) == [("VAD", "project", "uses", "PyTorch", "technology")]

    junk = json.dumps({"triples": [
        {"s": "VAD", "sk": "project", "p": "loves", "o": "PyTorch", "ok": "technology"},   # bad predicate
        {"s": "VAD", "sk": "alien", "p": "uses", "o": "PyTorch", "ok": "technology"},      # bad kind
        {"s": "VAD", "sk": "project", "p": "uses", "o": "vad", "ok": "concept"},           # self-loop
        {"s": "", "sk": "project", "p": "uses", "o": "PyTorch", "ok": "technology"},       # empty
        {"s": "x" * 99, "sk": "project", "p": "uses", "o": "PyTorch", "ok": "technology"}, # too long
        {"s": "VAD", "sk": "project", "p": "uses", "o": "464 MB", "ok": "concept"},        # quantity
        {"s": "npm-cache", "sk": "artifact", "p": "uses", "o": "SAFE TO DELETE", "ok": "concept"},
        {"s": "VAD", "sk": "project", "p": "Depends On", "o": "VisDrone", "ok": "artifact"},
    ]})
    assert parse_triples(junk) == [("VAD", "project", "depends_on", "VisDrone", "artifact")],         parse_triples(junk)
    assert parse_triples("not json at all") is None
    for bad in ("464 MB", "20.7", "40%", "SAFE TO DELETE", "Completed so far",
                "assistant", "redundant", "", "a", "1.5 GB",
                "some long phrase that is really a whole sentence fragment"):
        assert not looks_like_entity(bad), bad
    for bad in ("D--Projects---AI-ML-LLMs-VAD", "d--Instagram-Reels", "C--"):
        assert not looks_like_entity(bad), bad
    assert project_name("D--Projects---AI-ML-LLMs-VAD", r"D:\Projects - AI ML LLMs\VAD") == "VAD"
    assert project_name("D--Instagram-Reels", None) == "Instagram Reels"
    assert project_name("C--", "C:\\") == "C--"   # drive root has no basename; slug stands
    for good in ("PyTorch", "VisDrone", "majority voting", "capacity_probe.py",
                 "two-stage cascade", "384px downscale", "Aditya", "gpt-5-nano"):
        assert looks_like_entity(good), good
    # "decided" must have a person subject
    d = lambda sk: json.dumps({"triples": [{"s": "Aditya", "sk": sk, "p": "decided",
                                            "o": "two-stage cascade", "ok": "decision"}]})
    assert parse_triples(d("project")) == []
    assert len(parse_triples(d("person"))) == 1
    assert parse_triples('{"triples":[]}') == []

    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    cache = {}
    a = entity_id(db, "PyTorch", "technology", cache)
    b = entity_id(db, " pytorch ", "technology", cache)   # normalized dedup
    c = entity_id(db, "PyTorch", "concept", cache)        # different kind -> different node
    assert a == b and a != c, (a, b, c)
    assert entity_id(db, "PyTorch", "technology", {}) == a, "dedup must survive a cold cache"
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit()

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA_SQL)
    if not args.stats:
        key = os.environ.get("SEGMIND_API_KEY")
        if not key:
            env = os.path.join(HERE, ".env")
            if os.path.exists(env):
                for line in open(env, encoding="utf-8"):
                    if line.startswith("SEGMIND_API_KEY="):
                        key = line.split("=", 1)[1].strip()
        if not key:
            sys.exit("no SEGMIND_API_KEY in environment or .env")
        run(con, Segmind(key), limit=args.limit, retry=args.retry, workers=args.workers)
        print()
    stats(con)
