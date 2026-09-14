"""Generate a MULTI-HOP eval set, where the answer is deliberately not in the
chunk the question was written from.

Usage:
    python make_multihop.py --n 20
    python make_multihop.py --show

Why this exists: evalset.json asks a question written from chunk C and marks C
as gold. Embedding similarity to C then answers it directly, so that set measures
single-hop recall and structurally favours the vector baseline. It cannot tell us
whether the graph is worth anything.

Construction here:

    entity E appears in chunk C1 (session S1) and chunk C2 (session S2), S1 != S2

    the question is written from C1 only, and asks where ELSE E came up
    gold is C2

C2's wording never reaches the question generator, so lexical and embedding
similarity to the question cannot surface C2 except by luck. The link C1 -> E ->
C2 is exactly one hop through the graph. This is the test the project's claim
actually rests on.

Honest caveat: this construction is symmetrically UNfavourable to vector search
in the same way evalset.json is unfavourable to the graph. Report both, not one.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import embed
import graph
from extract import Segmind

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "evalset_multihop.json")

PROMPT = """Below is one exchange from a developer's history, and one thing mentioned
in it: "{entity}".

Write ONE question the developer might later ask their memory system, asking where
ELSE this thing came up in their work -- in a DIFFERENT project or session than
this one.

Rules:
- The question must name "{entity}" (or an obvious natural phrasing of it).
- It must ask about other/earlier/elsewhere usage, not about this exchange.
  Good shapes: "where else did I use X?", "what other project did I use X in?",
  "have I run into X before this?"
- Do NOT describe or quote anything specific from the exchange below. The
  exchange is only context for what X is.
- One sentence. No preamble.

Return ONLY JSON: {{"question": "..."}}

EXCHANGE (project: {project}):
{text}"""


def pairs(db, n):
    """Entities bridging two sessions, with the two chunks that mention them."""
    rows = db.execute("""
        SELECT e.id, e.name, e.kind, COUNT(DISTINCT t.session_id) ns
        FROM entities e JOIN triples t ON (t.subj = e.id OR t.obj = e.id)
        WHERE e.kind IN ('technology','concept','artifact','problem','decision')
        GROUP BY e.id HAVING ns >= 2 AND COUNT(DISTINCT t.chunk_id) BETWEEN 2 AND 14
        ORDER BY RANDOM() LIMIT ?""", (n * 3,)).fetchall()

    out = []
    for ent in rows:
        chunks = db.execute("""
            SELECT DISTINCT c.id, c.text, c.project, c.session_id, s.title
            FROM triples t JOIN chunks c ON c.id = t.chunk_id
            JOIN sessions s ON s.id = c.session_id
            WHERE (t.subj = ? OR t.obj = ?) AND c.n_chars > 600
            GROUP BY c.session_id ORDER BY RANDOM()""",
            (ent["id"], ent["id"])).fetchall()
        if len({c["session_id"] for c in chunks}) < 2:
            continue
        src, gold = chunks[0], chunks[1]
        if src["session_id"] == gold["session_id"]:
            continue
        out.append((ent, src, gold))
    return out


def build(db, api, n, workers=6):
    cands = pairs(db, n)
    print(f"{len(cands)} bridging entities found; drafting questions...", flush=True)

    def work(item):
        ent, src, gold = item
        prompt = PROMPT.format(entity=ent["name"], project=src["project"],
                               text=src["text"][:6000])
        try:
            raw = api.chat(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            return item, (json.loads(raw) or {}).get("question", "").strip()
        except Exception as e:                       # noqa: BLE001
            return item, f"ERROR {e}"

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, c) for c in cands]):
            (ent, src, gold), q = fut.result()
            if not q or q.startswith("ERROR") or len(q) < 15:
                continue
            out.append({
                "question": q,
                "bridge_entity": ent["name"],
                "entity_kind": ent["kind"],
                "source_chunk_id": src["id"],          # where the question came from
                "source_project": src["project"],
                "gold_chunk_id": gold["id"],           # the answer, in another session
                "project": gold["project"],
                "session_id": gold["session_id"],
                "session_title": gold["title"],
                "ts": None,
            })
            print(f"  [{len(out):2d}] ({ent['name']}) {q[:80]}", flush=True)
            if len(out) >= n:
                break

    cross = sum(1 for q in out if q["source_project"] != q["project"])
    print(f"\n{len(out)} questions | {cross} cross-project, "
          f"{len(out) - cross} cross-session within a project")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        for i, q in enumerate(json.load(open(args.out, encoding="utf-8")), 1):
            print(f"{i:2d}. [{q['bridge_entity']}] {q['question']}")
            print(f"    from chunk {q['source_chunk_id']} ({q['source_project'][:20]})"
                  f"  ->  gold chunk {q['gold_chunk_id']} ({q['project'][:20]})")
        sys.exit()

    key = embed.api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")
    con = graph.connect(args.db)
    questions = build(con, Segmind(key), args.n)
    json.dump(questions, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.out}")
