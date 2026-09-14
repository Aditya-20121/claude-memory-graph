"""Generate an evaluation set from real chunks, with ground truth by construction.

Usage:
    python make_evalset.py --n 25          # write evalset.json
    python make_evalset.py --show          # print it for review

Method, stated plainly because it bounds what the numbers mean:

  A chunk is chosen, then gpt-5-nano writes a question answerable from THAT
  chunk. The chunk is the gold answer by construction -- nobody hand-labels and
  nothing is guessed after the fact.

  The generator is shown only the chunk text. It never sees the graph, the
  embeddings, or any retriever. So the gold set cannot favour graph retrieval
  over vector retrieval; whatever difference the eval reports is a property of
  the retrievers, not of how the questions were made.

  The prompt forbids quoting rare literal strings, because a question that
  copies a distinctive filename reduces the task to string matching and both
  retrievers would ace it for the wrong reason.

Known limitation: a single gold chunk per question understates recall when the
same fact appears in several chunks. That biases every mode equally, so mode
comparison stays valid even though absolute recall is pessimistic.
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
OUT = os.path.join(HERE, "evalset.json")

PROMPT = """Below is one exchange between a developer and an AI assistant, from the
developer's own history.

Write ONE question the developer might later ask their memory system, that this
exchange answers.

Rules:
- It must be answerable from this exchange. Do not ask about anything not stated here.
- Write it the way a person recalls something months later, vaguely: "why did I
  switch away from X", "what was blocking the Y run", "what did I pick for Z".
- Do NOT quote rare literal strings from the text -- no exact filenames, hashes,
  variable names, or error strings. Use the ordinary words a person would use.
- One sentence. No preamble.
- If this exchange states no durable fact worth recalling (pure chit-chat, a bare
  status ping, a tool dump), return exactly: SKIP

Return ONLY JSON: {{"question": "..."}} or {{"question": "SKIP"}}

EXCHANGE (project: {project}):
{text}"""


def pick_chunks(db, n):
    """Stratify across sessions, preferring chunks with real content."""
    return db.execute("""
        SELECT c.id, c.text, c.project, c.session_id, c.ts, s.title
        FROM chunks c JOIN sessions s ON s.id = c.session_id
        WHERE c.n_chars BETWEEN 800 AND 6000
          AND EXISTS (SELECT 1 FROM triples t WHERE t.chunk_id = c.id)
        GROUP BY c.id
        ORDER BY RANDOM() LIMIT ?""", (n * 3,)).fetchall()


def build(db, api, n, workers=6):
    rows = pick_chunks(db, n)
    print(f"drafting questions from {len(rows)} candidate chunks...", flush=True)

    def work(row):
        prompt = PROMPT.format(project=row["project"], text=row["text"][:8000])
        try:
            raw = api.chat(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            q = (json.loads(raw) or {}).get("question", "").strip()
            return row, q
        except Exception as e:                      # noqa: BLE001 - log and skip
            return row, f"ERROR {e}"

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in rows]):
            row, q = fut.result()
            if not q or q == "SKIP" or q.startswith("ERROR") or len(q) < 15:
                continue
            out.append({
                "question": q,
                "gold_chunk_id": row["id"],
                "project": row["project"],
                "session_id": row["session_id"],
                "session_title": row["title"],
                "ts": row["ts"],
            })
            print(f"  [{len(out):2d}] {q[:96]}", flush=True)
            if len(out) >= n:
                break

    # how many sessions the gold chunks span -- a set drawn from one session
    # would not test cross-session recall at all
    sessions = {q["session_id"] for q in out}
    projects = {q["project"] for q in out}
    print(f"\n{len(out)} questions over {len(sessions)} sessions, {len(projects)} projects")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        data = json.load(open(args.out, encoding="utf-8"))
        for i, q in enumerate(data, 1):
            print(f"{i:2d}. [{q['project'][:22]}] {q['question']}")
            print(f"    gold=chunk {q['gold_chunk_id']}  {q['session_title'] or ''}")
        sys.exit()

    key = embed.api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")
    con = graph.connect(args.db)
    questions = build(con, Segmind(key), args.n)
    json.dump(questions, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.out}")
