"""Hybrid retrieval: vector similarity + multi-hop graph traversal, fused with RRF.

Usage:
    python retrieve.py "why did the VAD eval stall?"
    python retrieve.py --mode vector "..."    # baseline for the eval
    python retrieve.py --mode graph  "..."
    python retrieve.py --explain "..."        # show why each chunk surfaced

Modes: `vector` (pure embeddings, the baseline), `graph` (pure traversal),
`hybrid` (fixed-weight RRF), `auto` (adaptive -- vector confidence sets the graph
weight). Fixed-weight `hybrid` loses to both its inputs; `auto` exists because of
that result, not in spite of it.
"""
import argparse
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict

import embed
import graph

HERE = os.path.dirname(os.path.abspath(__file__))
RRF_K = 60   # standard reciprocal-rank-fusion constant; damps the top-rank advantage
GRAPH_WEIGHT = 1.0   # relative weight of the graph ranking in fixed-weight hybrid

# Adaptive fusion ("auto" mode). When the best chunk vector is far from the query,
# no chunk *says* the answer and the answer is more likely to be reachable than
# stateable -- so lean on the graph. Both constants were fit on the 45 eval
# questions in this repo and are reported as tuned, not as an independent result.
AUTO_HI = 0.55   # cosine at or above this: trust the embedding, no graph weight
AUTO_LO = 0.40   # at or below this: weight the graph at AUTO_MAX_W
AUTO_MAX_W = 3.0


def auto_weight(top_cos):
    """Graph weight from vector self-confidence. Linear between LO and HI."""
    if top_cos >= AUTO_HI:
        return 0.0
    if top_cos <= AUTO_LO:
        return AUTO_MAX_W
    span = (AUTO_HI - top_cos) / (AUTO_HI - AUTO_LO)
    return AUTO_MAX_W * span


def vector_candidates(db, embedder, query, top_k, index=None):
    """Ranked chunk ids by embedding cosine."""
    hits = embed.search(db, embedder, query, top_k=top_k, index=index)
    return [h["chunk_id"] for h in hits], {h["chunk_id"]: h["score"] for h in hits}


def graph_candidates(db, query, hops=2, top_k=30, embedder=None, ent_index=None):
    """Ranked chunk ids by multi-hop traversal from entities named in the query.

    A chunk scores on how many reached entities it mentions and how close they
    were to a seed -- a 1-hop entity is worth more than a 3-hop one.

    With an embedder, seeds are resolved semantically; without one it falls back
    to lexical matching, which is the Phase-7 ablation.
    """
    if embedder is not None:
        seeds = graph.resolve_semantic(db, query, embedder, index=ent_index)
    else:
        seeds = graph.resolve(db, query)
    if not seeds:
        return [], {}, [], []
    reached = graph.neighbors(db, [s["id"] for s in seeds], hops=hops)
    if not reached:
        return [], {}, seeds, []

    # Hop distance alone treats "VAD" (in every chunk of the project) as worth the
    # same as "OVERLAP_FOR_POSITIVE" (in two). Weight by inverse chunk frequency so
    # a rare entity actually discriminates.
    n_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    reached_ids = ",".join(str(int(r["id"])) for r in reached)
    freq = dict(db.execute(
        f"SELECT e.id, COUNT(DISTINCT t.chunk_id) FROM entities e"
        f" JOIN triples t ON (t.subj = e.id OR t.obj = e.id)"
        f" WHERE e.id IN ({reached_ids}) GROUP BY e.id"))
    weight = {
        r["id"]: (1.0 / (1 + r["hops"]))
                 * math.log(1 + n_chunks / (1 + freq.get(r["id"], 0)))
        for r in reached}
    ids = ",".join(str(int(i)) for i in weight)

    # Count DISTINCT matched entities per chunk, not triples. Summing per triple
    # just rewards whichever chunk produced the most triples -- long rambling
    # exchanges outrank short precise ones regardless of relevance.
    matched = defaultdict(dict)
    for row in db.execute(f"SELECT chunk_id, subj, obj FROM triples"
                          f" WHERE subj IN ({ids}) OR obj IN ({ids})"):
        for end in (row["subj"], row["obj"]):
            if end in weight:
                matched[row["chunk_id"]][end] = weight[end]

    # and damp by how much the chunk says overall, so verbosity is not relevance
    totals = dict(db.execute("SELECT chunk_id, COUNT(*) FROM triples GROUP BY chunk_id"))
    scores = {cid: sum(hits.values()) / math.sqrt(1 + totals.get(cid, 0))
              for cid, hits in matched.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [c for c, _ in ranked], dict(ranked), seeds, reached


def rrf(*rankings, k=RRF_K, weights=None):
    """Reciprocal rank fusion. Position in each list is all that matters, so the
    vector cosine and the graph score never have to be put on a common scale.

    `weights` exists because unweighted RRF measurably loses to the better of its
    two inputs on both eval sets: mixing a weak ranking in at equal strength drags
    the strong one down.
    """
    weights = weights or [1.0] * len(rankings)
    fused = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            fused[item] += weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def hydrate(db, chunk_ids):
    if not chunk_ids:
        return {}
    ids = ",".join(str(int(c)) for c in chunk_ids)
    return {r["id"]: dict(r) for r in db.execute(
        f"SELECT c.id, c.text, c.project, c.session_id, c.ts, c.n_chars, s.title"
        f" FROM chunks c JOIN sessions s ON s.id = c.session_id WHERE c.id IN ({ids})")}


def retrieve(db, embedder, query, mode="hybrid", top_k=8, hops=2, index=None,
             ent_index=None, semantic_seeds=True):
    started = time.time()
    vec_ids, vec_scores = [], {}
    gr_ids, gr_scores, seeds, reached = [], {}, [], []

    weight = GRAPH_WEIGHT
    if mode in ("vector", "hybrid", "auto"):
        vec_ids, vec_scores = vector_candidates(db, embedder, query, top_k * 3, index)
    if mode == "auto":
        # vector runs first: its own top score decides how much graph to mix in
        weight = auto_weight(max(vec_scores.values(), default=0.0))
    if mode in ("graph", "hybrid") or (mode == "auto" and weight > 0):
        gr_ids, gr_scores, seeds, reached = graph_candidates(
            db, query, hops, top_k * 3,
            embedder=embedder if semantic_seeds else None, ent_index=ent_index)

    if mode == "vector":
        fused = [(c, 1.0 / (RRF_K + i + 1)) for i, c in enumerate(vec_ids)]
    elif mode == "graph":
        fused = [(c, 1.0 / (RRF_K + i + 1)) for i, c in enumerate(gr_ids)]
    else:
        fused = rrf(vec_ids, gr_ids, weights=[1.0, weight])

    fused = fused[:top_k]
    meta = hydrate(db, [c for c, _ in fused])
    results = []
    for chunk_id, score in fused:
        row = meta.get(chunk_id)
        if not row:
            continue
        results.append({
            "chunk_id": chunk_id, "score": round(score, 6),
            "text": row["text"], "project": row["project"],
            "session_id": row["session_id"], "ts": row["ts"], "title": row["title"],
            "why": {"vector_rank": vec_ids.index(chunk_id) + 1 if chunk_id in vec_ids else None,
                    "graph_rank": gr_ids.index(chunk_id) + 1 if chunk_id in gr_ids else None,
                    "vector_score": round(vec_scores.get(chunk_id, 0.0), 4) or None,
                    "graph_score": round(gr_scores.get(chunk_id, 0.0), 4) or None},
        })
    return {
        "query": query, "mode": mode, "results": results,
        "seeds": [s["name"] for s in seeds],
        "reached": [{"name": r["name"], "kind": r["kind"], "hops": r["hops"]}
                    for r in reached[:40]],
        "graph_weight": round(weight, 3),
        "latency_ms": round((time.time() - started) * 1000, 1),
    }


def render(out, explain=False):
    print(f"[{out['mode']}] {out['latency_ms']} ms  {len(out['results'])} results")
    if out["seeds"]:
        print(f"seeds: {', '.join(out['seeds'])}")
        if explain and out["reached"]:
            near = [f"{r['name']}({r['hops']})" for r in out["reached"][:12]]
            print(f"reached: {', '.join(near)}")
    print()
    for i, r in enumerate(out["results"], 1):
        head = " ".join(r["text"].split())[:220]
        tag = f"[{r['project'][:24]}]"
        print(f"{i}. {r['score']:.5f} {tag} {r['title'] or ''}  {r['ts'][:10] if r['ts'] else ''}")
        if explain:
            w = r["why"]
            print(f"   vector_rank={w['vector_rank']} graph_rank={w['graph_rank']} "
                  f"cos={w['vector_score']} graph={w['graph_score']}")
        print(f"   {head}\n")


def selfcheck():
    # RRF: an item ranked well in both lists must beat one ranked well in only one
    both = rrf(["a", "b", "c"], ["a", "c", "b"])
    assert both[0][0] == "a", both
    only_one = rrf(["x", "a"], ["a", "y"])
    assert only_one[0][0] == "a", only_one
    assert abs(rrf(["a"])[0][1] - 1.0 / (RRF_K + 1)) < 1e-12
    # fusion is order-insensitive across lists
    assert dict(rrf(["a", "b"], ["b", "a"])) == dict(rrf(["b", "a"], ["a", "b"]))
    # an item in neither list cannot appear
    assert "zzz" not in dict(rrf(["a"], ["b"]))
    # weighting must be able to flip which list wins
    heavy = rrf(["v1", "v2"], ["g1", "g2"], weights=[1.0, 5.0])
    assert heavy[0][0] == "g1", heavy
    light = rrf(["v1", "v2"], ["g1", "g2"], weights=[1.0, 0.0])
    assert light[0][0] == "v1" and dict(light)["g1"] == 0.0, light
    assert dict(rrf(["a"], ["a"], weights=[1.0, 1.0]))["a"] ==            2 * dict(rrf(["a"], weights=[1.0]))["a"], "agreement accumulates"

    # adaptive weighting: confident vector -> no graph; unconfident -> full graph
    assert auto_weight(0.90) == 0.0, "a near-exact chunk needs no traversal"
    assert auto_weight(AUTO_HI) == 0.0
    assert auto_weight(AUTO_LO) == AUTO_MAX_W
    assert auto_weight(0.10) == AUTO_MAX_W, "below the floor stays clamped"
    mid = auto_weight((AUTO_HI + AUTO_LO) / 2)
    assert abs(mid - AUTO_MAX_W / 2) < 1e-9, mid
    lo, hi = auto_weight(0.44), auto_weight(0.52)
    assert lo > hi, "weight must fall monotonically as confidence rises"
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--mode", choices=("auto", "hybrid", "vector", "graph"),
                    default="auto")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--lexical-seeds", action="store_true",
                    help="disable semantic seed resolution (Phase-7 ablation)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit()
    if not args.query:
        sys.exit("give me a query")

    con = graph.connect(args.db)
    key = embed.api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")
    render(retrieve(con, embed.Embedder(key), args.query,
                    mode=args.mode, top_k=args.top_k, hops=args.hops,
                    semantic_seeds=not args.lexical_seeds),
           explain=args.explain)
