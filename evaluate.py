"""Benchmark hybrid retrieval against vector-only and graph-only baselines.

Usage:
    python evaluate.py                     # all modes, top-10
    python evaluate.py --ablate            # also run graph with lexical seeds
    python evaluate.py --sweep-k           # RRF constant sweep

Metrics, all computed against gold chunks fixed before any retriever ran:
    hit@k   fraction of questions whose gold chunk appears in the top k
    MRR     mean reciprocal rank of the gold chunk (0 if absent)
    p50/p95 retrieval latency, embedding API time separated out

The embedding API call dominates wall-clock and varies 0.6-15s run to run, so
latency is reported twice: end-to-end, and with API time subtracted. Only the
second number says anything about the retrieval code.
"""
import argparse
import json
import os
import statistics
import sys
import time

import embed
import embed_entities
import graph
import retrieve

HERE = os.path.dirname(os.path.abspath(__file__))


class TimedEmbedder:
    """Wraps the embedder to separate API wait from retrieval compute."""

    def __init__(self, inner):
        self.inner = inner
        self.elapsed = 0.0
        self.calls = 0

    def __call__(self, text):
        start = time.time()
        try:
            return self.inner(text)
        finally:
            self.elapsed += time.time() - start
            self.calls += 1


def score(results, gold_id, ks=(1, 3, 5, 10)):
    ids = [r["chunk_id"] for r in results]
    rank = ids.index(gold_id) + 1 if gold_id in ids else None
    return {f"hit@{k}": float(rank is not None and rank <= k) for k in ks} | {
        "rr": 1.0 / rank if rank else 0.0, "rank": rank}


def run_mode(db, embedder, questions, mode, top_k, hops, index, ent_index,
             semantic_seeds=True):
    rows, latencies, api_times = [], [], []
    empty = 0
    for q in questions:
        timed = TimedEmbedder(embedder)
        started = time.time()
        out = retrieve.retrieve(db, timed, q["question"], mode=mode, top_k=top_k,
                                hops=hops, index=index, ent_index=ent_index,
                                semantic_seeds=semantic_seeds)
        wall = (time.time() - started) * 1000
        latencies.append(wall)
        api_times.append(timed.elapsed * 1000)
        if not out["results"]:
            empty += 1
        rows.append(score(out["results"], q["gold_chunk_id"]))

    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n for k in rows[0] if k.startswith("hit@")}
    agg["MRR"] = sum(r["rr"] for r in rows) / n
    agg["empty"] = empty
    agg["p50_ms"] = statistics.median(latencies)
    agg["p95_ms"] = sorted(latencies)[int(0.95 * (n - 1))]
    agg["p50_api_ms"] = statistics.median(api_times)
    agg["p50_compute_ms"] = statistics.median(
        [l - a for l, a in zip(latencies, api_times)])
    return agg, rows


def table(name_to_agg, ks=(1, 3, 5, 10)):
    head = f"{'mode':<22}" + "".join(f"{'hit@'+str(k):>8}" for k in ks) + \
           f"{'MRR':>8}{'empty':>7}{'p50ms':>9}{'compute':>9}"
    print(head)
    print("-" * len(head))
    for name, a in name_to_agg.items():
        row = f"{name:<22}" + "".join(f"{a['hit@'+str(k)]:>8.2f}" for k in ks)
        print(row + f"{a['MRR']:>8.3f}{a['empty']:>7}"
                    f"{a['p50_ms']:>9.0f}{a['p50_compute_ms']:>9.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--evalset", default=os.path.join(HERE, "evalset.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--sweep-k", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results.json"))
    args = ap.parse_args()

    if not os.path.exists(args.evalset):
        sys.exit(f"no evalset at {args.evalset} -- run make_evalset.py first")
    questions = json.load(open(args.evalset, encoding="utf-8"))
    key = embed.api_key()
    if not key:
        sys.exit("no SEGMIND_API_KEY in environment or .env")

    db = graph.connect(args.db)
    embedder = embed.Embedder(key)
    index = embed.load_index(db)
    ent_index = embed_entities.load(db)
    print(f"{len(questions)} questions | {len(index)} chunk vectors | "
          f"{len(ent_index)} entity vectors\n")

    results = {}
    for mode in ("vector", "graph", "hybrid", "auto"):
        agg, _ = run_mode(db, embedder, questions, mode, args.top_k, args.hops,
                          index, ent_index)
        results[mode] = agg
    if args.ablate:
        for mode in ("graph", "hybrid"):
            agg, _ = run_mode(db, embedder, questions, mode, args.top_k, args.hops,
                              index, ent_index, semantic_seeds=False)
            results[f"{mode} (lexical seeds)"] = agg

    table(results)

    if args.sweep_k:
        print("\nRRF constant sweep (hybrid):")
        original = retrieve.RRF_K
        sweep = {}
        for k in (5, 10, 20, 60, 120):
            retrieve.RRF_K = k
            agg, _ = run_mode(db, embedder, questions, "hybrid", args.top_k,
                              args.hops, index, ent_index)
            sweep[f"hybrid k={k}"] = agg
        retrieve.RRF_K = original
        table(sweep)
        results |= sweep

    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")
