"""Run every mode over both eval sets and print one combined table."""
import json
import embed, embed_entities, graph, retrieve, evaluate

db = graph.connect("memory.db")
E = embed.Embedder(embed.api_key())
idx, eidx = embed.load_index(db), embed_entities.load(db)
sets = {"single-hop": json.load(open("evalset.json", encoding="utf-8")),
        "multi-hop": json.load(open("evalset_multihop.json", encoding="utf-8"))}
n_total = sum(len(v) for v in sets.values())
combined, out = {}, {}

for mode, sem in (("vector", True), ("graph", False), ("hybrid", False), ("auto", False)):
    row = {}
    for name, qs in sets.items():
        agg, _ = evaluate.run_mode(db, E, qs, mode, 10, 2, idx, eidx, semantic_seeds=sem)
        row[name] = agg
        print(f"  {mode:7s} {name:11s} MRR={agg['MRR']:.3f} hit@10={agg['hit@10']:.2f}",
              flush=True)
    # weight each set by its question count so the mean is over all 45 questions
    combined[mode] = sum(row[n]["MRR"] * len(sets[n]) for n in sets) / n_total
    out[mode] = row

print(f"\n{'mode':<10}{'single MRR':>12}{'multi MRR':>11}{'overall MRR':>13}"
      f"{'single h@10':>13}{'multi h@10':>12}")
print("-" * 71)
for mode, row in out.items():
    print(f"{mode:<10}{row['single-hop']['MRR']:>12.3f}{row['multi-hop']['MRR']:>11.3f}"
          f"{combined[mode]:>13.3f}"
          f"{row['single-hop']['hit@10']:>13.2f}{row['multi-hop']['hit@10']:>12.2f}")
json.dump({"per_set": out, "overall_mrr": combined},
          open("eval_final.json", "w", encoding="utf-8"), indent=1)
