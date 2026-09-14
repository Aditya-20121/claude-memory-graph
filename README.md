# claude-memory-graph

Persistent, queryable memory over my own Claude Code history: a knowledge graph
plus a vector index over past conversations, exposed as an MCP server so Claude
can recall my own history mid-session.

Everything runs locally except two API calls (entity extraction and embeddings).
No framework, no ORM, no vector database — Python 3.14 standard library, one
SQLite file, 27 MB.

**[Interactive 3D graph →](https://aditya-20121.github.io/claude-memory-graph/)**
(entity names limited to public tools; private ones masked)

**Headline result, in two parts.**

The hybrid retrieval this project set out to build — fixed-weight RRF over vector
and graph — **loses to both of its own inputs**. Vector and graph turn out to be
strongly complementary with neither dominating, and blending them at a fixed
ratio is worse than picking one.

Replacing fusion with **routing** fixes it. Letting vector's own confidence decide
how much graph to mix in gets overall MRR **0.438 vs 0.353** for the best fixed
mode (+24%), and matches graph's cross-session recall (hit@10 0.85) while keeping
almost all of vector's single-chunk recall (0.84 vs 0.88).

The negative result came first and is what produced the working one, so both stay
at the top rather than only the flattering half. Numbers in
[Evaluation](#evaluation).

---

## What it does

```
~/.claude/projects/**/*.jsonl          25 sessions, 2026-05-26 → 2026-09-11
Claude Desktop session store           session titles
<project>/CLAUDE.md                    curated project context
          │
          ├─ ingest.py        normalize + chunk        →  411 chunks, 1.1M chars
          │                                               3,075 structural refs
          ├─ extract.py       LLM triple extraction    →  2,126 triples
          ├─ graph.py         + structural edges       →  2,579 entities
          │                     prune, renormalize        2,895 triples total
          ├─ embed.py         chunk vectors            →  411 × 1536-d
          ├─ embed_entities.py entity-name vectors     →  2,579 × 1536-d
          │
          ├─ retrieve.py      auto | vector | graph | hybrid
          └─ server.py        MCP: search_memory, get_related_entities, add_memory
```

## Quick start

```bash
cp .env.example .env          # add SEGMIND_API_KEY
python ingest.py              # transcripts → memory.db
python extract.py             # triples (≈32 min, ≈0.14 Segmind credits)
python graph.py --build       # structural edges + prune + renormalize
python embed.py               # chunk vectors (≈7 min)
python embed_entities.py      # entity vectors (≈25 min)
python retrieve.py "what blocked the model eval?"           # auto mode
python evaluate.py --ablate                              # per-set tables
python benchmark_all.py                                  # the combined table above
python visualize.py                                      # 3D graph, full names -> graph.html
python visualize.py --publish                            # masked names -> docs/index.html
```

Every module has a runnable self-check with no framework or fixtures:

```bash
for m in ingest extract graph embed retrieve server; do python $m.py --selfcheck; done
# ingest extract graph embed retrieve server -> 6/6 ok
```

## Data sources

Claude Desktop is installed as an **MSIX/Store package**, so its AppData is
virtualized — `%APPDATA%\Claude` does not exist and the real store lives at:

```
%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\
```

What is actually there, after checking rather than assuming:

| source | status |
|---|---|
| `~/.claude/projects/**/*.jsonl` | full transcripts — the real corpus |
| `claude-code-sessions/**/local_*.json` | 17 session records with auto-generated **titles**; 16/17 `cliSessionId`s already had local transcripts |
| `IndexedDB/https_claude.ai_0` | **composer draft cache, not history** — 3 MB of progressive keystroke snapshots, no assistant replies |
| `Local Storage/leveldb` | telemetry queue only |
| `claude-code/2.1.*` | 419 MB of bundled CLI binaries, no transcripts |

claude.ai chat history is server-side only. The desktop store contributes
titles, not conversations.

## Chunking

One chunk = one **exchange** (a human prompt plus the replies it drew), split at
6,000 chars on paragraph boundaries. The exchange is already the coherent unit —
one intent, one resolution — so no topic-shift heuristic was needed.

98% of the bytes are tool-call payloads. They are dropped as text but mined for
3,075 structural refs (`file`, `command`, `mcp`, `url`, `search`), which become
graph edges that **cannot be hallucinated** because no model produced them.

## Graph schema

Closed vocabulary, enforced at validation — anything outside it is dropped
rather than stored. Freeform extraction invents a new predicate per sentence and
the graph stops being traversable.

**Entity kinds** — artifact, concept, technology, project, problem, decision, person
**Predicates** — uses, implements, evaluates, causes, fixes, replaces, depends_on,
compared_to, part_of, blocked_by, decided

`triples.src` separates `llm` (2,126) from `structural` (769) so the eval can
ablate them.

## Evaluation

Two eval sets, because **one of them alone would have been misleading**.

| | construction | favours |
|---|---|---|
| `evalset.json` (25 q) | question written from chunk C, gold = C | vector |
| `evalset_multihop.json` (20 q) | question written from C1 about entity E, gold = C2 in a *different session* | graph |

Neither is neutral. Both are reported.

### Single-hop recall — 25 questions

```
mode                     hit@1   hit@3   hit@5  hit@10     MRR    p50ms  compute
vector                    0.44    0.56    0.88    0.88   0.576      939    139.8
graph                     0.24    0.36    0.40    0.56   0.323     2033   1080.9
hybrid                    0.40    0.48    0.60    0.76   0.482     3228   1225.1
graph (lexical seeds)     0.16    0.20    0.20    0.28   0.191      307    307.1
hybrid (lexical seeds)    0.24    0.28    0.60    0.76   0.349     1405    476.2
```

### Cross-session recall — 20 questions

```
mode                     hit@1   hit@3   hit@5  hit@10     MRR    p50ms  compute
vector                    0.05    0.10    0.10    0.15   0.074     1452    149.8
graph                     0.20    0.45    0.55    0.75   0.344     2121   1090.5
hybrid                    0.05    0.35    0.35    0.60   0.216     3227   1177.0
graph (lexical seeds)     0.25    0.50    0.65    0.85   0.420      307    307.3
hybrid (lexical seeds)    0.05    0.40    0.40    0.65   0.234     1170    421.1
```

`compute` is latency with embedding-API time subtracted. The Segmind call ranged
0.6–15 s for identical requests and dominates wall clock, so end-to-end latency
mostly measures Segmind's load, not this code.

### Routing, and why it exists

Fixed fusion failing is what motivated `auto`. All four modes, both sets, lexical
seeds throughout, 45 questions total:

```text
mode        single MRR  multi MRR  overall MRR  single h@10  multi h@10
vector           0.576      0.074        0.353         0.88        0.15
graph            0.191      0.420        0.293         0.28        0.85
hybrid           0.349      0.234        0.298         0.76        0.65
auto             0.538      0.313        0.438         0.84        0.85
```

`auto` runs vector first and lets its top cosine set the graph weight:

```text
top cosine >= 0.55  ->  graph weight 0.0   (skip traversal entirely)
top cosine <= 0.40  ->  graph weight 3.0
between             ->  linear
```

The reasoning: when the nearest chunk vector is far from the query, no chunk
*states* the answer, so the answer is likelier reachable than stateable.
Continuous rather than a hard switch, so borderline queries degrade gracefully;
and at high confidence it skips traversal, making confident queries *faster* than
fixed hybrid rather than slower.

### What the numbers say

1. **Fixed hybrid loses on both sets** — 0.349 vs vector's 0.576 single-hop,
   0.234 vs graph's 0.420 multi-hop. Unweighted RRF mixes a weak ranking in at
   equal strength and drags the strong one down.
2. **Routing recovers it.** Overall MRR 0.438 vs 0.353 for the best fixed mode.
   Recall is the clearer story: `auto` matches graph's multi-hop hit@10 exactly
   (0.85) and keeps nearly all of vector's single-hop hit@10 (0.84 vs 0.88).
   It does *not* beat each specialist's MRR on that specialist's home set
   (0.538 vs 0.576; 0.313 vs 0.420) — ranking is slightly diluted, coverage is
   near-best on both.
3. **Graph beats vector 5× on cross-session recall** (hit@10 0.85 vs 0.15) and
   loses badly on single-chunk recall. Complementarity, not superiority.
4. **Semantic vs lexical seeding flips between sets.** Semantic wins single-hop
   (0.323 vs 0.191), lexical wins multi-hop (0.420 vs 0.344) and is 7× faster.
   The multi-hop questions name the bridge entity verbatim, which suits exact
   matching — a construction artifact, not a general result.
5. **Latency:** graph traversal is ~300 ms with lexical seeds and no network.
   Vector needs one embedding call. Fixed hybrid pays for both and wins neither.

### The routing result is overfit, and by how much

`AUTO_HI`, `AUTO_LO` and `AUTO_MAX_W` were fit on the same 45 questions reported
above. That is textbook overfitting on a small sample from one person's history.
The honest claim is "this works on my corpus", not "this generalises".

A rejected alternative is worth recording: routing on the phrase "where else /
other project" would have scored better still — and would have been meaningless,
because `make_multihop.py` *instructs* the generator to use that phrasing. The
router would have been detecting my own prompt. Vector self-confidence was chosen
because it is a property of the retrieval, not of how the questions were written.

### Caveats that bound all of the above

- 411 chunks, 25 sessions, 23 active days. Small.
- One gold chunk per question understates recall when a fact appears in several
  chunks. It penalises every mode equally, so comparisons hold and absolute
  numbers read pessimistically.
- Questions are LLM-generated from real chunks, not hand-written. The generator
  saw only chunk text — never the graph, the embeddings, or any retriever — so
  it cannot favour one retriever, but it does inherit the gold chunk's phrasing.
- The single-hop set was built first, and I only recognised it favoured vector
  *after* vector won. That is post-hoc, which is why the mirror-image set exists.
- Some questions still leak literal strings ("video-04", "336 px") despite the
  prompt forbidding it, partially reducing those to string matching.

## What didn't work

**Every embedding was identical and the benchmark looked fine.** Segmind's
embedding endpoint takes `prompt`; I sent OpenAI-style `input`. The unknown field
is ignored and a well-formed 1536-d vector comes back — for a default string. All
411 chunks embedded to the same vector, every cosine was exactly 1.0, and vector
search returned chunks in arbitrary order while reporting plausible results. I
caught it only because `--explain` printed `cos=1.0` on every row. `verify_api()`
now embeds two unrelated sentences and refuses to build the index if they score
above 0.9 (real value: 0.076).

**Three graph bugs, all mine, all silent:**

- `build_structural` collapsed paths to basename, fusing `VAD/README.md` with
  `chat-connect/README.md` into one node spanning 4 projects — fabricating exactly
  the multi-hop paths the project claims to find. Generic filenames are now
  project-qualified.
- `cd` reached degree 159 across 7 projects. Every project runs `cd`, so one node
  became adjacent to everything. Navigation commands and shell keywords (`for`,
  `foreach` — the ref extractor was reading loop syntax as tool names) excluded.
- The extractor wrote `false_alarm_rate` while queries say "false alarm", so the
  graph held the answer and could never reach it. Normalising `_`/`-` to spaces
  merged 52 duplicate nodes and turned a miss into an exact hit.

**Hub explosion.** one project node has degree 347; two hops reached the whole project and
traversal returned a blob. Traversal now refuses to expand *through* high-degree
nodes — including seed nodes, which was the version that mattered.

**A 3-question probe lied.** It showed semantic seeds beating lexical 0.583 vs
0.333 and I used it to justify 2,579 entity embeddings. At full scale the result
splits by eval set. Three questions is a smoke test, not evidence.

**Cost of a masked failure.** The first extraction run burned credits and
committed nothing: `pool.map` yields in submission order, commits fired every 25
results, and stdout was buffered. Now `as_completed`, commit per result,
unbuffered — a kill cannot discard paid-for work.

## Design decisions

**SQLite, not Neo4j.** Docker is not installed and the machine has ~890 MB free
RAM of 7.5 GB. At 2,579 nodes, traversal is a recursive CTE taking ~300 ms;
Neo4j would buy resume signalling, not performance. `graph.py` keeps edges in a
plain table, so a Neo4j export is ~30 lines if it is ever worth it.

**One LLM call per chunk, not batched.** gpt-5-nano costs ≈0.000345 credits/call,
so batching would trade exact provenance for a saving too small to measure.

**`text-embedding-3-small`, not large.** 1536-d at 0.9 s vs 3072-d at 7.2 s. At
411 chunks the extra dimensions are not measurable; the 8× latency is.
Note: passing a **list** to the embedding endpoint silently returns only the
first element's vector, so `Embedder` raises `TypeError` on a list rather than
letting a batching "optimization" corrupt the index.

**Pure-Python cosine.** 411 × 1536 pre-normalized float32 vectors, dot product
over `array('f')`, ~50 ms. An ANN index would be slower at this size.

## MCP server

```json
"memory-graph": {
  "command": "C:\\Python314\\python.exe",
  "args": ["D:\\Projects - AI ML LLMs\\personal brain\\server.py"],
  "env": {"PYTHONIOENCODING": "utf-8"}
}
```

Verified over a real stdio transport (`initialize` → `tools/list` → `tools/call`),
not just by importing the module. MCP 2.x renamed `FastMCP` to `MCPServer`.

`add_memory` embeds the note and refreshes the in-memory index so it is
searchable in the same session; if embedding fails it rolls back, because a chunk
with no vector is invisible to search and worse than not saving at all.

## Known limitations

- `get_related_entities` spends its result budget on seed variants when semantic
  seeding returns many near-identical matches (`Remotion`, `Remotion Studio`,
  `Remotion pipeline`). Seeds should be capped separately from results.
- `add_memory` does not extract triples — the note is vector-searchable at once
  but reaches the graph only on the next `extract.py` run.
- Reindexing is manual. No file watcher, no cron.
- The corpus has real cross-project **tool** reuse (Remotion, React, Python,
  Google Fonts) but thin cross-project **concept** reuse — the same kind of work
  was rarely done twice. That bounds how much any graph method can recover.
- `auto` uses lexical seeds throughout, because they win on the multi-hop set.
  Semantic seeds win on single-hop, so seeding could itself be routed. Untested.
- 45 questions is too few to separate a 0.02 MRR difference from noise. Treat
  only the large gaps (0.15+) as real.

## A note on the examples

Project names in this README are anonymized (`project-a`, `video-04`). Every
number, bug and behaviour described is real and came from the author's own
corpus; only the project labels are placeholders. The eval question sets are not
published — they are generated from private conversations — but the harness that
builds them (`make_evalset.py`, `make_multihop.py`) and every result file are, so
the method is reproducible against your own history.

## Privacy

Personal use only. Transcripts include unpublished research and patent material.
`memory.db`, `.env` and `api.txt` are gitignored. Chunk text does leave the
machine for entity extraction and embedding (Segmind); nothing else does.
