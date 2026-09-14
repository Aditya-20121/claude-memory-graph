"""MCP server exposing the memory graph to any MCP client.

Run directly for stdio:
    python server.py

Tools:
    search_memory(query, top_k, mode)      hybrid retrieval over graph + vectors
    get_related_entities(entity, hops)     pure graph traversal
    add_memory(text, source)               ingest a note now, no reindex needed

The heavy state -- 411 chunk vectors, 2579 entity vectors -- is loaded once at
startup and reused. Reloading per call would add ~50ms and, worse, would miss
writes made by add_memory in the same session.
"""
import os
import sqlite3
import sys
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer

import embed
import embed_entities
import graph
import retrieve
from schema import SCHEMA_SQL

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("MEMORY_DB", os.path.join(HERE, "memory.db"))

mcp = MCPServer(
    name="claude-memory-graph",
    instructions=(
        "Persistent memory over the user's own Claude Code history: a knowledge "
        "graph plus a vector index over their past conversations. Use "
        "search_memory to recall what the user did or decided before; "
        "get_related_entities to explore how their tools, files and concepts "
        "connect across projects; add_memory to record something new."),
)


class State:
    """Lazily opened so importing this module never touches the network or disk."""

    def __init__(self):
        self.db = None
        self.embedder = None
        self.index = None
        self.ent_index = None

    def load(self):
        if self.db is not None:
            return self
        key = embed.api_key()
        if not key:
            raise RuntimeError("no SEGMIND_API_KEY in environment or .env")
        self.db = graph.connect(DB_PATH)
        self.db.executescript(SCHEMA_SQL)
        self.db.executescript(embed.SCHEMA)
        self.db.executescript(embed_entities.SCHEMA)
        self.embedder = embed.Embedder(key)
        self.index = embed.load_index(self.db)
        self.ent_index = embed_entities.load(self.db)
        return self

    def refresh_index(self):
        self.index = embed.load_index(self.db)


STATE = State()


@mcp.tool(description="Search the user's own conversation history. Returns past "
                      "exchanges with project, date and why each one matched.")
def search_memory(query: str, top_k: int = 5, mode: str = "auto") -> dict:
    """Adaptive retrieval over vectors and the knowledge graph.

    mode: "auto" (default -- vector confidence sets how much graph to mix in),
    "vector" (embeddings only), "graph" (traversal only), "hybrid" (fixed-weight
    fusion, kept for comparison; it loses to both its inputs).
    """
    if mode not in ("auto", "hybrid", "vector", "graph"):
        return {"error": f"mode must be auto, hybrid, vector or graph; got {mode!r}"}
    top_k = max(1, min(int(top_k), 20))
    state = STATE.load()
    out = retrieve.retrieve(state.db, state.embedder, query, mode=mode,
                            top_k=top_k, index=state.index,
                            ent_index=state.ent_index, semantic_seeds=False)
    return {
        "query": query,
        "mode": mode,
        "entities_matched": out["seeds"],
        "graph_weight": out["graph_weight"],
        "latency_ms": out["latency_ms"],
        "results": [{
            "text": r["text"][:2500],
            "project": r["project"],
            "session_title": r["title"],
            "date": (r["ts"] or "")[:10],
            "score": r["score"],
            "matched_by": ("both" if r["why"]["vector_rank"] and r["why"]["graph_rank"]
                           else "vector" if r["why"]["vector_rank"] else "graph"),
        } for r in out["results"]],
    }


@mcp.tool(description="Explore how one entity connects to others in the user's "
                      "history: which tools, files, concepts and projects relate to it.")
def get_related_entities(entity: str, hops: int = 2, limit: int = 40) -> dict:
    """Pure graph traversal, no embeddings. Hub nodes are not traversed through."""
    hops = max(1, min(int(hops), 4))
    state = STATE.load()
    seeds = graph.resolve_semantic(state.db, entity, state.embedder,
                                   index=state.ent_index)
    if not seeds:
        return {"entity": entity, "found": False,
                "message": "no entity in the graph matches that"}

    reached = graph.neighbors(state.db, [s["id"] for s in seeds],
                              hops=hops, limit=limit)
    edges = graph.edges_between(state.db, [r["id"] for r in reached], limit=120)
    return {
        "entity": entity,
        "found": True,
        "seeds": [{"name": s["name"], "kind": s["kind"]} for s in seeds],
        "related": [{"name": r["name"], "kind": r["kind"], "hops": r["hops"]}
                    for r in reached],
        "connections": [{
            "subject": e["s"], "predicate": e["p"], "object": e["o"],
            "project": e["project"], "date": (e["ts"] or "")[:10],
            "source": e["src"],
        } for e in edges],
    }


@mcp.tool(description="Save a note into memory so it is searchable from now on. "
                      "Use when the user says to remember something.")
def add_memory(text: str, source: str = "note") -> dict:
    """Ingest one note: store it, embed it, make it immediately searchable.

    Entity extraction is deliberately skipped -- it needs a second LLM round trip
    and would make the tool slow and failure-prone. The note is reachable by
    vector search at once; `python extract.py` folds it into the graph later.
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    state = STATE.load()
    now = datetime.now(timezone.utc).isoformat()
    session_id = f"note:{source}"

    state.db.execute(
        "INSERT OR IGNORE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
        (session_id, "note", source, None, f"notes ({source})", now, now, 0, ""))
    seq = state.db.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM chunks WHERE session_id=?",
        (session_id,)).fetchone()[0]

    cur = state.db.execute(
        "INSERT INTO chunks (session_id, seq, part, ts, project, text, n_chars)"
        " VALUES (?,?,0,?,?,?,?)", (session_id, seq, now, source, text, len(text)))
    chunk_id = cur.lastrowid
    state.db.execute(
        "UPDATE sessions SET n_exchanges = n_exchanges + 1, ended_at = ? WHERE id = ?",
        (now, session_id))

    try:
        vec = state.embedder(text)
    except (RuntimeError, TypeError) as e:
        state.db.rollback()          # an unsearchable chunk is worse than none
        return {"ok": False, "error": f"embedding failed, nothing saved: {e}"}

    state.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                     (chunk_id, embed.DIM, embed.to_blob(vec), embed.MODEL))
    state.db.commit()
    state.refresh_index()            # searchable in this same session
    return {"ok": True, "chunk_id": chunk_id, "source": source,
            "chars": len(text), "searchable": True}


def selfcheck():
    """Exercise the tools against a scratch DB, no MCP transport involved."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    global DB_PATH
    DB_PATH = path
    try:
        con = sqlite3.connect(path)
        con.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, project TEXT,"
            " cwd TEXT, title TEXT, started_at TEXT, ended_at TEXT,"
            " n_exchanges INTEGER, models TEXT);"
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id TEXT, seq INTEGER,"
            " part INTEGER, ts TEXT, project TEXT, text TEXT, n_chars INTEGER,"
            " UNIQUE(session_id, seq, part));"
            "CREATE TABLE refs (chunk_id INTEGER, kind TEXT, value TEXT);")
        con.executescript(SCHEMA_SQL)
        con.executescript(embed.SCHEMA)
        con.executescript(embed_entities.SCHEMA)
        con.execute("ALTER TABLE triples ADD COLUMN src TEXT DEFAULT 'llm'")
        con.commit()
        con.close()

        assert search_memory("x", mode="bogus")["error"], "bad mode must be rejected"
        assert add_memory("")["ok"] is False, "empty note must be rejected"

        # a failing embedder must leave nothing behind
        class Boom:
            def __call__(self, text):
                raise RuntimeError("no network")

        STATE.db = graph.connect(path)
        STATE.embedder = Boom()
        STATE.index, STATE.ent_index = [], []
        before = STATE.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        got = add_memory("something worth remembering")
        after = STATE.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert got["ok"] is False and "embedding failed" in got["error"], got
        assert before == after, "a chunk with no vector must not survive"

        # and a working one must land and be searchable
        STATE.embedder = lambda t: [0.1] * embed.DIM
        ok = add_memory("remotion transitions were the slow part", source="test")
        assert ok["ok"] and ok["searchable"], ok
        rows = STATE.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert rows == 1, rows
        assert len(STATE.index) == 1, "index must refresh so the note is findable now"
        print("selfcheck ok")
    finally:
        if STATE.db is not None:
            STATE.db.close()     # Windows will not unlink a file sqlite still holds
        STATE.db = None
        os.unlink(path)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        mcp.run(transport="stdio")
