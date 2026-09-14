"""Graph construction and traversal over memory.db.

Two edge sources, kept distinguishable by `triples.src` so the eval can ablate them:
  llm         -- extracted by extract.py, conceptual, occasionally wrong
  structural  -- derived from tool calls in Phase 2, cannot be hallucinated

Usage:
    python graph.py --build              # fold refs into the graph
    python graph.py --stats
    python graph.py --near VisDrone --hops 2
"""
import argparse
import os
import re
import sqlite3
import sys

from extract import project_name  # single source of truth for the project label

HERE = os.path.dirname(os.path.abspath(__file__))

# ref kind -> (entity kind, predicate linking it to its project)
REF_MAP = {
    "file": ("artifact", "part_of"),
    "command": ("technology", "uses"),
    "mcp": ("technology", "uses"),
}

# Every project runs `cd`. Keeping these makes one hub node adjacent to everything
# and multi-hop traversal stops meaning anything -- they are navigation, not tools.
NAV_COMMANDS = {
    "cd", "ls", "dir", "cat", "echo", "cp", "mv", "rm", "del", "mkdir", "rmdir",
    "touch", "head", "tail", "less", "more", "grep", "find", "wc", "sort", "uniq",
    "cut", "sed", "awk", "tr", "pwd", "which", "where", "type", "export", "set",
    "sleep", "clear", "exit", "true", "false", "test", "chmod", "chown", "du", "df",
    "get-childitem", "get-content", "set-content", "remove-item", "new-item",
    "select-object", "where-object", "foreach-object", "measure-object", "out-file",
    "write-host", "write-output", "get-process", "start-sleep", "test-path", "copy-item",
    # shell keywords -- the ref extractor takes the first bare word of a command,
    # so `for f in *; do ...` lands here as a "tool"
    "for", "foreach", "while", "do", "done", "if", "then", "else", "fi", "case", "esac",
    "timeout", "env", "source", "eval", "exec", "read", "printf", "xargs", "tee",
}

# Filenames that exist in every repo. Unqualified they fuse unrelated projects
# into one node, inventing multi-hop paths that do not exist.
GENERIC_FILES = {
    "readme.md", "requirements.txt", "claude.md", "package.json", "manifest.json",
    "index.js", "index.ts", "index.html", "main.py", "app.py", "setup.py", "config.json",
    "content.js", "style.css", "styles.css", "tsconfig.json", ".gitignore", ".env",
    "dockerfile", "makefile", "license", "notes.md", "todo.md", "status.md",
    "memory.md", "architecture.md", "plan.md", "schema.py", "utils.py", "test.py",
}

# Generic actors the extractor keeps inventing. They connect everything and mean nothing.
ACTOR_NOISE = {
    "developer", "the developer", "ai assistant", "ai_assistant", "assistant",
    "ai", "the ai", "user", "the user", "human", "claude", "chatgpt user",
    "ai agent", "agent", "engineer", "programmer",
}


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    cols = {r["name"] for r in db.execute("PRAGMA table_info(triples)")}
    if "src" not in cols:
        db.execute("ALTER TABLE triples ADD COLUMN src TEXT DEFAULT 'llm'")
        db.commit()
    return db


def normalize_name(name):
    """Canonical form for entity identity and query matching.

    The extractor writes `false_alarm_rate`, `false-alarm rate` and `false alarm
    rate` for one concept, so they must collapse to one node. It also means a
    natural-language query ("the false alarm rate") can match at all -- an
    underscored name is never a substring of an English sentence.
    """
    return " ".join(re.sub(r"[_\-/]+", " ", name.lower()).split())


def get_entity(db, name, kind, cache):
    norm = normalize_name(name)
    if (hit := cache.get((norm, kind))) is not None:
        return hit
    db.execute("INSERT OR IGNORE INTO entities (name, kind, norm) VALUES (?,?,?)",
               (name, kind, norm))
    eid = db.execute("SELECT id FROM entities WHERE norm=? AND kind=?",
                     (norm, kind)).fetchone()[0]
    cache[(norm, kind)] = eid
    return eid


def build_structural(db):
    """Fold Phase-2 refs into the graph: files and commands hang off their project."""
    db.execute("DELETE FROM triples WHERE src='structural'")
    cache, n, labels = {}, 0, project_labels(db)
    rows = db.execute("""
        SELECT r.kind, r.value, c.id AS chunk_id, c.session_id, c.ts,
               c.project AS slug, s.cwd
        FROM refs r JOIN chunks c ON c.id = r.chunk_id
        JOIN sessions s ON s.id = c.session_id
        WHERE r.kind IN ('file','command','mcp')""").fetchall()

    for r in rows:
        ent_kind, pred = REF_MAP[r["kind"]]
        value = r["value"]
        proj = labels.get(r["slug"]) or project_name(r["slug"], r["cwd"])
        if r["kind"] == "file":
            value = os.path.basename(value.replace("\\", "/").rstrip("/")) or value
            # README.md in two repos is two files; fusing them invents a multi-hop path
            if value.lower() in GENERIC_FILES:
                value = f"{proj}/{value}"
        elif value.lower() in NAV_COMMANDS:
            continue
        if not value or len(value) > 80:
            continue
        subj = get_entity(db, value, ent_kind, cache)
        obj = get_entity(db, proj, "project", cache)
        if subj == obj:
            continue
        cur = db.execute(
            "INSERT OR IGNORE INTO triples"
            " (subj, pred, obj, chunk_id, session_id, ts, project, src)"
            " VALUES (?,?,?,?,?,?,?,'structural')",
            (subj, pred, obj, r["chunk_id"], r["session_id"], r["ts"], r["slug"]))
        n += cur.rowcount

    # every session's chunks also sit inside a project: gives projects a real hub degree
    db.commit()
    return n


def renormalize(db):
    """Recompute every entity's norm and merge whatever now collides.

    Run after changing normalize_name. Cheaper and more honest than re-extracting:
    the model's output has not changed, only our idea of what counts as the same name.
    """
    rows = db.execute("SELECT id, name, kind, norm FROM entities").fetchall()
    canonical, merges = {}, []
    for r in rows:
        key = (normalize_name(r["name"]), r["kind"])
        if key in canonical:
            merges.append((r["id"], canonical[key]))
        else:
            canonical[key] = r["id"]

    for old, new in merges:                       # repoint edges, then drop the dup
        db.execute("UPDATE OR IGNORE triples SET subj=? WHERE subj=?", (new, old))
        db.execute("UPDATE OR IGNORE triples SET obj=? WHERE obj=?", (new, old))
        db.execute("DELETE FROM triples WHERE subj=? OR obj=?", (old, old))
        db.execute("DELETE FROM entities WHERE id=?", (old,))
    for (norm, _kind), eid in canonical.items():
        db.execute("UPDATE entities SET norm=? WHERE id=?", (norm, eid))

    db.execute("DELETE FROM triples WHERE subj=obj")   # merging can create self-loops
    db.commit()
    return len(merges)


def project_labels(db):
    """slug -> human label, resolved once via sessions.cwd.

    build_structural and prune_noise must agree, or the same project ends up
    under two different names and the split filenames stop matching.
    """
    return {r["project"]: project_name(r["project"], r["cwd"])
            for r in db.execute("SELECT DISTINCT project, cwd FROM sessions")}


def prune_noise(db):
    """Clean LLM entities in place; re-extracting would cost 30 minutes for no gain.

    Two jobs: drop generic actors outright, and split generic filenames that were
    merged across projects back into one node per project.
    """
    # the extractor emits the same navigation noise the structural pass filters out
    junk = sorted(ACTOR_NOISE | NAV_COMMANDS)
    victims = [r["id"] for r in db.execute(
        f"SELECT id FROM entities WHERE norm IN ({','.join('?' * len(junk))})", junk)]
    n_actors = 0
    if victims:
        ids = ",".join(str(i) for i in victims)
        n_actors = db.execute(f"DELETE FROM triples WHERE subj IN ({ids}) OR obj IN ({ids})").rowcount
        db.execute(f"DELETE FROM entities WHERE id IN ({ids})")

    # a generic filename touched by >1 project is really N distinct files
    n_split = 0
    cache, labels = {}, project_labels(db)
    shared = db.execute(f"""
        SELECT e.id, e.name, e.kind FROM entities e
        WHERE e.norm IN ({",".join("?" * len(GENERIC_FILES))})
          AND (SELECT COUNT(DISTINCT t.project) FROM triples t
               WHERE t.subj=e.id OR t.obj=e.id) > 1""", sorted(GENERIC_FILES)).fetchall()
    for ent in shared:
        for t in db.execute("SELECT * FROM triples WHERE subj=? OR obj=?",
                            (ent["id"], ent["id"])).fetchall():
            proj = labels.get(t["project"]) or project_name(t["project"], None)
            scoped = get_entity(db, f"{proj}/{ent['name']}", ent["kind"], cache)
            db.execute("UPDATE OR IGNORE triples SET subj=? WHERE id=? AND subj=?",
                       (scoped, t["id"], ent["id"]))
            db.execute("UPDATE OR IGNORE triples SET obj=? WHERE id=? AND obj=?",
                       (scoped, t["id"], ent["id"]))
            n_split += 1
        db.execute("DELETE FROM entities WHERE id=?", (ent["id"],))

    orphans = db.execute("""DELETE FROM entities WHERE NOT EXISTS
                            (SELECT 1 FROM triples t WHERE t.subj=entities.id
                                                        OR t.obj=entities.id)""").rowcount
    db.commit()
    return n_actors, len(shared), n_split, orphans


def resolve(db, text, limit=8):
    """Entities named in the query text. The most specific names win.

    Two ways to match, both on word boundaries -- a plain substring test makes
    "arm" match inside "false alarm" and seeds traversal from a word the user
    never typed:
      1. phrase: the whole entity name appears in the query
      2. all-tokens: every word of a multi-word entity appears somewhere in the
         query, so "false alarm rate" is still found in "the false alarm numbers"
    """
    query = normalize_name(text)
    q_tokens = set(query.split())
    seen, hits = set(), []
    for row in db.execute("SELECT id, name, kind, norm FROM entities"):
        norm = row["norm"]
        if len(norm) <= 2 or norm in seen:
            continue
        tokens = norm.split()
        phrase = re.search(rf"(?<!\w){re.escape(norm)}(?!\w)", query) is not None
        all_tokens = len(tokens) > 1 and all(t in q_tokens for t in tokens)
        if not (phrase or all_tokens):
            continue
        seen.add(norm)
        # prefer multi-word matches, then longer ones: more words matched is more
        # specific, so "false alarm rate" outranks a bare "rate"
        hits.append((len(tokens), len(norm), row["id"], row["name"], row["kind"]))
    hits.sort(reverse=True)
    return [{"id": i, "name": n, "kind": k} for _, _, i, n, k in hits[:limit]]


HUB_DEGREE = 40  # ponytail: tuned by hand against this corpus; re-check if it grows


def degrees(db):
    return dict(db.execute(
        "SELECT id, (SELECT COUNT(*) FROM triples t WHERE t.subj=e.id OR t.obj=e.id)"
        " FROM entities e"))


def resolve_semantic(db, text, embedder, limit=8, floor=0.45, index=None):
    """Seed entities by embedding similarity, falling back to lexical matches.

    Lexical resolution only fires when the query happens to use the extractor's
    wording. It misses "majority voting" -> `ensemble_voting_decision` entirely,
    and it confuses same-name entities from unrelated contexts (the Chrome of
    `chrome.storage.local` vs the Chrome of a browser cache cleanup).
    """
    import embed  # local: graph.py must stay importable without an API key

    lexical = resolve(db, text, limit=limit)
    rows = index if index is not None else db.execute(
        "SELECT entity_id, vec FROM entity_vecs").fetchall()
    if not rows:
        return lexical

    qvec = embed.normalize(embedder(text))
    scored = sorted(
        ((embed.dot(qvec, embed.from_blob(v)), eid) for eid, v in rows), reverse=True)
    top = [(s, eid) for s, eid in scored[:limit] if s >= floor]
    if not top:
        return lexical

    ids = ",".join(str(int(e)) for _, e in top)
    meta = {r["id"]: r for r in db.execute(
        f"SELECT id, name, kind FROM entities WHERE id IN ({ids})")}
    out, seen = [], set()
    for hit in lexical:                       # exact mentions still lead
        out.append(hit)
        seen.add(hit["id"])
    for score, eid in top:
        row = meta.get(eid)
        if row and eid not in seen:
            out.append({"id": eid, "name": row["name"], "kind": row["kind"],
                        "score": round(score, 4)})
            seen.add(eid)
    return out[:limit]


def neighbors(db, entity_ids, hops=2, limit=200, hub_degree=HUB_DEGREE):
    """Undirected BFS out to `hops`, refusing to expand THROUGH hub nodes.

    A project node like VAD has degree 347: two hops through it reaches the whole
    project and traversal stops discriminating. Hubs are still returned when
    reached, they just are not used as a bridge to anything further.
    """
    if not entity_ids:
        return []
    seeds = ",".join(str(int(i)) for i in entity_ids)
    return db.execute(f"""
        WITH RECURSIVE
        deg(id, n) AS (
            SELECT e.id, (SELECT COUNT(*) FROM triples t
                          WHERE t.subj = e.id OR t.obj = e.id) FROM entities e
        ),
        walk(id, d) AS (
            SELECT id, 0 FROM entities WHERE id IN ({seeds})
            UNION
            SELECT CASE WHEN t.subj = w.id THEN t.obj ELSE t.subj END, w.d + 1
            FROM triples t
            JOIN walk w ON (t.subj = w.id OR t.obj = w.id)
            JOIN deg ON deg.id = w.id
            -- the cap applies at hop 0 too: a hub SEED (a query naming "VAD")
            -- would otherwise fan out to its whole project and drown the
            -- specific seed sitting next to it
            WHERE w.d < ? AND deg.n <= ?
        )
        SELECT e.id, e.name, e.kind, MIN(w.d) AS hops
        FROM walk w JOIN entities e ON e.id = w.id
        GROUP BY e.id ORDER BY hops, e.name LIMIT ?""",
        (hops, hub_degree, limit)).fetchall()


def edges_between(db, entity_ids, limit=300):
    """Every edge among a set of entities, with provenance. This is the context payload."""
    if not entity_ids:
        return []
    ids = ",".join(str(int(i)) for i in entity_ids)
    return db.execute(f"""
        SELECT a.name AS s, t.pred AS p, b.name AS o, t.src, t.project, t.ts,
               t.session_id, t.chunk_id
        FROM triples t
        JOIN entities a ON a.id = t.subj
        JOIN entities b ON b.id = t.obj
        WHERE t.subj IN ({ids}) AND t.obj IN ({ids})
        ORDER BY t.src DESC, t.ts LIMIT ?""", (limit,)).fetchall()


def stats(db):
    one = lambda sql: (db.execute(sql).fetchone() or [0])[0]
    print(f"entities : {one('SELECT COUNT(*) FROM entities')}")
    print(f"triples  : {one('SELECT COUNT(*) FROM triples')}"
          f"  (llm {one(chr(39).join(['SELECT COUNT(*) FROM triples WHERE src=', 'llm', '']))},"
          f" structural {one(chr(39).join(['SELECT COUNT(*) FROM triples WHERE src=', 'structural', '']))})")
    iso = one("""SELECT COUNT(*) FROM entities e WHERE NOT EXISTS
                 (SELECT 1 FROM triples t WHERE t.subj=e.id OR t.obj=e.id)""")
    print(f"isolated : {iso}")

    print("\nhubs (degree, cross-project reach):")
    for r in db.execute("""
            SELECT e.name, e.kind, COUNT(*) deg, COUNT(DISTINCT t.project) projs
            FROM entities e JOIN triples t ON t.subj=e.id OR t.obj=e.id
            GROUP BY e.id ORDER BY deg DESC LIMIT 12"""):
        span = f"  [{r['projs']} projects]" if r["projs"] > 1 else ""
        print(f"  {r['deg']:4d}  {r['kind']:11s} {r['name']}{span}")

    print("\nentities appearing in more than one project (multi-hop bridges):")
    rows = db.execute("""
            SELECT e.name, e.kind, COUNT(DISTINCT t.project) projs
            FROM entities e JOIN triples t ON t.subj=e.id OR t.obj=e.id
            GROUP BY e.id HAVING projs > 1 ORDER BY projs DESC, e.name LIMIT 20""").fetchall()
    for r in rows:
        print(f"  {r['projs']}x  {r['kind']:11s} {r['name']}")
    if not rows:
        print("  (none -- multi-hop questions will be weak)")


def show_near(db, term, hops):
    seeds = resolve(db, term)
    if not seeds:
        print(f"no entity matches {term!r}")
        return
    print("seeds:", ", ".join(f"{s['name']} ({s['kind']})" for s in seeds))
    rows = neighbors(db, [s["id"] for s in seeds], hops=hops)
    print(f"\n{len(rows)} entities within {hops} hops:")
    for r in rows:
        print(f"  {r['hops']}  {r['kind']:11s} {r['name']}")
    print("\nedges:")
    for e in edges_between(db, [r["id"] for r in rows])[:40]:
        tag = "~" if e["src"] == "llm" else "="
        print(f"  {tag} ({e['s']}) -[{e['p']}]-> ({e['o']})   {e['project'][:26]}")


def selfcheck():
    from schema import SCHEMA_SQL
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    db.execute("ALTER TABLE triples ADD COLUMN src TEXT DEFAULT 'llm'")
    cache = {}
    edge_sql = ("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
                " VALUES (?,'uses',?,1,'s','t','p','llm')")
    names = ("alpha", "beta", "gamma", "delta", "zulu")
    ids = {n: get_entity(db, n, "concept", cache) for n in names}
    chain = [("alpha", "beta"), ("beta", "gamma"), ("gamma", "delta")]  # zulu stays isolated
    for s, o in chain:
        db.execute("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
                   " VALUES (?,'uses',?,1,'s','t','p','llm')", (ids[s], ids[o]))
    db.commit()

    at = lambda h: {r["name"]: r["hops"] for r in neighbors(db, [ids["alpha"]], hops=h)}
    assert at(1) == {"alpha": 0, "beta": 1}, at(1)
    assert at(2) == {"alpha": 0, "beta": 1, "gamma": 2}, at(2)
    assert at(3) == {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}, at(3)
    assert "zulu" not in at(3), "isolated node must not appear"
    # traversal is undirected: walking back from the tail reaches the head
    assert at(3).keys() == {r["name"] for r in neighbors(db, [ids["delta"]], hops=3)}

    found = {r["name"] for r in neighbors(db, [ids["alpha"]], hops=2)}
    assert len(edges_between(db, [ids[n] for n in found])) == 2, "alpha-beta, beta-gamma"
    assert resolve(db, "we tried alpha and also delta today")[0]["name"] in ("alpha", "delta")
    # word boundaries: "arm" must not match inside "false alarm"
    arm = get_entity(db, "arm", "concept", {})
    db.execute("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
               " VALUES (?,'uses',?,1,'s','t','p','llm')", (arm, ids["alpha"]))
    db.commit()
    assert not any(h["name"] == "arm" for h in resolve(db, "the false alarm rate")),         "substring match would seed traversal from a word the user never typed"
    assert any(h["name"] == "arm" for h in resolve(db, "the arm moved")), "real mention"

    # snake_case / kebab-case must collapse to one node and match English queries
    assert normalize_name("false_alarm_rate") == "false alarm rate"
    assert normalize_name("false-alarm  Rate") == "false alarm rate"
    c3 = {}
    snake = get_entity(db, "false_alarm_rate", "concept", c3)
    kebab = get_entity(db, "false-alarm rate", "concept", c3)
    assert snake == kebab, "underscore and hyphen spellings are the same entity"
    db.execute(edge_sql, (snake, ids["alpha"]))
    db.commit()
    named = [h["name"] for h in resolve(db, "what was the false alarm rate here")]
    assert any(normalize_name(n) == "false alarm rate" for n in named), named
    # all-token match: entity words present but not contiguous
    loose = [h["name"] for h in resolve(db, "the false alarm numbers and the rate")]
    assert any(normalize_name(n) == "false alarm rate" for n in loose), loose

    # a hub stays a legitimate result but must not bridge to anything beyond it
    # (last: the filler nodes below would pollute the assertions above)
    hub = get_entity(db, "hubhub", "project", cache)
    far = get_entity(db, "faraway", "concept", cache)
    edge = ("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
            " VALUES (?,'uses',?,1,'s','t','p','llm')")
    db.execute(edge, (ids["alpha"], hub))
    for i in range(12):   # inflate the hub's degree past the test threshold
        db.execute(edge, (hub, get_entity(db, f"filler{i}", "concept", cache)))
    db.execute(edge, (hub, far))
    db.commit()
    capped = {r["name"] for r in neighbors(db, [ids["alpha"]], hops=2, hub_degree=5)}
    assert "hubhub" in capped, "the hub itself is still a legitimate result"
    assert "faraway" not in capped, "traversal must not tunnel through a hub"
    assert "faraway" in {r["name"] for r in
                         neighbors(db, [ids["alpha"]], hops=2, hub_degree=999)}, \
        "with the cap lifted the same path must reappear"
    assert resolve(db, "nothing relevant here") == []
    # generic filenames touched by two projects must split into two nodes
    db2 = sqlite3.connect(":memory:"); db2.row_factory = sqlite3.Row
    db2.executescript(SCHEMA_SQL)
    db2.execute("ALTER TABLE triples ADD COLUMN src TEXT DEFAULT 'llm'")
    db2.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, project TEXT,"
                " cwd TEXT, title TEXT, started_at TEXT, ended_at TEXT,"
                " n_exchanges INTEGER, models TEXT)")
    c2 = {}
    # slugs + cwd, so labels resolve the same way they do for real data
    slugs = {"D--Projects---AI-ML-LLMs-VAD": "D:\\Projects - AI ML LLMs\\VAD",
             "d--Projects---AI-ML-LLMs-chat-connect": "D:\\x\\chat-connect"}
    for slug, cwd in slugs.items():
        db2.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                    (slug, "claude_code", slug, cwd, None, None, None, 1, ""))
    readme = get_entity(db2, "README.md", "artifact", c2)
    for slug in slugs:
        other = get_entity(db2, f"thing-{slug[-6:]}", "concept", c2)
        db2.execute("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
                    " VALUES (?,'part_of',?,1,'s','t',?,'llm')", (readme, other, slug))
    noise = get_entity(db2, "developer", "person", c2)
    keep = get_entity(db2, "Remotion", "technology", c2)
    db2.execute("INSERT INTO triples (subj,pred,obj,chunk_id,session_id,ts,project,src)"
                " VALUES (?,'uses',?,1,'s','t',?,'llm')",
                (noise, keep, "D--Projects---AI-ML-LLMs-VAD"))
    db2.commit()
    prune_noise(db2)
    names = {r[0] for r in db2.execute("SELECT name FROM entities")}
    assert "README.md" not in names, names
    assert {"VAD/README.md", "chat-connect/README.md"} <= names, names
    assert "developer" not in names, names
    assert "Remotion" not in names, "orphaned by the actor purge, so it should go too"
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--renormalize", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--near")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit()
    con = connect(args.db)
    if args.renormalize or args.build:
        print(f"renormalized: {renormalize(con)} duplicate entities merged")
    if args.build:
        print(f"structural edges: {build_structural(con)}")
    if args.build or args.prune:
        actors, files, edges, orphans = prune_noise(con)
        print(f"pruned: {actors} actor edges, {files} shared filenames split over "
              f"{edges} edges, {orphans} orphan entities\n")
    if args.near:
        show_near(con, args.near, args.hops)
    else:
        stats(con)
