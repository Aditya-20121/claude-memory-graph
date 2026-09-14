"""Normalize Claude Code transcripts + Claude Desktop session metadata into memory.db.

Usage:
    python ingest.py              # ingest, then print stats
    python ingest.py --stats      # stats only
    python ingest.py --selfcheck  # run the parser self-check

Sources (all local):
    ~/.claude/projects/<slug>/<uuid>.jsonl                full transcripts
    %LOCALAPPDATA%/Packages/Claude_pzs8sxrjxfjjc/...      desktop session titles
    <project>/CLAUDE.md                                   curated project context
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
TRANSCRIPTS = os.path.join(HOME, ".claude", "projects")
DESKTOP = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Packages", "Claude_pzs8sxrjxfjjc",
    "LocalCache", "Roaming", "Claude", "claude-code-sessions")
# Directories to scan for CLAUDE.md project context. Override with
# CLAUDE_MD_ROOTS="C:/code;D:/work" (os.pathsep-separated); defaults to none.
CLAUDE_MD_ROOTS = [p for p in os.environ.get("CLAUDE_MD_ROOTS", "").split(os.pathsep) if p]

MAX_CHARS = 6000  # ponytail: fixed cap; revisit if the embedding model's window changes

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, source TEXT, project TEXT, cwd TEXT, title TEXT,
  started_at TEXT, ended_at TEXT, n_exchanges INTEGER, models TEXT);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY, session_id TEXT, seq INTEGER, part INTEGER,
  ts TEXT, project TEXT, text TEXT, n_chars INTEGER,
  UNIQUE(session_id, seq, part));
CREATE TABLE IF NOT EXISTS refs (
  chunk_id INTEGER, kind TEXT, value TEXT);
CREATE INDEX IF NOT EXISTS refs_kind ON refs(kind, value);
CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);
"""


def human_prompt(rec):
    """The text of a typed user message, or None for tool results / injected blocks."""
    if rec.get("type") != "user":
        return None
    content = rec["message"]["content"]
    if isinstance(content, list):
        if any(b.get("type") == "tool_result" for b in content):
            return None
        content = "\n".join(
            b.get("text", "") for b in content if b.get("type") == "text")
    content = content.strip()
    # <command-name>, <local-command-stdout>, <system-reminder> wrappers carry no intent
    return content if content and not content.startswith("<") else None


def extract_refs(block):
    """Structural facts from a tool_use block -- no LLM, so nothing to hallucinate."""
    name = block.get("name", "")
    inp = block.get("input")
    if not isinstance(inp, dict):
        return
    if inp.get("file_path"):
        yield "file", inp["file_path"]
    if inp.get("command"):
        match = re.match(r"\s*([\w.-]+)", str(inp["command"]))
        if match:  # first bare word is the program; enough for a graph edge
            yield "command", match.group(1)
    if inp.get("url"):
        yield "url", inp["url"]
    if inp.get("query"):
        yield "search", str(inp["query"])[:200]
    if name.startswith("mcp__"):
        yield "mcp", name.split("__")[1]


def split(text):
    """Chunk oversized exchanges on paragraph boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n\n"):
        if buf and len(buf) + len(para) + 2 > MAX_CHARS:
            parts.append(buf)
            buf = ""
        while len(para) > MAX_CHARS:  # a single paragraph over the cap: hard split
            parts.append(para[:MAX_CHARS])
            para = para[MAX_CHARS:]
        buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        parts.append(buf)
    return parts


def parse_transcript(path):
    """Return (exchanges, meta). An exchange is one human prompt + the replies it drew."""
    recs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    meta = {"cwd": None, "models": set(), "start": None, "end": None}
    exchanges = []
    seq, prompt, ts, body, refs = 0, None, None, [], []

    def flush():
        if prompt is None:
            return
        text = "USER: " + prompt
        if body:
            text += "\n\nASSISTANT: " + "\n\n".join(body)
        exchanges.append((seq, ts, text, refs))

    for rec in recs:
        if rec.get("cwd"):
            meta["cwd"] = rec["cwd"]
        stamp = rec.get("timestamp")
        if stamp:
            meta["start"] = min(meta["start"] or stamp, stamp)
            meta["end"] = max(meta["end"] or stamp, stamp)

        prompt_text = human_prompt(rec)
        if prompt_text is not None:
            flush()
            seq, prompt, ts, body, refs = seq + 1, prompt_text, stamp, [], []
        elif rec.get("type") == "assistant" and prompt is not None:
            msg = rec["message"]
            if msg.get("model"):
                meta["models"].add(msg["model"])
            for block in msg.get("content", []):
                kind = block.get("type")
                if kind == "text" and block.get("text", "").strip():
                    body.append(block["text"].strip())
                elif kind == "tool_use":
                    refs.extend(extract_refs(block))
    flush()
    return exchanges, meta


def desktop_titles():
    """cliSessionId -> title, from the MSIX-virtualized Claude Desktop store."""
    out = {}
    for path in glob.glob(os.path.join(DESKTOP, "*", "*", "local_*.json")):
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("cliSessionId") and rec.get("title"):
            out[rec["cliSessionId"]] = rec["title"]
    return out


def add_chunks(db, session_id, project, exchanges):
    added = 0
    for seq, ts, text, refs in exchanges:
        for part, piece in enumerate(split(text)):
            cur = db.execute(
                "INSERT OR IGNORE INTO chunks"
                " (session_id, seq, part, ts, project, text, n_chars)"
                " VALUES (?,?,?,?,?,?,?)",
                (session_id, seq, part, ts, project, piece, len(piece)))
            added += cur.rowcount
            if cur.rowcount and part == 0 and refs:
                db.executemany("INSERT INTO refs VALUES (?,?,?)",
                               [(cur.lastrowid, k, v) for k, v in refs])
    return added


def ingest(db):
    db.executescript(SCHEMA)
    titles = desktop_titles()
    n_sessions = n_chunks = 0

    for path in glob.glob(os.path.join(TRANSCRIPTS, "**", "*.jsonl"), recursive=True):
        session_id = os.path.basename(path)[:-6]
        project = os.path.relpath(path, TRANSCRIPTS).split(os.sep)[0]
        exchanges, meta = parse_transcript(path)
        if not exchanges:
            continue
        db.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, "claude_code", project, meta["cwd"], titles.get(session_id),
             meta["start"], meta["end"], len(exchanges), ",".join(sorted(meta["models"]))))
        n_sessions += 1
        n_chunks += add_chunks(db, session_id, project, exchanges)

    for root in CLAUDE_MD_ROOTS:
        for path in glob.glob(os.path.join(root, "**", "CLAUDE.md"), recursive=True):
            if "node_modules" in path:
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            session_id = "claude_md:" + rel
            project = os.path.basename(os.path.dirname(path))
            ts = datetime.fromtimestamp(
                os.path.getmtime(path), timezone.utc).isoformat()
            db.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, "claude_md", project, os.path.dirname(path),
                 f"CLAUDE.md ({project})", ts, ts, 1, ""))
            n_sessions += 1
            n_chunks += add_chunks(db, session_id, project, [(0, ts, text, [])])

    db.commit()
    return n_sessions, n_chunks


def stats(db):
    one = lambda sql: db.execute(sql).fetchone()[0]
    print(f"sessions : {one('SELECT COUNT(*) FROM sessions')}")
    print(f"chunks   : {one('SELECT COUNT(*) FROM chunks')}")
    print(f"chars    : {one('SELECT SUM(n_chars) FROM chunks'):,}")
    print(f"refs     : {one('SELECT COUNT(*) FROM refs')}")

    print("\nper project:")
    for proj, sess, n, chars in db.execute(
            "SELECT project, COUNT(DISTINCT session_id), COUNT(*), SUM(n_chars)"
            " FROM chunks GROUP BY 1 ORDER BY 3 DESC"):
        print(f"  {n:5d} chunks {chars:9,} chars  {sess:2d} sess  {proj}")

    print("\nref kinds:")
    for kind, n, uniq in db.execute(
            "SELECT kind, COUNT(*), COUNT(DISTINCT value) FROM refs"
            " GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {kind:9s} {n:5d} ({uniq} unique)")

    print("\ntitled sessions:")
    for title, proj, n in db.execute(
            "SELECT title, project, n_exchanges FROM sessions"
            " WHERE title IS NOT NULL ORDER BY started_at"):
        print(f"  {n:4d} ex  {proj[:34]:34s} {title}")


def selfcheck():
    stamp = lambda **kw: {"timestamp": "2026-01-01T00:00:00Z", **kw}
    user = lambda c: stamp(type="user", message={"content": c}, cwd="C:\\x")
    lines = [
        user("first question"),
        stamp(type="assistant", message={"model": "m", "content": [
            {"type": "text", "text": "answer one"},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}}]}),
        user([{"type": "tool_result", "content": "noise"}]),   # dropped
        user("<system-reminder>ignore me</system-reminder>"),   # dropped
        stamp(type="attachment", attachment={}),                # dropped
        user("second question"),
        stamp(type="assistant", message={"model": "m", "content": [
            {"type": "thinking", "thinking": "hidden"},         # never enters text
            {"type": "text", "text": "answer two"}]}),
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(line) + "\n" for line in lines)
    try:
        exchanges, meta = parse_transcript(path)
    finally:
        os.unlink(path)

    assert len(exchanges) == 2, f"expected 2 exchanges, got {len(exchanges)}"
    assert exchanges[0][2] == "USER: first question\n\nASSISTANT: answer one"
    assert exchanges[0][3] == [("file", "a.py")], exchanges[0][3]
    assert exchanges[1][2] == "USER: second question\n\nASSISTANT: answer two"
    assert meta["cwd"] == "C:\\x" and meta["models"] == {"m"}

    assert split("a" * 100) == ["a" * 100]
    parts = split(("x" * 5000 + "\n\n") * 3)
    assert all(len(p) <= MAX_CHARS for p in parts), [len(p) for p in parts]
    assert len(split("y" * (MAX_CHARS * 2 + 5))) == 3
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="memory.db")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        sys.exit()
    con = sqlite3.connect(args.db)
    if not args.stats:
        n_sessions, n_chunks = ingest(con)
        print(f"ingested {n_sessions} sessions, {n_chunks} new chunks -> {args.db}\n")
    stats(con)
