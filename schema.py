"""Closed vocabulary for the knowledge graph.

Deliberately small. A freeform extractor invents a new predicate for every
sentence and the graph stops being traversable; anything outside these sets is
dropped at validation rather than stored.
"""

ENTITY_KINDS = {
    "technology",  # library, framework, model, service, language  (PyTorch, Remotion, Sonnet)
    "concept",     # technique, method, metric, architecture       (majority voting, RRF, FA rate)
    "artifact",    # file, dataset, document, deliverable          (STATUS.md, ACL poster, VisDrone)
    "problem",     # bug, blocker, error, limitation               (OOM on 8GB, flicker at 384px)
    "decision",    # a choice that was made                        (switched to two-stage cascade)
    "project",     # a body of work                                (VAD, chat-connect)
    "person",      # a person or organization                      (Aditya, TCS, Anthropic)
}

# subject -> object. Kept transitive-ish so multi-hop traversal means something.
PREDICATES = {
    "uses",         # VAD uses PyTorch
    "implements",   # score.py implements majority voting
    "evaluates",    # capacity_probe evaluates FA rate
    "causes",       # 384px downscale causes flicker
    "fixes",        # two-stage cascade fixes latency blocker
    "replaces",     # Remotion replaces ffmpeg pipeline
    "depends_on",   # ingestion depends on transcript schema
    "compared_to",  # Haiku compared_to Sonnet
    "part_of",      # capacity_probe part_of VAD
    "blocked_by",   # eval blocked_by 9-of-20 windows
    "decided",      # Aditya decided two-stage cascade
}

# Structural relations, derived from tool calls in Phase 2. Never LLM-generated.
STRUCTURAL = {"touched", "ran", "searched", "fetched", "in_project", "in_session"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY, name TEXT, kind TEXT, norm TEXT,
  UNIQUE(norm, kind));
CREATE TABLE IF NOT EXISTS triples (
  id INTEGER PRIMARY KEY, subj INTEGER, pred TEXT, obj INTEGER,
  chunk_id INTEGER, session_id TEXT, ts TEXT, project TEXT,
  UNIQUE(subj, pred, obj, chunk_id));
CREATE TABLE IF NOT EXISTS extracted (
  chunk_id INTEGER PRIMARY KEY, n_triples INTEGER, error TEXT, at TEXT);
CREATE INDEX IF NOT EXISTS triples_subj ON triples(subj);
CREATE INDEX IF NOT EXISTS triples_obj ON triples(obj);
CREATE INDEX IF NOT EXISTS entities_norm ON entities(norm);
"""
