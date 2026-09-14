"""Render the knowledge graph as a self-contained interactive HTML file.

Usage:
    python visualize.py                  # top 300 nodes by degree -> graph.html
    python visualize.py --nodes 600
    python visualize.py --project VAD    # only one project's subgraph
    python visualize.py --anonymize      # replace entity names with kind+id

Writes one HTML file with the data inlined. No CDN, no network, no server --
the graph contains private conversation content, so it must open from disk with
nothing leaving the machine. Use --anonymize to produce a shareable structural
view with the names stripped.

1,776 of 2,579 entities have degree 1. Rendering everything gives a hairball of
leaves, so the default view is the top N by degree with edges among them.
"""
import argparse
import json
import os

import graph

HERE = os.path.dirname(os.path.abspath(__file__))

KIND_COLORS = {
    "technology": "#4c9aff",
    "concept": "#a78bfa",
    "artifact": "#34d399",
    "project": "#fbbf24",
    "problem": "#f87171",
    "decision": "#f472b6",
    "person": "#94a3b8",
}


def collect(db, limit, project=None, anonymize=False):
    where = "WHERE t.project LIKE ?" if project else ""
    args = [f"%{project}%"] if project else []
    top = db.execute(f"""
        SELECT e.id, e.name, e.kind, COUNT(*) AS deg
        FROM entities e JOIN triples t ON (t.subj = e.id OR t.obj = e.id)
        {where}
        GROUP BY e.id ORDER BY deg DESC LIMIT ?""", (*args, limit)).fetchall()
    if not top:
        return [], []

    ids = {r["id"] for r in top}
    id_list = ",".join(str(i) for i in ids)
    edges = db.execute(f"""
        SELECT t.subj, t.obj, t.pred, t.src, t.project,
               COUNT(DISTINCT t.chunk_id) AS n
        FROM triples t
        WHERE t.subj IN ({id_list}) AND t.obj IN ({id_list})
        GROUP BY t.subj, t.obj, t.pred""").fetchall()

    index = {r["id"]: i for i, r in enumerate(top)}
    counter = {}
    nodes = []
    for r in top:
        if anonymize:
            counter[r["kind"]] = counter.get(r["kind"], 0) + 1
            label = f"{r['kind']}-{counter[r['kind']]}"
        else:
            label = r["name"]
        nodes.append({"id": index[r["id"]], "name": label,
                      "kind": r["kind"], "deg": r["deg"]})
    links = [{"s": index[e["subj"]], "t": index[e["obj"]], "p": e["pred"],
              "src": e["src"], "n": e["n"]}
             for e in edges if e["subj"] in index and e["obj"] in index]
    return nodes, links


HTML = """<!doctype html>
<meta charset="utf-8">
<title>claude-memory-graph</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d;
    --fg:#e6edf3; --dim:#8b949e;
  }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; overflow:hidden;
    background:var(--bg); color:var(--fg);
    font:13px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  #wrap { display:flex; height:100%; }
  canvas { flex:1; display:block; cursor:grab; }
  canvas.drag { cursor:grabbing; }
  aside { width:320px; flex:none; background:var(--panel);
    border-left:1px solid var(--line); padding:16px; overflow-y:auto; }
  h1 { font-size:15px; margin:0 0 2px; letter-spacing:-.01em; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .box { border-top:1px solid var(--line); padding-top:12px; margin-top:12px; }
  .lbl { text-transform:uppercase; letter-spacing:.07em; font-size:10px;
    color:var(--dim); margin-bottom:7px; }
  input { width:100%; background:#0d1117; border:1px solid var(--line);
    color:var(--fg); border-radius:6px; padding:6px 9px; font-size:12px; }
  input:focus { outline:none; border-color:#58a6ff; }
  .kinds { display:flex; flex-wrap:wrap; gap:5px; }
  .kind { display:flex; align-items:center; gap:5px; padding:3px 8px;
    border:1px solid var(--line); border-radius:99px; cursor:pointer;
    font-size:11px; user-select:none; }
  .kind.off { opacity:.32; }
  .dot { width:8px; height:8px; border-radius:50%; flex:none; }
  .stat { display:flex; justify-content:space-between; padding:2px 0;
    font-variant-numeric:tabular-nums; }
  .stat span:last-child { color:var(--dim); }
  #sel .name { font-size:15px; font-weight:600; word-break:break-word;
    margin-bottom:3px; }
  .edge { padding:5px 0; border-bottom:1px solid #21262d; font-size:12px; }
  .edge .p { color:#58a6ff; }
  .edge .meta { color:var(--dim); font-size:10.5px; }
  .hint { color:var(--dim); font-size:11.5px; }
  button { background:#21262d; border:1px solid var(--line); color:var(--fg);
    border-radius:6px; padding:5px 10px; font-size:11.5px; cursor:pointer; }
  button:hover { background:#30363d; }
</style>
<div id="wrap">
  <canvas id="cv"></canvas>
  <aside>
    <h1>claude-memory-graph</h1>
    <div class="sub">__SUBTITLE__</div>

    <div class="lbl">Search</div>
    <input id="q" placeholder="entity name…" autocomplete="off">

    <div class="box">
      <div class="lbl">Entity kinds</div>
      <div class="kinds" id="kinds"></div>
    </div>

    <div class="box">
      <div class="lbl">Edges</div>
      <div class="stat"><span>— solid</span><span>LLM-extracted</span></div>
      <div class="stat"><span>·· dashed</span><span>structural (tool calls)</span></div>
    </div>

    <div class="box" id="sel">
      <div class="lbl">Selection</div>
      <div class="hint">Click a node. Drag to pan, scroll to zoom.</div>
    </div>
    <div class="box"><button id="reset">Reset view</button></div>
  </aside>
</div>
<script>
const NODES = __NODES__, LINKS = __LINKS__, COLORS = __COLORS__;

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let W = 0, H = 0, dpr = Math.min(devicePixelRatio || 1, 2);
function resize() {
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resize).observe(cv);
resize();

// ---- layout: plain O(n^2) force sim. At a few hundred nodes a quadtree
// would be more code than it saves.
const N = NODES.length;
const adj = NODES.map(() => []);
LINKS.forEach((l, i) => { adj[l.s].push(i); adj[l.t].push(i); });
NODES.forEach((n, i) => {
  const a = (i / N) * Math.PI * 2, r = 120 + Math.random() * 220;
  n.x = Math.cos(a) * r; n.y = Math.sin(a) * r; n.vx = 0; n.vy = 0;
  n.r = 3.2 + Math.sqrt(n.deg) * 1.7;
});
let alpha = 1;
function tick() {
  if (alpha < 0.002) return;
  for (let i = 0; i < N; i++) {
    const a = NODES[i];
    for (let j = i + 1; j < N; j++) {
      const b = NODES[j];
      let dx = b.x - a.x, dy = b.y - a.y;
      let d2 = dx * dx + dy * dy || 0.01;
      if (d2 > 250000) continue;                // ignore far pairs (>500px)
      const f = 900 / d2, d = Math.sqrt(d2);
      const fx = (dx / d) * f, fy = (dy / d) * f;
      a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
    }
    // gravity must outrun repulsion or flung-out nodes never return before
    // alpha cools -- at 0.0022 the layout stretched to 9864px on one axis
    a.vx -= a.x * 0.022; a.vy -= a.y * 0.022;
  }
  for (const l of LINKS) {
    const a = NODES[l.s], b = NODES[l.t];
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
    const f = (d - 70) * 0.012;
    const fx = (dx / d) * f, fy = (dy / d) * f;
    a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
  }
  for (const n of NODES) {
    n.x += (n.vx *= 0.82) * alpha; n.y += (n.vy *= 0.82) * alpha;
  }
  alpha *= 0.994;
}

// ---- view
let zoom = 1, panX = 0, panY = 0, sel = null, hover = null, query = '';
const off = new Set();
const toScreen = n => [n.x * zoom + panX + W / 2, n.y * zoom + panY + H / 2];
const visible = n => !off.has(n.kind);
const matches = n => query && n.name.toLowerCase().includes(query);

function draw() {
  tick();
  ctx.clearRect(0, 0, W, H);

  for (const l of LINKS) {
    const a = NODES[l.s], b = NODES[l.t];
    if (!visible(a) || !visible(b)) continue;
    const near = sel !== null && (l.s === sel || l.t === sel);
    const [ax, ay] = toScreen(a), [bx, by] = toScreen(b);
    ctx.beginPath();
    ctx.setLineDash(l.src === 'structural' ? [2, 3] : []);
    ctx.strokeStyle = near ? '#58a6ff' : 'rgba(139,148,158,.20)';
    ctx.lineWidth = near ? 1.6 : 0.7;
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  ctx.setLineDash([]);

  for (let i = 0; i < N; i++) {
    const n = NODES[i];
    if (!visible(n)) continue;
    const [x, y] = toScreen(n);
    const r = n.r * Math.max(zoom, 0.45);
    const on = i === sel || i === hover || matches(n);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832);
    ctx.fillStyle = COLORS[n.kind] || '#8b949e';
    ctx.globalAlpha = query && !matches(n) && i !== sel ? 0.22 : 1;
    ctx.fill();
    if (on) { ctx.lineWidth = 2; ctx.strokeStyle = '#e6edf3'; ctx.stroke(); }
    if (zoom > 0.75 && (n.deg > 3 || on)) {
      ctx.fillStyle = on ? '#e6edf3' : 'rgba(230,237,243,.62)';
      ctx.font = (on ? '600 ' : '') + '11px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText(n.name.slice(0, 26), x + r + 4, y + 3.5);
    }
    ctx.globalAlpha = 1;
  }
  requestAnimationFrame(draw);
}
draw();

function pick(mx, my) {
  let best = null, bd = 16 * 16;
  for (let i = 0; i < N; i++) {
    const n = NODES[i];
    if (!visible(n)) continue;
    const [x, y] = toScreen(n);
    const d = (x - mx) ** 2 + (y - my) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

let dragging = false, lastX = 0, lastY = 0, moved = 0;
cv.addEventListener('mousedown', e => {
  dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
  cv.classList.add('drag');
});
addEventListener('mouseup', e => {
  cv.classList.remove('drag');
  if (dragging && moved < 5) { sel = pick(e.clientX, e.clientY); showSel(); }
  dragging = false;
});
cv.addEventListener('mousemove', e => {
  if (dragging) {
    moved += Math.abs(e.movementX) + Math.abs(e.movementY);
    panX += e.movementX; panY += e.movementY;
  } else hover = pick(e.clientX, e.clientY);
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const mx = e.clientX - W / 2 - panX, my = e.clientY - H / 2 - panY;
  panX -= mx * (k - 1); panY -= my * (k - 1);
  zoom = Math.max(0.15, Math.min(6, zoom * k));
}, { passive: false });

function showSel() {
  const box = document.getElementById('sel');
  if (sel === null) {
    box.innerHTML = '<div class="lbl">Selection</div>' +
      '<div class="hint">Click a node. Drag to pan, scroll to zoom.</div>';
    return;
  }
  const n = NODES[sel];
  const rows = adj[sel].map(i => LINKS[i]).map(l => {
    const out = l.s === sel;
    const other = NODES[out ? l.t : l.s];
    return `<div class="edge">${out ? '' : other.name + ' '}` +
           `<span class="p">${l.p}</span>${out ? ' ' + other.name : ''}` +
           `<div class="meta">${l.src}${l.n > 1 ? ' · ' + l.n + ' chunks' : ''}</div></div>`;
  }).join('') || '<div class="hint">no edges within this view</div>';
  box.innerHTML = `<div class="lbl">Selection</div>
    <div class="name">${n.name}</div>
    <div class="stat"><span>${n.kind}</span><span>degree ${n.deg}</span></div>
    <div style="margin-top:10px">${rows}</div>`;
}

const kinds = [...new Set(NODES.map(n => n.kind))].sort();
document.getElementById('kinds').innerHTML = kinds.map(k =>
  `<div class="kind" data-k="${k}"><span class="dot" style="background:${COLORS[k]}"></span>${k}</div>`
).join('');
document.getElementById('kinds').onclick = e => {
  const el = e.target.closest('.kind'); if (!el) return;
  const k = el.dataset.k;
  off.has(k) ? off.delete(k) : off.add(k);
  el.classList.toggle('off', off.has(k));
  alpha = Math.max(alpha, 0.25);
};
document.getElementById('q').oninput = e => { query = e.target.value.toLowerCase().trim(); };
document.getElementById('reset').onclick = () => {
  zoom = 1; panX = panY = 0; sel = null; alpha = 1; showSel();
};
</script>
"""


def render(nodes, links, subtitle, out):
    html = (HTML
            .replace("__NODES__", json.dumps(nodes))
            .replace("__LINKS__", json.dumps(links))
            .replace("__COLORS__", json.dumps(KIND_COLORS))
            .replace("__SUBTITLE__", subtitle))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--project")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "graph.html"))
    args = ap.parse_args()

    db = graph.connect(args.db)
    nodes, links = collect(db, args.nodes, args.project, args.anonymize)
    if not nodes:
        raise SystemExit("no nodes matched -- check --project")

    total_e = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    total_t = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    sub = (f"{len(nodes)} of {total_e} entities · {len(links)} of {total_t} edges"
           f"{' · ' + args.project if args.project else ''}"
           f"{' · anonymized' if args.anonymize else ''}")
    path = render(nodes, links, sub, args.out)
    print(f"{sub}\nwrote {path}")
