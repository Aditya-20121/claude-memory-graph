"""Render the knowledge graph as a self-contained interactive 3D page.

Usage:
    python visualize.py                          # full names -> graph.html (LOCAL ONLY)
    python visualize.py --publish                # safe names -> docs/index.html
    python visualize.py --publish --list-names   # show what --publish would reveal
    python visualize.py --nodes 400

Three name modes:
    default     every entity name verbatim. Keeps file names, project internals
                and research terms. Never publish this.
    --publish   only names on PUBLIC_NAMES survive; everything else becomes
                "<kind> N". Safe for a public page.
    --anonymize every name replaced. Structure only.

The page embeds its own data and needs no CDN, no server and no network.
"""
import argparse
import json
import os
import re

import graph

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/Aditya-20121/claude-memory-graph"

KIND_COLORS = {
    "technology": "#4c9aff", "concept": "#a78bfa", "artifact": "#34d399",
    "project": "#fbbf24", "problem": "#f87171", "decision": "#f472b6",
    "person": "#94a3b8",
}

# Widely known public tools, models and standards. Anything not here is masked
# under --publish: an allowlist fails closed, a blocklist fails open.
PUBLIC_NAMES = {
    "python", "pytorch", "opencv", "numpy", "pandas", "scipy", "sklearn",
    "huggingface", "transformers", "accelerate", "peft", "unsloth", "lora",
    "cuda", "jupyter", "kaggle", "colab", "tensorboard", "wandb",
    "qwen3-vl", "qwen3 vl", "qwen3-vl-2b", "qwen3 vl 2b", "qwen3-14b",
    "qwen3 14b", "clip", "whisper", "yolo", "sam",
    "gemini", "chatgpt", "claude", "claude code", "openai", "anthropic", "mcp",
    "react", "next.js", "vue", "svelte", "typescript", "javascript", "node",
    "npm", "pnpm", "yarn", "vite", "webpack", "tailwind", "css", "html",
    "remotion", "ffmpeg", "ffprobe", "manim", "pil", "three.js", "d3",
    "fastapi", "flask", "django", "sqlite", "postgres", "redis", "docker",
    "git", "github", "gh", "gh cli", "pip", "bash", "powershell", "linux",
    "windows", "macos", "vscode", "chrome", "firefox", "playwright", "puppeteer",
    "wcag", "material design", "google fonts", "figma", "json", "sql", "regex",
    "rag", "embedding", "embeddings", "knowledge graph", "vector search",
    "cosine similarity", "accessibility", "typography", "design system",
}


def public_name(name):
    n = " ".join(re.sub(r"[_\-/]+", " ", name.lower()).split())
    return n in PUBLIC_NAMES or name.lower() in PUBLIC_NAMES


def collect(db, limit, project=None, mode="private"):
    where = "WHERE t.project LIKE ?" if project else ""
    args = [f"%{project}%"] if project else []
    top = db.execute(f"""
        SELECT e.id, e.name, e.kind, COUNT(*) AS deg
        FROM entities e JOIN triples t ON (t.subj = e.id OR t.obj = e.id)
        {where} GROUP BY e.id ORDER BY deg DESC LIMIT ?""",
        (*args, limit)).fetchall()
    if not top:
        return [], [], []

    idx = {r["id"]: i for i, r in enumerate(top)}
    ids = ",".join(str(i) for i in idx)
    edges = db.execute(f"""
        SELECT t.subj, t.obj, t.pred, t.src, COUNT(DISTINCT t.chunk_id) AS n
        FROM triples t WHERE t.subj IN ({ids}) AND t.obj IN ({ids})
        GROUP BY t.subj, t.obj, t.pred""").fetchall()

    counter, nodes, kept = {}, [], []
    for r in top:
        if mode == "private" or (mode == "publish" and public_name(r["name"])):
            label = r["name"]
            if mode == "publish":
                kept.append(r["name"])
            masked = 0
        else:
            counter[r["kind"]] = counter.get(r["kind"], 0) + 1
            label = f"{r['kind']} {counter[r['kind']]}"
            masked = 1
        nodes.append({"n": label, "k": r["kind"], "d": r["deg"], "m": masked})
    links = [{"s": idx[e["subj"]], "t": idx[e["obj"]], "p": e["pred"],
              "x": e["src"], "c": e["n"]} for e in edges]
    return nodes, links, kept


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>claude-memory-graph</title>
<meta name="description" content="A knowledge graph built from my own Claude Code history. Routing between graph and vector retrieval beats fusing them.">
<style>
  :root{--bg:#090c12;--panel:#11161f;--line:#232b38;--fg:#e8eef6;--dim:#8b98ab;--acc:#58a6ff}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--fg);
    font:13px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-font-smoothing:antialiased}
  #app{display:flex;height:100%}
  #stage{flex:1 1 0;min-width:0;position:relative}
  canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
  canvas.drag{cursor:grabbing}
  aside{width:340px;flex:none;background:var(--panel);border-left:1px solid var(--line);
    display:flex;flex-direction:column;overflow:hidden}
  .scroll{overflow-y:auto;padding:20px}
  h1{font-size:16px;letter-spacing:-.02em;font-weight:650}
  h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
    font-weight:600;margin-bottom:9px}
  .sub{color:var(--dim);font-size:12px;margin-top:3px}
  .sec{border-top:1px solid var(--line);margin-top:18px;padding-top:16px}
  input{width:100%;background:#090c12;border:1px solid var(--line);color:var(--fg);
    border-radius:7px;padding:7px 10px;font-size:12px;font-family:inherit}
  input:focus{outline:none;border-color:var(--acc)}
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{display:flex;align-items:center;gap:6px;padding:4px 9px;border:1px solid var(--line);
    border-radius:99px;cursor:pointer;font-size:11px;user-select:none;transition:opacity .12s}
  .chip.off{opacity:.3}
  .dot{width:8px;height:8px;border-radius:50%;flex:none}
  .row{display:flex;justify-content:space-between;padding:3px 0;font-variant-numeric:tabular-nums}
  .row span:last-child{color:var(--dim)}
  .name{font-size:16px;font-weight:650;word-break:break-word;letter-spacing:-.01em}
  .edge{padding:6px 0;border-bottom:1px solid #1a212c;font-size:12px}
  .edge em{color:var(--acc);font-style:normal}
  .edge small{color:var(--dim);font-size:10.5px;display:block}
  .hint{color:var(--dim);font-size:11.5px}
  button{background:#1a212c;border:1px solid var(--line);color:var(--fg);border-radius:7px;
    padding:6px 11px;font-size:11.5px;cursor:pointer;font-family:inherit}
  button:hover{background:#232b38}
  a{color:var(--acc);text-decoration:none}
  a:hover{text-decoration:underline}
  #about{position:absolute;inset:0;background:rgba(9,12,18,.97);overflow-y:auto;
    padding:40px;display:none;z-index:10}
  #about.on{display:block}
  .doc{max-width:660px;margin:0 auto}
  .doc h3{font-size:22px;letter-spacing:-.02em;margin-bottom:6px}
  .doc p{margin:12px 0;color:#c3cede}
  .doc ol{margin:12px 0 12px 20px}
  .doc li{margin:7px 0;color:#c3cede}
  .doc code{background:#11161f;border:1px solid var(--line);border-radius:4px;
    padding:1px 5px;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .doc table{width:100%;border-collapse:collapse;margin:14px 0;font-size:12.5px;
    font-variant-numeric:tabular-nums}
  .doc th,.doc td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right}
  .doc th:first-child,.doc td:first-child{text-align:left}
  .doc th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;
    letter-spacing:.05em}
  .win{background:rgba(88,166,255,.10)}
  .kicker{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.09em}
  .close{position:absolute;top:22px;right:26px}
  .note{border-left:2px solid var(--line);padding-left:14px;color:var(--dim);font-size:12.5px}
  @media(max-width:860px){aside{width:min(88vw,340px);position:absolute;right:0;top:0;bottom:0}
    #about{padding:24px 18px}}
</style></head><body>
<div id="app">
  <div id="stage"><canvas id="cv"></canvas>
    <div id="about"><button class="close" id="x">close</button><div class="doc">__DOC__</div></div>
  </div>
  <aside><div class="scroll">
    <h1>claude-memory-graph</h1>
    <div class="sub">__SUBTITLE__</div>
    <div style="margin-top:12px;display:flex;gap:8px">
      <button id="read">What is this?</button>
      <button id="spin">Pause spin</button>
    </div>
    <div class="sec"><h2>Search</h2><input id="q" placeholder="entity name&hellip;" autocomplete="off"></div>
    <div class="sec"><h2>Entity kinds</h2><div class="chips" id="kinds"></div></div>
    <div class="sec"><h2>Edges</h2>
      <div class="row"><span>solid</span><span>LLM-extracted</span></div>
      <div class="row"><span>dashed</span><span>from tool calls</span></div>
      <div class="hint" style="margin-top:7px">Tool-call edges cannot be hallucinated &mdash; no model produced them.</div>
    </div>
    <div class="sec" id="sel"><h2>Selection</h2>
      <div class="hint">Click a node. Drag to orbit, scroll to zoom.</div></div>
    <div class="sec"><a href="__REPO__" target="_blank" rel="noopener">Source on GitHub &rarr;</a></div>
  </div></aside>
</div>
<script>
const ND=__NODES__, LK=__LINKS__, CO=__COLORS__;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let W=0,H=0,dpr=Math.min(devicePixelRatio||1,2);
function resize(){const s=document.getElementById('stage');
  W=s.clientWidth;H=s.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);}
new ResizeObserver(resize).observe(document.getElementById('stage'));resize();

// ---- 3D force layout. O(n^2) at a few hundred nodes; a BVH would be more code
// than it saves.
const N=ND.length, adj=ND.map(()=>[]);
LK.forEach((l,i)=>{adj[l.s].push(i);adj[l.t].push(i);});
ND.forEach((n,i)=>{ // golden-spiral sphere: an even start beats random clumping
  const y=N>1?1-(i/(N-1))*2:0, r=Math.sqrt(Math.max(0,1-y*y)), th=i*2.39996, R=260;
  n.x=Math.cos(th)*r*R; n.y=y*R; n.z=Math.sin(th)*r*R;
  n.vx=n.vy=n.vz=0; n.r=2.6+Math.sqrt(n.d)*1.6;});
let alpha=1;
function tick(){
  if(alpha<0.002)return;
  for(let i=0;i<N;i++){const a=ND[i];
    for(let j=i+1;j<N;j++){const b=ND[j];
      let dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z,d2=dx*dx+dy*dy+dz*dz||.01;
      if(d2>490000)continue;
      const f=5200/d2,d=Math.sqrt(d2),fx=dx/d*f,fy=dy/d*f,fz=dz/d*f;
      a.vx-=fx;a.vy-=fy;a.vz-=fz;b.vx+=fx;b.vy+=fy;b.vz+=fz;}
    a.vx-=a.x*.009;a.vy-=a.y*.009;a.vz-=a.z*.009;}
  for(const l of LK){const a=ND[l.s],b=ND[l.t];
    const dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
    const d=Math.sqrt(dx*dx+dy*dy+dz*dz)||.01,f=(d-150)*.008;
    const fx=dx/d*f,fy=dy/d*f,fz=dz/d*f;
    a.vx+=fx;a.vy+=fy;a.vz+=fz;b.vx-=fx;b.vy-=fy;b.vz-=fz;}
  for(const n of ND){n.x+=(n.vx*=.82)*alpha;n.y+=(n.vy*=.82)*alpha;n.z+=(n.vz*=.82)*alpha;}
  alpha*=.994;}
for(let i=0;i<700;i++)tick();   // settle before first paint

// ---- camera
let yaw=.5,pitch=-.25,dist=1,spin=true,sel=null,hover=null,q='';
const off=new Set();
let cx=0,cy=0,cz=0,span=1;
(function centre(){const xs=ND.map(n=>n.x),ys=ND.map(n=>n.y),zs=ND.map(n=>n.z);
  cx=(Math.max(...xs)+Math.min(...xs))/2;cy=(Math.max(...ys)+Math.min(...ys))/2;
  cz=(Math.max(...zs)+Math.min(...zs))/2;
  span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),
                Math.max(...zs)-Math.min(...zs))||1;})();

function project(n){
  const x=n.x-cx,y=n.y-cy,z=n.z-cz;
  const cyw=Math.cos(yaw),syw=Math.sin(yaw);
  let X=x*cyw-z*syw, Z=x*syw+z*cyw;
  const cp=Math.cos(pitch),sp=Math.sin(pitch);
  let Y=y*cp-Z*sp; Z=y*sp+Z*cp;
  const fov=Math.min(W,H)*1.15, cam=span*1.9*dist;   // dist moves the camera only
  const k=fov/(cam+Z);
  return [W/2+X*k, H/2+Y*k, Z, k];}
const vis=n=>!off.has(n.k), hit=n=>q&&n.n.toLowerCase().includes(q);

function fitView(){           // perspective: measure, correct, repeat
  if(!W||!H)return;
  for(let pass=0;pass<12;pass++){
    let minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;
    for(const n of ND){const p=project(n);if(p[3]<=0)continue;
      if(p[0]<minX)minX=p[0];if(p[0]>maxX)maxX=p[0];
      if(p[1]<minY)minY=p[1];if(p[1]>maxY)maxY=p[1];}
    const w=maxX-minX,h=maxY-minY;
    if(!(w>0&&h>0))return;
    const want=Math.min((W-110)/w,(H-110)/h);
    if(Math.abs(want-1)<0.02)return;
    dist=Math.max(.15,Math.min(6,dist/want));
  }
}

let drag=false,moved=0;
let fitted=false;
function draw(){
  tick();
  if(!fitted&&W>0){fitView();fitted=true;}
  if(spin&&!drag)yaw+=.0016;
  ctx.clearRect(0,0,W,H);
  const P=ND.map(project);

  for(const l of LK){const a=ND[l.s],b=ND[l.t];
    if(!vis(a)||!vis(b))continue;
    const A=P[l.s],B=P[l.t];
    if(A[3]<=0||B[3]<=0)continue;
    const near=sel!==null&&(l.s===sel||l.t===sel);
    ctx.beginPath();ctx.setLineDash(l.x==='structural'?[2,3]:[]);
    const depth=Math.max(.06,Math.min(.5,(A[3]+B[3])));
    ctx.strokeStyle=near?'rgba(88,166,255,.85)':'rgba(139,152,171,'+(depth*.42)+')';
    ctx.lineWidth=near?1.7:.65;
    ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();}
  ctx.setLineDash([]);

  const order=ND.map((n,i)=>i).filter(i=>vis(ND[i])&&P[i][3]>0)
                .sort((a,b)=>P[b][2]-P[a][2]);   // painter's algorithm: far first
  const labels=[];
  for(const i of order){const n=ND[i],x=P[i][0],y=P[i][1],k=P[i][3];
    const r=Math.max(1.2,n.r*k*1.5);
    const on=i===sel||i===hover||hit(n);
    const fade=Math.max(.3,Math.min(1,k*2.4));
    ctx.globalAlpha=(q&&!hit(n)&&!on)?.12:fade;
    ctx.beginPath();ctx.arc(x,y,r,0,6.2832);
    ctx.fillStyle=CO[n.k]||'#8b98ab';ctx.fill();
    if(on){ctx.globalAlpha=1;ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
    ctx.globalAlpha=1;
    if(on||(!n.m&&n.d>=2))labels.push({n:n,x:x,y:y,r:r,on:on,k:k});}

  labels.sort(function(a,b){return (b.on-a.on)||(b.k-a.k);});
  const placed=[];
  for(const L of labels){
    if(q&&!hit(L.n)&&!L.on)continue;
    const t=L.n.n.length>26?L.n.n.slice(0,25)+'…':L.n.n;
    ctx.font=(L.on?'600 ':'')+'11px ui-sans-serif,system-ui,sans-serif';
    const w=ctx.measureText(t).width,bx=L.x+L.r+5,by=L.y-6;
    if(bx>W||bx+w<0||by>H||by+13<0)continue;
    let clash=false;
    for(const p of placed)
      if(bx<p.x+p.w+4&&bx+w+4>p.x&&by<p.y+p.h+3&&by+16>p.y){clash=true;break;}
    if(clash&&!L.on)continue;
    placed.push({x:bx,y:by,w:w,h:13});
    ctx.globalAlpha=L.on?1:Math.max(.45,Math.min(1,L.k*2.6));
    ctx.fillStyle=L.on?'#fff':'rgba(232,238,246,.8)';
    ctx.fillText(t,bx,L.y+3.5);ctx.globalAlpha=1;}
  requestAnimationFrame(draw);}
draw();

function pick(mx,my){let best=null,bd=18*18;
  const rect=cv.getBoundingClientRect();mx-=rect.left;my-=rect.top;
  for(let i=0;i<N;i++){const n=ND[i];if(!vis(n))continue;
    const p=project(n);if(p[3]<=0)continue;
    const d=(p[0]-mx)*(p[0]-mx)+(p[1]-my)*(p[1]-my);
    if(d<bd){bd=d;best=i;}}
  return best;}

cv.addEventListener('pointerdown',function(e){drag=true;moved=0;
  cv.setPointerCapture(e.pointerId);cv.classList.add('drag');});
cv.addEventListener('pointerup',function(e){cv.classList.remove('drag');
  if(drag&&moved<6){sel=pick(e.clientX,e.clientY);showSel();}drag=false;});
cv.addEventListener('pointermove',function(e){
  if(drag){moved+=Math.abs(e.movementX)+Math.abs(e.movementY);
    yaw+=e.movementX*.006;
    pitch=Math.max(-1.45,Math.min(1.45,pitch+e.movementY*.006));}
  else hover=pick(e.clientX,e.clientY);});
cv.addEventListener('wheel',function(e){e.preventDefault();
  dist=Math.max(.15,Math.min(6,dist*(e.deltaY<0?1/1.1:1.1)));},{passive:false});

function showSel(){const b=document.getElementById('sel');
  if(sel===null){b.innerHTML='<h2>Selection</h2><div class="hint">Click a node. '+
    'Drag to orbit, scroll to zoom.</div>';return;}
  const n=ND[sel];
  const rows=adj[sel].map(function(i){return LK[i];}).map(function(l){
    const out=l.s===sel,o=ND[out?l.t:l.s];
    return '<div class="edge">'+(out?'':o.n+' ')+'<em>'+l.p+'</em>'+(out?' '+o.n:'')+
      '<small>'+l.x+(l.c>1?' · '+l.c+' chunks':'')+'</small></div>';}).join('')
    ||'<div class="hint">no edges inside this view</div>';
  b.innerHTML='<h2>Selection</h2><div class="name">'+n.n+'</div>'+
    '<div class="row"><span>'+n.k+'</span><span>degree '+n.d+'</span></div>'+
    (n.m?'<div class="hint" style="margin-top:6px">name withheld — private corpus</div>':'')+
    '<div style="margin-top:11px">'+rows+'</div>';}

const kinds=Array.from(new Set(ND.map(function(n){return n.k;}))).sort();
document.getElementById('kinds').innerHTML=kinds.map(function(k){
  return '<div class="chip" data-k="'+k+'"><span class="dot" style="background:'+
    CO[k]+'"></span>'+k+'</div>';}).join('');
document.getElementById('kinds').onclick=function(e){
  const el=e.target.closest('.chip');if(!el)return;
  const k=el.dataset.k;
  if(off.has(k))off.delete(k);else off.add(k);
  el.classList.toggle('off',off.has(k));};
document.getElementById('q').oninput=function(e){q=e.target.value.toLowerCase().trim();};
const ab=document.getElementById('about');
document.getElementById('read').onclick=function(){ab.classList.add('on');};
document.getElementById('x').onclick=function(){ab.classList.remove('on');};
addEventListener('keydown',function(e){if(e.key==='Escape')ab.classList.remove('on');});
const sp=document.getElementById('spin');
sp.onclick=function(){spin=!spin;sp.textContent=spin?'Pause spin':'Resume spin';};
</script></body></html>"""

DOC = """
<div class="kicker">Personal project</div>
<h3>claude-memory-graph</h3>
<p><strong>The problem.</strong> Claude has no memory between sessions. Months of my
own engineering conversations sit in local transcript files, and nothing can
query them.</p>

<p><strong>What you are looking at.</strong> Every dot is an entity pulled out of my own
Claude Code history &mdash; a tool, a file, a concept, a problem, a decision. Every
line is a relationship between two of them. Roughly four months of work, as a graph.</p>

<ol>
  <li><strong>Ingest</strong> &mdash; parse local transcripts into a normalised store.
      One chunk = one exchange. 25 sessions &rarr; 411 chunks.</li>
  <li><strong>Knowledge graph</strong> &mdash; an LLM extracts
      <code>(entity, relation, entity)</code> triples against a <em>closed</em>
      vocabulary of 7 entity kinds and 11 predicates; anything outside it is
      dropped rather than stored. Further edges come straight from tool calls, so
      they cannot be hallucinated. 2,579 entities, 2,895 relations.</li>
  <li><strong>Vector index</strong> &mdash; embed every chunk; cosine search in pure
      Python, no vector database.</li>
  <li><strong>MCP server</strong> &mdash; three tools, so Claude can query all of it
      mid-conversation.</li>
</ol>

<p><strong>The result was not the one I set out to prove.</strong> The plan was to show
that hybrid retrieval &mdash; fusing graph and vector search &mdash; beats plain
vector search. I built two evaluation sets with deliberately <em>opposite</em>
biases so neither could flatter the architecture, and measured it.</p>

<p>Hybrid lost to both of its own inputs.</p>

<table>
  <tr><th>mode</th><th>single-hop</th><th>cross-session</th><th>overall</th></tr>
  <tr><td>vector only</td><td>0.576</td><td>0.074</td><td>0.353</td></tr>
  <tr><td>graph only</td><td>0.191</td><td>0.420</td><td>0.293</td></tr>
  <tr><td>hybrid (fixed fusion)</td><td>0.349</td><td>0.234</td><td>0.298</td></tr>
  <tr class="win"><td><strong>adaptive routing</strong></td><td>0.538</td><td>0.313</td>
      <td><strong>0.438</strong></td></tr>
</table>
<p class="hint" style="margin-top:-6px">Mean reciprocal rank over 45 questions.
Higher is better.</p>

<p>The two retrievers turned out to be <em>complementary rather than additive</em>:
vector search is ~8&times; better at &ldquo;what did this conversation say&rdquo;,
the graph ~5&times; better at &ldquo;where else did this come up&rdquo;. Blending
them at a fixed ratio is worse than picking one.</p>

<p>So I replaced fusion with <strong>routing</strong>: let vector search's own
confidence decide how much graph to mix in. When nothing in the corpus closely
matches the query, the answer is more likely <em>reachable</em> than
<em>stateable</em> &mdash; so lean on the graph. That is <strong>+24%
overall</strong> against the best fixed strategy, and it matches the graph's
cross-session recall while keeping the vector's.</p>

<p class="note">Reported honestly: the routing thresholds are tuned on the same 45
questions they are scored on, and 45 questions from one person's history is a
small sample. The README documents this and the other caveats rather than
burying them.</p>

<p><strong>Built with</strong> the Python standard library &mdash; SQLite for both the
graph and the vectors. No ORM, no vector database, no graph database. The MCP SDK
is the only third-party dependency.</p>

<p style="margin-top:22px"><a href="__REPO__" target="_blank" rel="noopener">
Source and full write-up on GitHub &rarr;</a></p>
<p class="hint">Entity names here are limited to well-known public tools;
everything drawn from private conversations is masked.</p>
"""


def render(nodes, links, subtitle, out):
    html = (PAGE
            .replace("__NODES__", json.dumps(nodes, separators=(",", ":")))
            .replace("__LINKS__", json.dumps(links, separators=(",", ":")))
            .replace("__COLORS__", json.dumps(KIND_COLORS))
            .replace("__SUBTITLE__", subtitle)
            .replace("__DOC__", DOC)
            .replace("__REPO__", REPO))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "memory.db"))
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--project")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--list-names", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    mode = "publish" if args.publish else "anon" if args.anonymize else "private"
    out = args.out or (os.path.join(HERE, "docs", "index.html")
                       if args.publish else os.path.join(HERE, "graph.html"))

    db = graph.connect(args.db)
    nodes, links, kept = collect(db, args.nodes, args.project, mode)
    if not nodes:
        raise SystemExit("no nodes matched")

    if args.list_names:
        print(f"{len(kept)} of {len(nodes)} names would be PUBLIC:\n")
        for i in range(0, len(kept), 4):
            print("  " + "  ".join(f"{k[:24]:<24}" for k in kept[i:i + 4]))
        print(f"\n{len(nodes) - len(kept)} masked as '<kind> N'")
        raise SystemExit

    total = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    tot_t = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    sub = (f"{len(nodes)} of {total} entities &middot; {len(links)} of {tot_t} relations"
           + (" &middot; names masked" if mode != "private" else ""))
    print(f"{sub}\nmode={mode}\nwrote {render(nodes, links, sub, out)}")
