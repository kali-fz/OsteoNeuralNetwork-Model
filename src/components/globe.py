"""Interactive orthographic globe showing ONNM's geographic reach.

Renders a D3-geo canvas globe embedded via streamlit.components.v1.html with:
  - Continuous auto-rotation at 6°/s (prefers-reduced-motion aware)
  - Click-and-drag spin with yaw+pitch, ±75° pitch clamp
  - Inertia on release decaying to rest
  - Auto-rotation resumes 3 s after last interaction
  - Pointer Events + setPointerCapture (mouse/touch/pen)
  - Keyboard arrow-key nudge when canvas has focus
  - Country-level markers with tooltips (country name + count)
  - Two distinct visual layers: signup and contributor
  - Initial view centred on the strongest country-level activity
  - Square-root scaled marker area
  - Visibility and IntersectionObserver pausing (zero CPU in background)
  - devicePixelRatio capped at 2

Optional detailed map assets can be installed into src/components/assets/ from
pinned CDN URLs. Rendering never downloads them: when they are absent the same
country markers are drawn on a dependency-free canvas globe immediately.

Privacy contract (section 3B of REDESIGN_BRIEF.md):
  - Markers carry only {lat, lng, label, count, layer}: no user id, no email,
    no timestamp, no sub-country coordinate.
  - Never call the browser Geolocation API.
  - Never embed any API key in the HTML (it is browser-readable).
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from pathlib import Path

# streamlit is imported lazily inside render_globe and _load_static_assets so
# that the module is importable in test environments without a running server.

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

# Pinned CDN versions — bump only these two constants to upgrade.
_ASSETS: dict[str, str] = {
    "d3-array.min.js": (
        "https://cdn.jsdelivr.net/npm/d3-array@3.2.4/dist/d3-array.min.js"
    ),
    "d3-geo.min.js": (
        "https://cdn.jsdelivr.net/npm/d3-geo@3.1.0/dist/d3-geo.min.js"
    ),
    "topojson-client.min.js": (
        "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"
    ),
    "countries-110m.json": (
        "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json"
    ),
}

# Sample markers used when the backend is unreachable or the migration has not
# run.  These are plausible illustrative counts, not real data.
SAMPLE_MARKERS: list[dict] = [
    {"lat": 51.5, "lng": -0.1, "label": "United Kingdom", "count": 14, "layer": "signup"},
    {"lat": 40.4, "lng": -3.7, "label": "Spain", "count": 9, "layer": "signup"},
    {"lat": 35.9, "lng": 104.2, "label": "China", "count": 18, "layer": "signup"},
    {"lat": 37.1, "lng": -95.7, "label": "United States", "count": 32, "layer": "signup"},
    {"lat": -14.2, "lng": -51.9, "label": "Brazil", "count": 11, "layer": "signup"},
    {"lat": 48.9, "lng": 2.3, "label": "France", "count": 7, "layer": "signup"},
    {"lat": 52.5, "lng": 13.4, "label": "Germany", "count": 8, "layer": "signup"},
    {"lat": 35.7, "lng": 139.7, "label": "Japan", "count": 6, "layer": "signup"},
    {"lat": 28.6, "lng": 77.2, "label": "India", "count": 15, "layer": "signup"},
    {"lat": 51.5, "lng": -0.1, "label": "United Kingdom", "count": 6, "layer": "contributor"},
    {"lat": 37.1, "lng": -95.7, "label": "United States", "count": 12, "layer": "contributor"},
    {"lat": 35.9, "lng": 104.2, "label": "China", "count": 7, "layer": "contributor"},
    {"lat": 28.6, "lng": 77.2, "label": "India", "count": 5, "layer": "contributor"},
]


def _ensure_assets() -> bool:
    """Download globe assets once; return True when all are available.

    Downloads happen once per server process.  After that the files are read
    from disk.  A timeout of 30 s is generous for ~200 KB total; if it fails
    the globe renders a fallback sphere with no land outlines.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in _ASSETS.items():
        path = ASSETS_DIR / name
        if path.exists():
            continue
        try:
            logger.info("One-time download of globe asset: %s", name)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ONNM-GlobeSetup/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                path.write_bytes(resp.read())
            logger.info("Saved %s (%d bytes)", name, path.stat().st_size)
        except Exception as exc:  # noqa: BLE001 — intentional, fail soft
            logger.warning("Could not download globe asset %s: %s", name, exc)
            ok = False
    return ok


def _load_static_assets() -> tuple[str, str, str] | None:
    """Load vendored JS + world JSON once per server startup into memory.

    Returns (d3_script, topojson_script, world_json) or None when unavailable.
    This function deliberately does not download anything. A landing-page
    decoration must not add network requests or a possible 90-second delay to
    an app rerun. ``_ensure_assets`` remains available as an explicit setup
    helper, while production renders use the dependency-free fallback below
    when the optional files have not been vendored.
    When running inside Streamlit, uses cache_resource so the ~200 KB read
    happens once regardless of how many visitors request the landing page.
    When running in a test environment without Streamlit, falls back to a plain
    call (assets are loaded on every test invocation, which is acceptable).
    """
    def _load():
        paths = [ASSETS_DIR / name for name in _ASSETS]
        if not all(path.is_file() for path in paths):
            return None
        try:
            # d3-geo's browser bundle expects d3-array on the shared d3 global.
            # Concatenate them in dependency order before embedding the script.
            d3_array = (ASSETS_DIR / "d3-array.min.js").read_text(encoding="utf-8")
            d3_geo = (ASSETS_DIR / "d3-geo.min.js").read_text(encoding="utf-8")
            d3 = f"{d3_array}\n{d3_geo}"
            topo = (ASSETS_DIR / "topojson-client.min.js").read_text(encoding="utf-8")
            world = (ASSETS_DIR / "countries-110m.json").read_text(encoding="utf-8")
            return d3, topo, world
        except OSError as exc:
            logger.warning("Globe asset read failed: %s", exc)
            return None

    try:
        import streamlit as st
        return st.cache_resource(show_spinner=False)(_load)()
    except ImportError:
        return _load()


def _normalise_markers(markers: list[dict]) -> list[dict]:
    """Return the small, privacy-safe marker contract accepted by the canvas.

    The community payload is already aggregated, but keeping this boundary
    strict prevents a future backend field (or malformed value) from being
    copied into browser-readable HTML by accident.
    """
    clean: list[dict] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        try:
            lat = float(marker.get("lat"))
            lng = float(marker.get("lng"))
            count = int(marker.get("count"))
        except (TypeError, ValueError, OverflowError):
            continue
        layer = str(marker.get("layer") or "")
        if (
            not math.isfinite(lat)
            or not math.isfinite(lng)
            or not -90 <= lat <= 90
            or not -180 <= lng <= 180
            or count <= 0
            or layer not in {"signup", "contributor"}
        ):
            continue
        clean.append(
            {
                "lat": round(lat, 3),
                "lng": round(lng, 3),
                "label": str(marker.get("label") or "Country")[:80],
                "count": count,
                "layer": layer,
            }
        )
    return clean


def _merge_country_markers(markers: list[dict]) -> list[dict]:
    """Collapse registration and contributor layers into one country marker."""
    countries: dict[str, dict] = {}
    for marker in markers:
        key = str(marker["label"]).casefold()
        country = countries.setdefault(
            key,
            {
                "lat": marker["lat"],
                "lng": marker["lng"],
                "label": marker["label"],
                "signupCount": 0,
                "contributorCount": 0,
            },
        )
        count_key = (
            "contributorCount"
            if marker["layer"] == "contributor"
            else "signupCount"
        )
        country[count_key] += marker["count"]
    return list(countries.values())


def _json_for_script(value: object) -> str:
    """Encode JSON without permitting an embedded ``</script>`` boundary."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _initial_focus(markers: list[dict]) -> dict | None:
    """Choose a populated country for the globe's first visible hemisphere.

    The largest marker wins. An approved-contributor marker wins an exact tie,
    which keeps the more meaningful activity visible without hiding a larger
    registered-user community.
    """
    if not markers:
        return None
    return max(
        markers,
        key=lambda marker: (
            int(marker.get("count") or 0),
            marker.get("layer") == "contributor",
        ),
    )


def _build_fallback_html(
    markers_json: str,
    auto_rotate: bool,
    height: int,
) -> str:
    """Build a self-contained canvas globe with no CDN or map dependency."""
    try:
        marker_values = json.loads(markers_json)
    except (TypeError, ValueError):
        marker_values = []
    clean_markers = _normalise_markers(
        marker_values if isinstance(marker_values, list) else []
    )
    markers_json = _json_for_script(clean_markers)
    display_markers_json = _json_for_script(_merge_country_markers(clean_markers))
    focus = _initial_focus(clean_markers)
    initial_yaw = -float(focus["lng"]) if focus else -15.0
    initial_pitch = float(focus["lat"]) if focus else -12.0
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html,body {{ width:100%; height:{height}px; overflow:hidden; background:transparent; }}
  #wrap {{ position:relative; width:100%; height:{height}px; display:flex;
           flex-direction:column; align-items:center; justify-content:center; }}
  canvas {{ display:block; max-width:100%; cursor:grab; touch-action:none; outline:none; }}
  canvas:active {{ cursor:grabbing; }}
  #tip {{ position:absolute; display:none; pointer-events:none; z-index:2;
          padding:6px 10px; border-radius:6px; color:#f8f5ee;
          background:rgba(30,38,31,.92); font:600 12px/1.3 Inter,Segoe UI,sans-serif;
          box-shadow:0 5px 18px rgba(22,31,25,.2); white-space:nowrap; }}
  #legend {{ display:flex; gap:16px; margin-top:8px; color:#6b6457;
             font:500 11px/1.3 Inter,Segoe UI,sans-serif; }}
  .item {{ display:flex; align-items:center; gap:6px; }}
  .dot {{ width:9px; height:9px; border-radius:50%; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="globe" tabindex="0" role="img"
    aria-label="Globe showing privacy-protected country-level community locations"></canvas>
  <div id="tip"></div>
  <div id="legend">
    <span class="item"><i class="dot" style="background:#e8a850"></i>Registered users</span>
    <span class="item"><i class="dot" style="background:#2e6b47"></i>Approved contributors</span>
  </div>
</div>
<script>
(() => {{
  'use strict';
  const markers = {markers_json};
  const displayMarkers = {display_markers_json};
  const canvas = document.getElementById('globe');
  const wrap = document.getElementById('wrap');
  const tip = document.getElementById('tip');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const shouldRotate = {str(auto_rotate).lower()} && !reduceMotion;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  let size=0, radius=0, cx=0, cy=0,
      yaw={initial_yaw}, pitch={initial_pitch};
  let dragging=false, lastX=0, lastY=0, visible=true, lastFrame=0;
  let projected=[];

  function resize() {{
    size = Math.max(180, Math.min(wrap.clientWidth, {height} - 38));
    radius = size * .47; cx = size / 2; cy = size / 2;
    canvas.style.width=size+'px'; canvas.style.height=size+'px';
    canvas.width=Math.round(size*dpr); canvas.height=Math.round(size*dpr);
    draw();
  }}
  function point(lat, lng) {{
    const phi=lat*Math.PI/180, lam=(lng+yaw)*Math.PI/180;
    const pp=pitch*Math.PI/180;
    const x=Math.cos(phi)*Math.sin(lam);
    const y0=Math.sin(phi), z0=Math.cos(phi)*Math.cos(lam);
    const y=y0*Math.cos(pp)-z0*Math.sin(pp);
    const z=y0*Math.sin(pp)+z0*Math.cos(pp);
    return {{x:cx+radius*x, y:cy-radius*y, z}};
  }}
  function grid(ctx) {{
    ctx.save(); ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2); ctx.clip();
    ctx.strokeStyle='rgba(255,255,255,.27)'; ctx.lineWidth=.75;
    for (let lat=-60; lat<=60; lat+=30) {{
      ctx.beginPath(); let open=false;
      for (let lng=-180; lng<=180; lng+=3) {{
        const p=point(lat,lng);
        if (p.z>=0) {{ open ? ctx.lineTo(p.x,p.y) : ctx.moveTo(p.x,p.y); open=true; }}
        else open=false;
      }} ctx.stroke();
    }}
    for (let lng=-150; lng<=180; lng+=30) {{
      ctx.beginPath(); let open=false;
      for (let lat=-90; lat<=90; lat+=3) {{
        const p=point(lat,lng);
        if (p.z>=0) {{ open ? ctx.lineTo(p.x,p.y) : ctx.moveTo(p.x,p.y); open=true; }}
        else open=false;
      }} ctx.stroke();
    }}
    ctx.restore();
  }}
  function draw() {{
    if (!size) return;
    const ctx=canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,size,size);
    const ocean=ctx.createRadialGradient(cx-radius*.32,cy-radius*.38,0,cx,cy,radius*1.08);
    ocean.addColorStop(0,'#e7f1f1');
    ocean.addColorStop(.58,'#b9d3d2');
    ocean.addColorStop(1,'#6f9896');
    ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2); ctx.fillStyle=ocean; ctx.fill();
    grid(ctx);
    ctx.beginPath(); ctx.arc(cx,cy,radius,0,Math.PI*2);
    ctx.strokeStyle='rgba(38,79,68,.28)'; ctx.lineWidth=1.5; ctx.stroke();
    projected=[];
    for (const m of displayMarkers) {{
        const p=point(m.lat,m.lng); if (p.z<=0) continue;
        const count = m.contributorCount || m.signupCount;
        const contributor = m.contributorCount > 0;
        const r=Math.min(13,3+Math.sqrt(count)*1.55);
        ctx.globalAlpha=Math.max(.28,Math.min(1,p.z*2.4));
        ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2);
        ctx.fillStyle=contributor?'#2e6b47':'#e8a850'; ctx.fill();
        ctx.strokeStyle=contributor?'#173b28':'#ad6c12'; ctx.lineWidth=1.2; ctx.stroke();
        projected.push({{...m,x:p.x,y:p.y,r}});
    }} ctx.globalAlpha=1;
  }}
  canvas.addEventListener('pointerdown', e => {{
    dragging=true; lastX=e.clientX; lastY=e.clientY;
    canvas.setPointerCapture(e.pointerId); tip.style.display='none';
  }});
  canvas.addEventListener('pointermove', e => {{
    if (dragging) {{
      yaw += (e.clientX-lastX)*.45; pitch=Math.max(-65,Math.min(65,pitch-(e.clientY-lastY)*.35));
      lastX=e.clientX; lastY=e.clientY; draw(); return;
    }}
    const rect=canvas.getBoundingClientRect(), x=e.clientX-rect.left, y=e.clientY-rect.top;
    let hit=null, distance=18;
    for (const p of projected) {{
      const d=Math.hypot(x-p.x,y-p.y);
      if (d<distance) {{hit=p;distance=d;}}
    }}
    if (!hit) {{ tip.style.display='none'; return; }}
    const parts=[];
    if (hit.signupCount) parts.push(
      hit.signupCount.toLocaleString()+' registered user'+(hit.signupCount===1?'':'s')
    );
    if (hit.contributorCount) parts.push(
      hit.contributorCount.toLocaleString()+' approved contributor'+
      (hit.contributorCount===1?'':'s')
    );
    tip.textContent=hit.label+' · '+parts.join(' · '); tip.style.display='block';
    tip.style.left=Math.min(wrap.clientWidth-tip.offsetWidth-6,e.clientX-rect.left+12)+'px';
    tip.style.top=Math.max(4,e.clientY-rect.top-tip.offsetHeight-8)+'px';
  }});
  const stop=()=>{{ dragging=false; }};
  canvas.addEventListener('pointerup',stop); canvas.addEventListener('pointercancel',stop);
  canvas.addEventListener('mouseleave',()=>{{ tip.style.display='none'; }});
  canvas.addEventListener('keydown',e=>{{
    if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) return;
    e.preventDefault(); yaw += e.key==='ArrowLeft'?5:e.key==='ArrowRight'?-5:0;
    pitch=Math.max(-65,Math.min(65,pitch+(e.key==='ArrowUp'?5:e.key==='ArrowDown'?-5:0))); draw();
  }});
  function tick(now) {{
    requestAnimationFrame(tick);
    if (!visible || dragging || !shouldRotate) {{ lastFrame=now; return; }}
    const elapsed=now-(lastFrame||now); if (elapsed<33) return;
    yaw += Math.min(80,elapsed)*.0028; lastFrame=now; draw();
  }}
  addEventListener('resize',resize);
  document.addEventListener('visibilitychange',()=>{{visible=!document.hidden;}});
  if (window.IntersectionObserver) {{
    new IntersectionObserver(
      e=>{{visible=e[0].isIntersecting;}},{{threshold:.05}}
    ).observe(canvas);
  }}
  resize(); requestAnimationFrame(tick);
}})();
</script>
</body>
</html>"""


def _build_html(
    d3_script: str,
    topojson_script: str,
    world_json: str,
    markers_json: str,
    auto_rotate: bool,
    height: int,
) -> str:
    try:
        marker_values = json.loads(markers_json)
    except (TypeError, ValueError):
        marker_values = []
    if not isinstance(marker_values, list):
        marker_values = []
    clean_markers = _normalise_markers(marker_values)
    markers_json = _json_for_script(clean_markers)
    display_markers_json = _json_for_script(_merge_country_markers(clean_markers))
    focus = _initial_focus(clean_markers)
    initial_rotation = (
        [-float(focus["lng"]), -float(focus["lat"]), 0]
        if focus
        else [0.0, -20.0, 0]
    )
    initial_rotation_json = _json_for_script(initial_rotation)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: {height}px; overflow: hidden;
                background: transparent; }}
  #globe-wrap {{
    position: relative; width: 100%; height: {height}px;
    display: flex; flex-direction: column; align-items: center;
  }}
  canvas {{
    touch-action: none; cursor: grab; outline: none;
    border-radius: 50%; display: block;
  }}
  canvas:active {{ cursor: grabbing; }}
  #tooltip {{
    position: absolute; pointer-events: none;
    background: rgba(28,26,23,.88); color: #f7f4ef;
    font-family: 'Inter','Segoe UI',Arial,sans-serif;
    font-size: 12px; font-weight: 600;
    padding: 5px 10px; border-radius: 4px;
    white-space: nowrap; display: none; z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,.28);
    letter-spacing: 0.02em;
  }}
  #legend {{
    display: flex; gap: 16px; margin-top: 10px;
    font-family: 'Inter','Segoe UI',Arial,sans-serif;
    font-size: 11px; color: #6b6457;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>
<div id="globe-wrap">
  <canvas id="globe" tabindex="0" role="img"
    aria-label="Interactive globe showing ONNM geographic reach"></canvas>
  <div id="tooltip"></div>
  <div id="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#e8a850;"></div>
      <span>Registered users</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#2e6b47;"></div>
      <span>Approved contributors</span>
    </div>
  </div>
</div>

<script>{d3_script}</script>
<script>{topojson_script}</script>
<script>
(function() {{
  'use strict';

  // ── Data ──────────────────────────────────────────────────────────────────
  const MARKERS = {markers_json};
  // A country may appear in both data layers. Render one marker for that
  // country, while retaining both counts for its tooltip.
  const DISPLAY_MARKERS = {display_markers_json};
  const WORLD   = {world_json};
  const AUTO_ROTATE = {str(auto_rotate).lower()};

  // ── Canvas setup ──────────────────────────────────────────────────────────
  const canvas = document.getElementById('globe');
  const tooltip = document.getElementById('tooltip');
  const wrap  = document.getElementById('globe-wrap');
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  function measureSize() {{
    const avail = Math.min(wrap.clientWidth, {height} - 44);
    return Math.max(160, avail);
  }}

  function applyCanvasSize(size) {{
    canvas.style.width  = size + 'px';
    canvas.style.height = size + 'px';
    canvas.width  = size * DPR;
    canvas.height = size * DPR;
  }}

  let SIZE = measureSize();
  applyCanvasSize(SIZE);
  const ctx = canvas.getContext('2d');
  ctx.scale(DPR, DPR);

  // ── Projection ────────────────────────────────────────────────────────────
  // d3geo is the UMD namespace from d3-geo.umd.min.js
  const d3geo = window.d3geo || window.d3;
  let projection, pathGen, graticule, sphere;

  function initProjection() {{
    projection = d3geo.geoOrthographic()
      .scale(SIZE / 2 - 4)
      .translate([SIZE / 2, SIZE / 2])
      .clipAngle(90)
      .precision(0.2)
      .rotate([ROT[0], ROT[1], ROT[2]]);
    // Bind the canvas path to the orthographic projection. Without this,
    // GeoJSON Sphere objects are streamed directly to the canvas context,
    // which has no sphere() method and crashes every animation frame.
    pathGen  = d3geo.geoPath(projection).context(ctx);
    graticule = d3geo.geoGraticule()();
    sphere    = {{ type: 'Sphere' }};
  }}

  // ── World geometry ────────────────────────────────────────────────────────
  let land = null, borders = null, countries = [];
  if (WORLD && typeof topojson !== 'undefined') {{
    try {{
      land    = topojson.feature(WORLD, WORLD.objects.land);
      countries = topojson.feature(WORLD, WORLD.objects.countries).features;
      // countries mesh for borders (no shared edges between countries)
      borders = topojson.mesh(WORLD, WORLD.objects.countries,
                              (a, b) => a !== b);
    }} catch(e) {{ console.warn('topojson parse error', e); }}
  }}

  // Resolve each aggregated country marker to its map polygon once. Keeping
  // this outside draw() avoids repeating geoContains work during animation.
  const activeCountries = countries.map(feature => {{
    const activity = MARKERS.filter(marker =>
      d3geo.geoContains(feature, [marker.lng, marker.lat])
    );
    if (!activity.length) return null;
    return {{
      feature,
      hasContributor: activity.some(marker => marker.layer === 'contributor'),
    }};
  }}).filter(Boolean);

  // ── Rotation state ────────────────────────────────────────────────────────
  let ROT    = {initial_rotation_json}; // starts on strongest activity
  let VEL    = [0, 0];        // [dLambda/ms, dPhi/ms]
  const INERTIA_DECAY = 0.94; // per frame at 60fps ≈ 0.94^60 ≈ 0.025 after 1 s

  // ── Drag state ────────────────────────────────────────────────────────────
  let dragging  = false;
  let lastPt    = null;
  let lastPtT   = 0;
  let prevPt    = null;
  let prevPtT   = 0;

  // ── Auto-rotate ───────────────────────────────────────────────────────────
  const AUTO_DEG_PER_S  = 6.0;
  let autoActive        = AUTO_ROTATE;
  let idleTimer         = null;
  let resumeEased       = 0; // 0→1 ease-in multiplier after resume
  const RESUME_DELAY_MS = 3000;
  const RESUME_EASE_MS  = 1200;

  function pauseAuto() {{
    autoActive  = false;
    resumeEased = 0;
    clearTimeout(idleTimer);
    if (AUTO_ROTATE) {{
      idleTimer = setTimeout(() => {{
        autoActive  = true;
        resumeEased = 0;
      }}, RESUME_DELAY_MS);
    }}
  }}

  // ── RAF loop ──────────────────────────────────────────────────────────────
  let lastTime = null;
  let rafId    = null;
  let visible  = true; // IntersectionObserver

  // prefers-reduced-motion: no auto-rotation (manual drag still works)
  const prefersReduced =
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function tick(now) {{
    rafId = requestAnimationFrame(tick);
    if (!visible) return;

    const dt = lastTime === null ? 16 : Math.min(now - lastTime, 64);
    lastTime = now;

    if (!dragging) {{
      // Apply inertia
      if (Math.abs(VEL[0]) > 1e-5 || Math.abs(VEL[1]) > 1e-5) {{
        ROT[0] += VEL[0] * dt;
        ROT[1]  = clampPhi(ROT[1] + VEL[1] * dt);
        const decay = Math.pow(INERTIA_DECAY, dt / 16.67);
        VEL[0] *= decay;
        VEL[1] *= decay;
        if (Math.abs(VEL[0]) < 1e-5) VEL[0] = 0;
        if (Math.abs(VEL[1]) < 1e-5) VEL[1] = 0;
      }}

      // Auto-rotation
      if (AUTO_ROTATE && autoActive && !prefersReduced) {{
        if (resumeEased < 1) resumeEased = Math.min(1, resumeEased + dt / RESUME_EASE_MS);
        const speed = AUTO_DEG_PER_S * resumeEased;
        ROT[0] += speed * dt / 1000;
      }}
    }}

    ROT[0] = ((ROT[0] + 180) % 360) - 180; // keep in [-180, 180]
    draw(dt);
  }}

  // ── Drawing ───────────────────────────────────────────────────────────────
  function draw(dt) {{
    ctx.clearRect(0, 0, SIZE, SIZE);
    projection.rotate([ROT[0], ROT[1], ROT[2]]);
    pathGen  = d3geo.geoPath(projection).context(ctx);

    // Ocean sphere with radial gradient for depth
    const grad = ctx.createRadialGradient(
      SIZE * 0.42, SIZE * 0.36, 0,
      SIZE / 2,    SIZE / 2,    SIZE / 2
    );
    grad.addColorStop(0,   '#d8eaf6');
    grad.addColorStop(0.7, '#b6cfdf');
    grad.addColorStop(1,   '#8fb3c8');

    ctx.beginPath(); pathGen(sphere);
    ctx.fillStyle = grad; ctx.fill();

    // Land
    if (land) {{
      ctx.beginPath(); pathGen(land);
      ctx.fillStyle = '#cdbf9e'; ctx.fill();
    }}

    // Fill countries represented in the community data. Contributor activity
    // takes the green layer; countries with registrations only use amber.
    activeCountries.forEach(country => {{
      ctx.beginPath(); pathGen(country.feature);
      ctx.fillStyle = country.hasContributor
        ? 'rgba(46,107,71,0.78)'
        : 'rgba(232,168,80,0.76)';
      ctx.fill();
      ctx.strokeStyle = country.hasContributor ? '#1e4830' : '#ad6c12';
      ctx.lineWidth = 1.15;
      ctx.stroke();
    }});

    // Graticule (subtle)
    ctx.beginPath(); pathGen(graticule);
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.lineWidth   = 0.45; ctx.stroke();

    // Country borders
    if (borders) {{
      ctx.beginPath(); pathGen(borders);
      ctx.strokeStyle = 'rgba(160,148,120,0.7)';
      ctx.lineWidth   = 0.5; ctx.stroke();
    }}

    // Globe outline
    ctx.beginPath(); pathGen(sphere);
    ctx.strokeStyle = 'rgba(90,120,150,0.35)';
    ctx.lineWidth   = 1.2; ctx.stroke();

    // Atmosphere glow at limb
    const atmo = ctx.createRadialGradient(
      SIZE/2, SIZE/2, SIZE/2 - 2,
      SIZE/2, SIZE/2, SIZE/2 + 10
    );
    atmo.addColorStop(0, 'rgba(200,225,255,0)');
    atmo.addColorStop(1, 'rgba(200,225,255,0.30)');
    ctx.beginPath();
    ctx.arc(SIZE/2, SIZE/2, SIZE/2 + 10, 0, Math.PI * 2);
    ctx.fillStyle = atmo; ctx.fill();

    // Directional light overlay (upper-left)
    const light = ctx.createRadialGradient(
      SIZE * 0.38, SIZE * 0.32, 0,
      SIZE / 2,    SIZE / 2,    SIZE / 2
    );
    light.addColorStop(0,   'rgba(255,252,244,0.20)');
    light.addColorStop(0.55,'rgba(255,252,244,0.03)');
    light.addColorStop(1,   'rgba(0,0,0,0.10)');
    ctx.beginPath(); pathGen(sphere);
    ctx.fillStyle = light; ctx.fill();

    // One marker per country. Contributor countries use the green layer;
    // registration-only countries use amber.
    drawMarkers();
  }}

  function visibleMarkerPoint(m) {{
      const centre = projection.invert([SIZE / 2, SIZE / 2]);
      if (!centre || d3geo.geoDistance([m.lng, m.lat], centre) >= Math.PI / 2) return null;
      const pt = projection([m.lng, m.lat]);
      return pt && pt[0] !== null ? pt : null;
  }}

  function drawMarkers() {{
    DISPLAY_MARKERS.forEach(m => {{
      const pt = visibleMarkerPoint(m);
      if (!pt) return;

      const count = m.contributorCount || m.signupCount;
      const contributor = m.contributorCount > 0;
      const r = Math.min(3 + Math.sqrt(count) * 1.6, 14);

      ctx.save();
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      ctx.fillStyle   = contributor ? '#2e6b47cc' : '#e8a850cc';
      ctx.fill();
      ctx.strokeStyle = contributor ? '#1e4830' : '#c8831a';
      ctx.lineWidth   = 1.2;
      ctx.stroke();
      ctx.restore();
    }});
  }}

  // ── Pointer Events ────────────────────────────────────────────────────────
  canvas.addEventListener('pointerdown', e => {{
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    dragging = true;
    pauseAuto();
    VEL = [0, 0];
    lastPt  = [e.clientX, e.clientY];
    lastPtT = performance.now();
    prevPt  = lastPt;
    prevPtT = lastPtT;
  }});

  canvas.addEventListener('pointermove', e => {{
    const pt = [e.clientX, e.clientY];

    // Tooltip when not dragging
    if (!dragging) {{
      updateTooltip(e);
    }}

    if (!dragging || !lastPt) return;
    e.preventDefault();

    const now  = performance.now();
    const dLam = (pt[0] - lastPt[0]) / SIZE * 180;
    const dPhi = (pt[1] - lastPt[1]) / SIZE * 180;

    ROT[0]  = ((ROT[0] + dLam + 180) % 360) - 180;
    ROT[1]  = clampPhi(ROT[1] - dPhi);

    const dtSeg = now - prevPtT;
    if (dtSeg > 8) {{
      VEL[0] =  (pt[0] - prevPt[0]) / SIZE * 180 / dtSeg;
      VEL[1] = -(pt[1] - prevPt[1]) / SIZE * 180 / dtSeg;
      prevPt  = lastPt;
      prevPtT = lastPtT;
    }}

    lastPt  = pt;
    lastPtT = now;
  }});

  canvas.addEventListener('pointerup', () => {{
    dragging = false;
    lastPt   = null;
    pauseAuto();
  }});

  canvas.addEventListener('pointercancel', () => {{
    dragging = false; lastPt = null; VEL = [0, 0];
  }});

  canvas.addEventListener('mouseleave', () => {{
    tooltip.style.display = 'none';
  }});

  // ── Tooltip ───────────────────────────────────────────────────────────────
  function updateTooltip(e) {{
    const rect  = canvas.getBoundingClientRect();
    const mx    = e.clientX - rect.left;
    const my    = e.clientY - rect.top;
    let   closest = null;
    let   minDist = 18;  // pixel hit radius

    DISPLAY_MARKERS.forEach(m => {{
      const pt = visibleMarkerPoint(m);
      if (!pt) return;
      const dist = Math.hypot(mx - pt[0], my - pt[1]);
      if (dist < minDist) {{ minDist = dist; closest = m; }}
    }});

    if (closest) {{
      const parts = [];
      if (closest.signupCount) parts.push(
        closest.signupCount + ' registered user' + (closest.signupCount === 1 ? '' : 's')
      );
      if (closest.contributorCount) parts.push(
        closest.contributorCount + ' approved contributor' +
        (closest.contributorCount === 1 ? '' : 's')
      );
      tooltip.textContent = closest.label + ': ' + parts.join(' · ');
      tooltip.style.display = 'block';
      // Clamp so tooltip stays inside the wrapper
      const tx = Math.min(mx + 12, SIZE - tooltip.offsetWidth - 4);
      const ty = Math.max(my - 28, 2);
      tooltip.style.left = tx + 'px';
      tooltip.style.top  = ty + 'px';
    }} else {{
      tooltip.style.display = 'none';
    }}
  }}

  // ── Keyboard ──────────────────────────────────────────────────────────────
  canvas.addEventListener('keydown', e => {{
    const NUDGE = 5;
    switch(e.key) {{
      case 'ArrowLeft':  ROT[0] -= NUDGE; break;
      case 'ArrowRight': ROT[0] += NUDGE; break;
      case 'ArrowUp':    ROT[1] = clampPhi(ROT[1] - NUDGE); break;
      case 'ArrowDown':  ROT[1] = clampPhi(ROT[1] + NUDGE); break;
      default: return;
    }}
    e.preventDefault();
    pauseAuto();
  }});

  // ── Visibility & Intersection pausing ────────────────────────────────────
  function handleVisibilityChange() {{
    visible = document.visibilityState !== 'hidden';
    if (visible && lastTime !== null) lastTime = null;
  }}
  document.addEventListener('visibilitychange', handleVisibilityChange);

  let intersectionObserver = null;
  if (window.IntersectionObserver) {{
    intersectionObserver = new IntersectionObserver(entries => {{
      visible = entries[0].isIntersecting;
      if (visible && lastTime !== null) lastTime = null;
    }}, {{ threshold: 0.1 }});
    intersectionObserver.observe(canvas);
  }}

  function handleResize() {{
    const nextSize = measureSize();
    if (nextSize === SIZE) return;
    SIZE = nextSize;
    applyCanvasSize(SIZE);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    initProjection();
    draw(0);
  }}

  let resizeObserver = null;
  if (window.ResizeObserver) {{
    resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(wrap);
  }} else {{
    window.addEventListener('resize', handleResize);
  }}

  // ── Helpers ───────────────────────────────────────────────────────────────
  function clampPhi(p) {{ return Math.max(-75, Math.min(75, p)); }}

  // ── Init ──────────────────────────────────────────────────────────────────
  initProjection();
  rafId = requestAnimationFrame(tick);

}})();
</script>
</body>
</html>"""


def render_globe(
    markers: list[dict],
    *,
    height: int = 460,
    auto_rotate: bool = True,
) -> None:
    """Render the interactive orthographic globe component.

    Parameters
    ----------
    markers:
        ``[{{"lat": float, "lng": float, "label": str, "count": int,
        "layer": "signup" | "contributor"}}]``
    height:
        Component height in pixels (the globe canvas scales to fill).
    auto_rotate:
        Whether to start with auto-rotation enabled.  Callers should pass
        False if ``prefers-reduced-motion`` is active at the Python level.
    """
    import streamlit as st

    markers_json = _json_for_script(_normalise_markers(markers))
    static = _load_static_assets()
    if static is None:
        html = _build_fallback_html(
            markers_json=markers_json,
            auto_rotate=auto_rotate,
            height=height,
        )
    else:
        d3_script, topojson_script, world_json = static
        html = _build_html(
            d3_script=d3_script,
            topojson_script=topojson_script,
            world_json=world_json,
            markers_json=markers_json,
            auto_rotate=auto_rotate,
            height=height,
        )

    # No key= anywhere below: neither embedding API takes one, and passing a
    # key raises TypeError.  Streamlit identifies the iframe by its position in
    # the element tree plus a hash of the HTML, so as long as the markers are
    # stable (they are cached for 5 min, ttl=300) the component is not
    # remounted across the landing page's widget interactions.
    #
    # st.components.v1.html is deprecated in favour of st.iframe and is slated
    # for removal, but requirements.txt allows streamlit>=1.42, which predates
    # st.iframe.  Prefer the new API and fall back so the globe keeps working
    # on both sides of that change rather than breaking on one of them.
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=False)
