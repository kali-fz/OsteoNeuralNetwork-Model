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
  - Square-root scaled marker area
  - Visibility and IntersectionObserver pausing (zero CPU in background)
  - devicePixelRatio capped at 2

Assets are downloaded once to src/components/assets/ from pinned CDN URLs and
read from disk on every subsequent render.  The component never makes a network
request during rendering; the download only ever happens once per server process
when the asset file is absent.

Privacy contract (section 3B of REDESIGN_BRIEF.md):
  - Markers carry only {lat, lng, label, count, layer}: no user id, no email,
    no timestamp, no sub-country coordinate.
  - Never call the browser Geolocation API.
  - Never embed any API key in the HTML (it is browser-readable).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

# streamlit is imported lazily inside render_globe and _load_static_assets so
# that the module is importable in test environments without a running server.

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

# Pinned CDN versions — bump only these two constants to upgrade.
_ASSETS: dict[str, str] = {
    "d3-geo.min.js": (
        "https://cdn.jsdelivr.net/npm/d3-geo@3.1.0/dist/d3-geo.umd.min.js"
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
    """Load JS + world JSON once per server startup into memory.

    Returns (d3_script, topojson_script, world_json) or None when unavailable.
    When running inside Streamlit, uses cache_resource so the ~200 KB read
    happens once regardless of how many visitors request the landing page.
    When running in a test environment without Streamlit, falls back to a plain
    call (assets are loaded on every test invocation, which is acceptable).
    """
    def _load():
        if not _ensure_assets():
            return None
        try:
            d3 = (ASSETS_DIR / "d3-geo.min.js").read_text(encoding="utf-8")
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


def _build_html(
    d3_script: str,
    topojson_script: str,
    world_json: str,
    markers_json: str,
    auto_rotate: bool,
    height: int,
) -> str:
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
  const WORLD   = {world_json};
  const AUTO_ROTATE = {str(auto_rotate).lower()};

  // ── Canvas setup ──────────────────────────────────────────────────────────
  const canvas = document.getElementById('globe');
  const tooltip = document.getElementById('tooltip');
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  function resize() {{
    const wrap  = document.getElementById('globe-wrap');
    const avail = Math.min(wrap.clientWidth, {height} - 44);
    const size  = Math.max(160, avail);
    canvas.style.width  = size + 'px';
    canvas.style.height = size + 'px';
    canvas.width  = size * DPR;
    canvas.height = size * DPR;
    return size;
  }}

  let SIZE = resize();
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
    pathGen  = d3geo.geoPath().context(ctx);
    graticule = d3geo.geoGraticule()();
    sphere    = {{ type: 'Sphere' }};
  }}

  // ── World geometry ────────────────────────────────────────────────────────
  let land = null, borders = null;
  if (WORLD && typeof topojson !== 'undefined') {{
    try {{
      land    = topojson.feature(WORLD, WORLD.objects.land);
      // countries mesh for borders (no shared edges between countries)
      borders = topojson.mesh(WORLD, WORLD.objects.countries,
                              (a, b) => a !== b);
    }} catch(e) {{ console.warn('topojson parse error', e); }}
  }}

  // ── Precompute marker 3-D unit vectors ────────────────────────────────────
  // These are fixed; only the rotation matrix changes per frame.
  const markerVecs = MARKERS.map(m => {{
    const lam = m.lng * Math.PI / 180;
    const phi = m.lat * Math.PI / 180;
    return {{
      ...m,
      _x: Math.cos(phi) * Math.cos(lam),
      _y: Math.cos(phi) * Math.sin(lam),
      _z: Math.sin(phi),
    }};
  }});

  // ── Rotation state ────────────────────────────────────────────────────────
  let ROT    = [0, -20, 0];   // [lambda, phi, gamma] degrees
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
    pathGen  = d3geo.geoPath().context(ctx);

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

    // Markers – signup layer first, then contributor on top
    drawMarkers('signup',      '#e8a850', '#c8831a', 1.6);
    drawMarkers('contributor', '#2e6b47', '#1e4830', 1.6);
  }}

  function drawMarkers(layer, fill, stroke, scale) {{
    // Compute rotation matrix once (reuse for all markers in this layer)
    const [l, p, g] = ROT.map(d => d * Math.PI / 180);
    const cl = Math.cos(-l), sl = Math.sin(-l);
    const cp = Math.cos(-p), sp = Math.sin(-p);
    const cg = Math.cos(-g), sg = Math.sin(-g);

    markerVecs.forEach(m => {{
      if (m.layer !== layer) return;

      // Rotate the precomputed unit vector into camera space
      // (inverse of the projection rotation)
      let x = m._x, y = m._y, z = m._z;
      // Apply -gamma rotation (around z-axis)
      let nx = x * cg - y * sg, ny = x * sg + y * cg; x = nx; y = ny;
      // Apply -phi rotation (around x-axis)
      let nz = z * cp - y * sp, ny2 = z * sp + y * cp; z = nz; y = ny2;
      // Apply -lambda rotation (around z-axis)
      let nx2 = x * cl - y * sl, ny3 = x * sl + y * cl; x = nx2; y = ny3;

      // z < 0 means marker is on the far hemisphere (not visible)
      if (z < -0.05) return;

      const pt = projection([m.lng, m.lat]);
      if (!pt || pt[0] === null) return;

      const r = Math.min(3 + Math.sqrt(m.count) * scale, 14);
      const alpha = z < 0.1 ? (z + 0.05) / 0.15 : 1; // soft edge fade

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      ctx.fillStyle   = fill + 'cc';
      ctx.fill();
      ctx.strokeStyle = stroke;
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

    ROT[0]  = ((ROT[0] - dLam + 180) % 360) - 180;
    ROT[1]  = clampPhi(ROT[1] + dPhi);

    const dtSeg = now - prevPtT;
    if (dtSeg > 8) {{
      VEL[0] = -(pt[0] - prevPt[0]) / SIZE * 180 / dtSeg;
      VEL[1] =  (pt[1] - prevPt[1]) / SIZE * 180 / dtSeg;
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

    markerVecs.forEach(m => {{
      const pt = projection([m.lng, m.lat]);
      if (!pt || pt[0] === null) return;
      const dist = Math.hypot(mx - pt[0], my - pt[1]);
      if (dist < minDist) {{ minDist = dist; closest = m; }}
    }});

    if (closest) {{
      tooltip.textContent = closest.label + ' — ' + closest.count +
        (closest.layer === 'contributor' ? ' contributor' : ' user') +
        (closest.count === 1 ? '' : 's');
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
  document.addEventListener('visibilitychange', () => {{
    visible = document.visibilityState !== 'hidden';
    if (visible && lastTime !== null) lastTime = null;
  }});

  if (window.IntersectionObserver) {{
    new IntersectionObserver(entries => {{
      visible = entries[0].isIntersecting;
      if (visible && lastTime !== null) lastTime = null;
    }}, {{ threshold: 0.1 }}).observe(canvas);
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
    static = _load_static_assets()
    if static is None:
        try:
            import streamlit as st
            st.markdown(
                "<div style='height:200px;display:flex;align-items:center;"
                "justify-content:center;color:#6b6457;font-size:14px;'>"
                "Globe unavailable — check network and restart the app.</div>",
                unsafe_allow_html=True,
            )
        except ImportError:
            pass
        return

    import streamlit.components.v1 as components

    d3_script, topojson_script, world_json = static
    markers_json = json.dumps(markers)

    html = _build_html(
        d3_script=d3_script,
        topojson_script=topojson_script,
        world_json=world_json,
        markers_json=markers_json,
        auto_rotate=auto_rotate,
        height=height,
    )

    # Stable key: Streamlit will NOT remount the iframe on reruns unless the
    # HTML content changes.  Since markers are cached for 5 min (ttl=300) the
    # component survives all of the widget interactions on the landing page.
    components.html(html, height=height, scrolling=False, key="onnm_globe")
