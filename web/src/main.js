import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import * as d3 from 'd3';

const DATA = await (await fetch(`${import.meta.env.BASE_URL}ecg_data.json`)).json();
const M = DATA.methods, FS = DATA.fs, NB = M[0].beats.length, NR = 180;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- cyclic cardiac-phase colour ----------
const STOPS = ['#F2665A', '#E8B04A', '#39BFA8', '#5B8CEB', '#A879E6'];
const phaseInterp = d3.interpolateRgbBasisClosed(STOPS);
const PAL_CSS = d3.range(256).map(i => phaseInterp(i / 256));
const PAL_LIN = PAL_CSS.map(c => { const r = d3.rgb(c); return new THREE.Color().setRGB(r.r / 255, r.g / 255, r.b / 255, THREE.SRGBColorSpace); });
const pidx = ph => ((Math.floor((((ph % 1) + 1) % 1) * 256)) % 256);

// ---------- time model ----------
M.forEach(m => { m.t1 = m.beats.map((b, i) => m.t0[i] + (b.length - 1) / FS); });
const tStart = M[0].t0[0], tEnd = M[0].t1[NB - 1];
function at(m, t) {
  for (let i = 0; i < NB; i++) {
    if (t >= m.t0[i] && t <= m.t1[i]) {
      const b = m.beats[i], f = (t - m.t0[i]) * FS, k = Math.min(Math.floor(f), b.length - 2), u = f - k;
      const p = b[k], q = b[k + 1];
      return [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, p[2] + (q[2] - p[2]) * u, u < .5 ? p[3] : q[3], i];
    }
  }
  return null;
}
// index-resampled beats for morphing between methods
M.forEach(m => {
  m.rs = m.beats.map(b => {
    const pos = new Float32Array(NR * 3), ph = new Float32Array(NR);
    for (let j = 0; j < NR; j++) {
      const f = j / (NR - 1) * (b.length - 1), k = Math.min(Math.floor(f), b.length - 2), u = f - k;
      for (let d = 0; d < 3; d++) pos[j * 3 + d] = b[k][d] + (b[k + 1][d] - b[k][d]) * u;
      ph[j] = u < .5 ? b[k][3] : b[k + 1][3];
    }
    return { pos, ph };
  });
});

const state = { cur: 0, from: 0, k: 1, t: tStart + 0.05, playing: !reduceMotion, speed: 0.5, trail: 0.4, bright: false };

// ---------- three.js stage ----------
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.prepend(renderer.domElement);
const labelRenderer = new CSS2DRenderer({ element: document.getElementById('labels') });

const scene = new THREE.Scene();
scene.background = new THREE.Color('#0B1016');
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
camera.position.set(2.6, 1.5, 3.1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.autoRotateSpeed = 0.8;

const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.2, 2.2, 2.2)),
  new THREE.LineBasicMaterial({ color: '#2A3542', transparent: true, opacity: 0.7 }));
scene.add(box);
const grid = new THREE.GridHelper(2.2, 11, '#22303D', '#18212B'); grid.position.y = -1.1; scene.add(grid);
const axisMat = c => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.55 });
const axisCols = ['#3F92D6', '#C97B38', '#9672D4'];
const axLabels = [];
[[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((v, i) => {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(...v).multiplyScalar(1.1)]);
  scene.add(new THREE.Line(g, axisMat(axisCols[i])));
  const el = document.createElement('div'); el.className = 'axlabel'; el.style.borderLeft = `2px solid ${axisCols[i]}`;
  const o = new CSS2DObject(el); o.position.set(...v.map(c => c * 1.22)); scene.add(o); axLabels.push(el);
});

const lineMat = new LineMaterial({ vertexColors: true, linewidth: 1.7, transparent: true, opacity: 0.4, depthWrite: false });
const beatLines = d3.range(NB).map(() => { const l = new Line2(new LineGeometry(), lineMat); scene.add(l); return l; });
function setBeatGeometry() {
  const A = M[state.from].rs, B = M[state.cur].rs, e = d3.easeCubicInOut(state.k);
  for (let i = 0; i < NB; i++) {
    const pos = new Float32Array(NR * 3), col = new Float32Array(NR * 3);
    for (let j = 0; j < NR * 3; j++) pos[j] = A[i].pos[j] + (B[i].pos[j] - A[i].pos[j]) * e;
    for (let j = 0; j < NR; j++) {
      const c = PAL_LIN[pidx(e < .5 ? A[i].ph[j] : B[i].ph[j])];
      col[j * 3] = c.r; col[j * 3 + 1] = c.g; col[j * 3 + 2] = c.b;
    }
    const g = new LineGeometry(); g.setPositions(pos); g.setColors(col);
    beatLines[i].geometry.dispose(); beatLines[i].geometry = g; beatLines[i].computeLineDistances();
  }
}

// comet trail: additive soft points
const TMAX = 260;
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TMAX * 3), 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TMAX * 3), 3));
trailGeo.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(TMAX), 1));
const trailMat = new THREE.ShaderMaterial({
  uniforms: { uPR: { value: renderer.getPixelRatio() } },
  vertexShader: `attribute float aSize; attribute vec3 color; varying vec3 vC; varying float vA;
    uniform float uPR;
    void main(){ vC = color; vA = aSize; vec4 mv = modelViewMatrix * vec4(position,1.0);
      gl_PointSize = (2.0 + 16.0*aSize) * uPR * (2.6 / -mv.z); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `varying vec3 vC; varying float vA;
    void main(){ float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
      float a = smoothstep(0.5, 0.0, d) * (0.15 + 0.85*vA); gl_FragColor = vec4(vC * (0.6 + 0.9*vA), a); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true
});
const trail = new THREE.Points(trailGeo, trailMat); scene.add(trail);
const head = new THREE.Mesh(new THREE.SphereGeometry(0.03, 20, 16), new THREE.MeshBasicMaterial({ color: '#ffffff' }));
scene.add(head);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.9, 0.55, 0.12);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  lineMat.resolution.set(w, h);
}
new ResizeObserver(resize).observe(stage); resize();

function posAt(t) {
  const b = at(M[state.cur], t);
  if (state.k >= 1 || !b) return b;
  const a = at(M[state.from], t); if (!a) return b;
  const e = d3.easeCubicInOut(state.k);
  return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e, b[3], b[4]];
}
function updateTrail() {
  const P = trailGeo.attributes.position.array, C = trailGeo.attributes.color.array, S = trailGeo.attributes.aSize.array;
  const n = Math.min(TMAX, Math.max(2, Math.round(state.trail * FS)));
  let w = 0;
  for (let j = 0; j < n; j++) {
    const t = state.t - (n - 1 - j) / FS, p = posAt(t);
    if (!p) continue;
    P[w * 3] = p[0]; P[w * 3 + 1] = p[1]; P[w * 3 + 2] = p[2];
    const c = PAL_LIN[pidx(p[3])]; C[w * 3] = c.r; C[w * 3 + 1] = c.g; C[w * 3 + 2] = c.b;
    S[w] = Math.pow(j / (n - 1), 1.6); w++;
  }
  trailGeo.setDrawRange(0, w);
  trailGeo.attributes.position.needsUpdate = trailGeo.attributes.color.needsUpdate = trailGeo.attributes.aSize.needsUpdate = true;
  const p = posAt(state.t);
  head.visible = !!p;
  if (p) { head.position.set(p[0], p[1], p[2]); head.material.color.copy(PAL_LIN[pidx(p[3])]).lerp(new THREE.Color(1, 1, 1), 0.55); }
  return p;
}

// ---------- phase clock ----------
const clock = document.getElementById('clock'), cx = clock.getContext('2d');
function drawClock(ph) {
  const W = clock.width, r = W * 0.36, c = W / 2;
  cx.clearRect(0, 0, W, W);
  for (let i = 0; i < 256; i++) {
    const a0 = (i / 256) * 2 * Math.PI - Math.PI / 2, a1 = ((i + 1.5) / 256) * 2 * Math.PI - Math.PI / 2;
    cx.beginPath(); cx.arc(c, c, r, a0, a1); cx.strokeStyle = PAL_CSS[i]; cx.lineWidth = 14; cx.stroke();
  }
  cx.font = '500 20px JetBrains Mono, monospace'; cx.fillStyle = '#B5BFCC'; cx.textAlign = 'center'; cx.textBaseline = 'middle';
  [['R', 0], ['T', 0.37], ['P', 0.82]].forEach(([s, f]) => {
    const a = f * 2 * Math.PI - Math.PI / 2; cx.fillText(s, c + Math.cos(a) * (r + 26), c + Math.sin(a) * (r + 26));
  });
  if (ph == null) return;
  const a = ph * 2 * Math.PI - Math.PI / 2;
  cx.beginPath(); cx.moveTo(c, c); cx.lineTo(c + Math.cos(a) * (r - 12), c + Math.sin(a) * (r - 12));
  cx.strokeStyle = '#E8ECF1'; cx.lineWidth = 3; cx.lineCap = 'round'; cx.stroke();
  cx.beginPath(); cx.arc(c + Math.cos(a) * r, c + Math.sin(a) * r, 10, 0, 2 * Math.PI);
  cx.fillStyle = PAL_CSS[pidx(ph)]; cx.fill(); cx.lineWidth = 3; cx.strokeStyle = '#0B1016'; cx.stroke();
  cx.font = '500 22px JetBrains Mono, monospace'; cx.fillStyle = '#E8ECF1';
  cx.fillText(ph.toFixed(2), c, c + 24);
}

// ---------- small multiples ----------
const cardsEl = document.getElementById('cards');
const cards = M.map((m, i) => {
  const b = document.createElement('button');
  b.className = 'card'; b.setAttribute('aria-pressed', 'false'); b.title = `Show ${m.name} (key ${i + 1})`;
  b.innerHTML = `<div class="nm"><span>${m.name}</span><i>${i + 1}</i></div><div class="cv"><canvas></canvas><canvas></canvas></div>`;
  b.addEventListener('click', () => select(i));
  cardsEl.appendChild(b);
  const [base, over] = b.querySelectorAll('canvas');
  return { el: b, base, over };
});
function drawCards() {
  const dpr = Math.min(devicePixelRatio, 2);
  cards.forEach((c, i) => {
    const s = c.base.clientWidth; if (!s) return;
    [c.base, c.over].forEach(cv => { cv.width = s * dpr; cv.height = s * dpr; });
    const g = c.base.getContext('2d'); g.scale(dpr, dpr);
    const sc = d3.scaleLinear([-1.08, 1.08], [4, s - 4]), sy = d3.scaleLinear([-1.08, 1.08], [s - 4, 4]);
    g.strokeStyle = '#1B232E'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(sc(0), 4); g.lineTo(sc(0), s - 4); g.moveTo(4, sy(0)); g.lineTo(s - 4, sy(0)); g.stroke();
    g.lineWidth = 1.1; g.globalAlpha = 0.75; g.lineCap = 'round';
    M[i].beats.forEach(b => {
      for (let k = 0; k < b.length - 1; k++) {
        g.strokeStyle = PAL_CSS[pidx(b[k][3])];
        g.beginPath(); g.moveTo(sc(b[k][0]), sy(b[k][1])); g.lineTo(sc(b[k + 1][0]), sy(b[k + 1][1])); g.stroke();
      }
    });
    c.sc = sc; c.sy = sy; c.s = s; c.dpr = dpr;
  });
}
function drawCardDots() {
  cards.forEach((c, i) => {
    if (!c.sc) return;
    const g = c.over.getContext('2d'); g.setTransform(c.dpr, 0, 0, c.dpr, 0, 0); g.clearRect(0, 0, c.s, c.s);
    const p = at(M[i], state.t); if (!p) return;
    g.beginPath(); g.arc(c.sc(p[0]), c.sy(p[1]), 4.5, 0, 2 * Math.PI);
    g.fillStyle = '#fff'; g.fill(); g.lineWidth = 2; g.strokeStyle = '#0B1016'; g.stroke();
  });
}
new ResizeObserver(drawCards).observe(cardsEl);

// ---------- ECG-paper trace ----------
const traceEl = document.getElementById('trace');
const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; traceEl.appendChild(tip);
const SER = ['--s-x', '--s-y', '--s-z'].map(v => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
let tr = null;
function seriesFor(m) {
  const pts = [];
  m.beats.forEach((b, i) => {
    if (i > 0 && m.t0[i] - m.t1[i - 1] > 1.5 / FS) pts.push(null);
    b.forEach((p, k) => pts.push({ t: m.t0[i] + k / FS, v: p }));
  });
  return pts;
}
function drawTrace() {
  const W = Math.max(traceEl.clientWidth, 560), H = 220, mg = { l: 34, r: 56, t: 8, b: 22 };
  const x = d3.scaleLinear([tStart, tEnd], [mg.l, W - mg.r]), y = d3.scaleLinear([-1.05, 1.05], [H - mg.b, mg.t]);
  d3.select(traceEl).select('svg').remove();
  const svg = d3.select(traceEl).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .style('min-width', '560px').style('cursor', 'crosshair').attr('role', 'img');
  const paper = svg.append('g');
  for (let t = Math.ceil(tStart / 0.04) * 0.04; t <= tEnd; t += 0.04) {
    const major = Math.abs(t / 0.2 - Math.round(t / 0.2)) < 1e-6;
    paper.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', mg.t).attr('y2', H - mg.b)
      .attr('stroke', major ? 'var(--paper-major)' : 'var(--paper-minor)');
  }
  for (let v = -1; v <= 1.0001; v += 0.1) {
    const major = Math.abs(v * 2 - Math.round(v * 2)) < 1e-6;
    paper.append('line').attr('x1', mg.l).attr('x2', W - mg.r).attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', major ? 'var(--paper-major)' : 'var(--paper-minor)');
  }
  svg.append('g').attr('transform', `translate(0,${H - mg.b})`).call(d3.axisBottom(x).ticks(12).tickFormat(d => d.toFixed(0) + ' s').tickSize(0).tickPadding(8)).select('.domain').remove();
  svg.append('g').attr('transform', `translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues([-1, -0.5, 0, 0.5, 1]).tickSize(0).tickPadding(6)).select('.domain').remove();
  const m = M[state.cur], pts = seriesFor(m);
  const g = svg.append('g');
  [0, 1, 2].forEach(d => {
    g.append('path').attr('fill', 'none').attr('stroke', SER[d]).attr('stroke-width', 1.6).attr('stroke-linejoin', 'round')
      .attr('d', d3.line().defined(p => p).x(p => x(p.t)).y(p => y(p.v[d]))(pts));
    const last = pts[pts.length - 1];
    g.append('text').attr('x', W - mg.r + 6).attr('y', y(last.v[d]) + [-8, 2, 12][d]).attr('fill', SER[d]).style('fill', 'var(--ink-2)').text(m.axes[d]);
  });
  const ph = svg.append('line').attr('y1', mg.t).attr('y2', H - mg.b).attr('stroke', 'var(--ink)').attr('stroke-width', 1.5);
  const dots = [0, 1, 2].map(d => svg.append('circle').attr('r', 4).attr('fill', SER[d]).attr('stroke', 'var(--panel)').attr('stroke-width', 2));
  const scrub = ev => { const [px] = d3.pointer(ev, svg.node()); state.t = Math.max(tStart, Math.min(tEnd, x.invert(px))); };
  let dragging = false;
  svg.on('pointerdown', ev => { dragging = true; scrub(ev); svg.node().setPointerCapture(ev.pointerId); })
     .on('pointerup pointercancel', () => { dragging = false; })
     .on('pointermove', ev => {
        if (dragging) scrub(ev);
        const [px, py] = d3.pointer(ev, svg.node()), t = x.invert(px), p = at(M[state.cur], t);
        if (!p || px < mg.l || px > W - mg.r) { tip.hidden = true; return; }
        tip.hidden = false;
        tip.innerHTML = `<span class="k">t</span> ${t.toFixed(3)} s · <span class="k">phase</span> ${p[3].toFixed(2)}<br>` +
          M[state.cur].axes.map((a, d) => `<span style="color:${SER[d]}">■</span> ${a} ${p[d] >= 0 ? ' ' : ''}${p[d].toFixed(3)}`).join('<br>');
        const r = traceEl.getBoundingClientRect(), sr = svg.node().getBoundingClientRect(), k = sr.width / W;
        let lx = (px * k) + sr.left - r.left + traceEl.scrollLeft + 14; if (lx > traceEl.clientWidth - 190) lx -= 200;
        tip.style.left = lx + 'px'; tip.style.top = (py * k + 4) + 'px';
     })
     .on('pointerleave', () => { tip.hidden = true; });
  tr = { x, y, ph, dots };
  document.getElementById('trace-title').textContent = `${m.name}: coordinates over time`;
  document.getElementById('trace-legend').innerHTML = m.axes.map((a, d) => `<span><i class="sw" style="border-color:${SER[d]}"></i>${a}</span>`).join('');
}
function updateTraceHead() {
  if (!tr) return;
  const X = tr.x(state.t); tr.ph.attr('x1', X).attr('x2', X);
  const p = at(M[state.cur], state.t);
  tr.dots.forEach((c, d) => p ? c.attr('display', null).attr('cx', X).attr('cy', tr.y(p[d])) : c.attr('display', 'none'));
}
new ResizeObserver(() => drawTrace()).observe(traceEl);

// ---------- Matryoshka vs PCA ----------
function drawRecon() {
  const el = document.getElementById('recon'), R = DATA.recon;
  const W = Math.max(el.clientWidth, 380), H = 250, mg = { l: 44, r: 92, t: 10, b: 34 };
  d3.select(el).selectAll('*').remove();
  const ser = [
    { k: 'mrl_slog', name: 'Matryoshka · log', c: 'var(--s-x)', dash: null, from: 2 },
    { k: 'mrl', name: 'Matryoshka · raw', c: 'var(--s-x)', dash: '5 4', from: 2 },
    { k: 'pca_slog', name: 'PCA · log', c: 'var(--s-y)', dash: null, from: 1 },
    { k: 'pca', name: 'PCA · raw', c: 'var(--s-y)', dash: '5 4', from: 1 },
  ].map(s => ({ ...s, pts: R[s.k].map((v, i) => ({ m: i + 1, v })).filter(p => p.m >= s.from) }));
  const x = d3.scaleLinear([1, 16], [mg.l, W - mg.r]), y = d3.scaleLog([0.008, 1], [H - mg.b, mg.t]).clamp(true);
  const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H).style('min-width', '380px');
  svg.append('g').selectAll('line').data([0.01, 0.03, 0.1, 0.3, 1]).join('line')
    .attr('x1', mg.l).attr('x2', W - mg.r).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', 'var(--line-soft)');
  svg.append('g').attr('transform', `translate(0,${H - mg.b})`).call(d3.axisBottom(x).tickValues([1, 2, 3, 4, 8, 12, 16]).tickSize(0).tickPadding(8)).select('.domain').attr('stroke', 'var(--line)');
  svg.append('text').attr('x', (mg.l + W - mg.r) / 2).attr('y', H - 2).attr('text-anchor', 'middle').text('dimensions kept (m)');
  svg.append('g').attr('transform', `translate(${mg.l},0)`).call(d3.axisLeft(y).tickValues([0.01, 0.03, 0.1, 0.3, 1]).tickFormat(d3.format('.0%')).tickSize(0).tickPadding(6)).select('.domain').remove();
  ser.forEach(s => {
    svg.append('path').attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2).attr('stroke-dasharray', s.dash)
      .attr('d', d3.line().x(p => x(p.m)).y(p => y(p.v))(s.pts));
  });
  // end labels, nudged apart so close finishes don't overlap
  const ends = ser.map(s => ({ name: s.name, y: y(s.pts[s.pts.length - 1].v) + 3 })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12);
  ends.forEach(e => svg.append('text').attr('x', W - mg.r + 6).attr('y', e.y).style('fill', 'var(--ink-2)').text(e.name));
  [2, 3, 4, 8, 16].forEach(m => ['mrl_slog', 'mrl'].forEach(k =>
    svg.append('circle').attr('cx', x(m)).attr('cy', y(R[k][m - 1])).attr('r', 4).attr('fill', 'var(--s-x)').attr('stroke', 'var(--panel)').attr('stroke-width', 2)));
  // hover crosshair
  const ch = svg.append('line').attr('y1', mg.t).attr('y2', H - mg.b).attr('stroke', 'var(--muted)').attr('stroke-dasharray', '2 3').attr('display', 'none');
  const tp = document.createElement('div'); tp.className = 'tip'; tp.hidden = true; el.appendChild(tp); el.style.position = 'relative';
  svg.append('rect').attr('x', mg.l).attr('y', mg.t).attr('width', W - mg.l - mg.r).attr('height', H - mg.t - mg.b).attr('fill', 'transparent')
    .on('pointermove', ev => {
      const [px] = d3.pointer(ev), m = Math.max(1, Math.min(16, Math.round(x.invert(px))));
      ch.attr('display', null).attr('x1', x(m)).attr('x2', x(m));
      tp.hidden = false;
      tp.innerHTML = `<span class="k">m =</span> ${m}<br>` + ser.map(s => {
        const p = s.pts.find(q => q.m === m); return `${s.name.padEnd(17)} ${p ? (100 * p.v).toFixed(1) + '%' : '–'}`;
      }).join('<br>');
      const k = svg.node().getBoundingClientRect().width / W;
      let lx = x(m) * k + 12; if (lx > el.clientWidth - 200) lx -= 220;
      tp.style.left = lx + 'px'; tp.style.top = '8px';
    })
    .on('pointerleave', () => { ch.attr('display', 'none'); tp.hidden = true; });
  const pct = v => (100 * v).toFixed(0) + '%';
  document.getElementById('recon-callout').innerHTML =
    `Reconstruction error of 128 ms, 9-lead windows (standardized, so 100% means nothing explained). With two dimensions, the Matryoshka prefix leaves <b>${pct(R.mrl[1])}</b> unexplained and PCA leaves <b>${pct(R.pca[1])}</b>. Dots mark the prefix sizes the loss trains directly (2, 3, 4, 8, 16). These errors are measured on the training data, with no held-out set.`;
}
new ResizeObserver(drawRecon).observe(document.getElementById('recon'));

// ---------- method list ----------
document.getElementById('methods').innerHTML = M.map(m => `<li><b>${m.name}</b><span>${m.blurb}</span></li>`).join('');
document.getElementById('sub').textContent = `${DATA.record} · ${tStart.toFixed(1)}–${tEnd.toFixed(1)} s · ${NB} beats · median R–R ${(DATA.rr * 1000).toFixed(0)} ms · ${FS} Hz`;

// ---------- selection & controls ----------
function select(i) {
  if (i === state.cur && state.k >= 1) return;
  state.from = state.k >= 1 ? state.cur : state.cur; state.cur = i; state.k = reduceMotion ? 1 : 0;
  cards.forEach((c, j) => c.el.setAttribute('aria-pressed', j === i ? 'true' : 'false'));
  document.getElementById('hud-eyebrow').textContent = `Method ${i + 1} of ${M.length}`;
  document.getElementById('hud-name').textContent = M[i].name;
  document.getElementById('hud-blurb').textContent = M[i].blurb;
  axLabels.forEach((el, d) => el.textContent = M[i].axes[d]);
  setBeatGeometry(); drawTrace();
}
const $ = id => document.getElementById(id);
const playBtn = $('play');
function setPlaying(p) { state.playing = p; playBtn.textContent = p ? 'Pause' : 'Play'; playBtn.setAttribute('aria-pressed', p); }
playBtn.onclick = () => setPlaying(!state.playing);
$('speed').oninput = e => { state.speed = +e.target.value; $('speed-o').textContent = state.speed.toFixed(2) + '×'; };
$('trail').oninput = e => { state.trail = +e.target.value; $('trail-o').textContent = Math.round(state.trail * 1000) + ' ms'; };
$('glow').oninput = e => { bloom.strength = +e.target.value; $('glow-o').textContent = (+e.target.value).toFixed(2); };
$('allbeats').onclick = e => { state.bright = !state.bright; e.target.setAttribute('aria-pressed', state.bright); lineMat.opacity = state.bright ? 0.85 : 0.4; };
$('rotate').onclick = e => { controls.autoRotate = !controls.autoRotate; e.target.setAttribute('aria-pressed', controls.autoRotate); };
let camAnim = null;
function faceOn() {
  const from = camera.position.clone(), to = new THREE.Vector3(0, 0, 3.6);
  controls.autoRotate = false; $('rotate').setAttribute('aria-pressed', 'false');
  camAnim = { from, to, k: reduceMotion ? 1 : 0 };
  controls.target.set(0, 0, 0);
}
$('face').onclick = faceOn;
window.addEventListener('keydown', e => {
  if (e.target.matches('input')) return;
  if (e.key >= '1' && e.key <= String(M.length)) select(+e.key - 1);
  else if (e.key === ' ') { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.key === 'f' || e.key === 'F') faceOn();
});

// ---------- loop ----------
setPlaying(state.playing);
state.cur = 0; state.from = 0; state.k = 1; select(0); setBeatGeometry();
const clk = new THREE.Clock();
const readout = $('readout');
function frame() {
  const dt = Math.min(clk.getDelta(), 0.05);
  if (state.playing) { state.t += dt * state.speed; if (state.t > tEnd) state.t = tStart; }
  if (state.k < 1) { state.k = Math.min(1, state.k + dt / 0.7); setBeatGeometry(); }
  if (camAnim) {
    camAnim.k = Math.min(1, camAnim.k + dt / 0.8);
    camera.position.lerpVectors(camAnim.from, camAnim.to, d3.easeCubicInOut(camAnim.k));
    if (camAnim.k >= 1) camAnim = null;
  }
  controls.update();
  const p = updateTrail();
  drawClock(p ? p[3] : null);
  readout.innerHTML = `t <b>${state.t.toFixed(3)} s</b><br>beat <b>${p ? p[4] + 1 : '–'}</b> / ${NB}`;
  drawCardDots(); updateTraceHead();
  composer.render(); labelRenderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
