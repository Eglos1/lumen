/* ============================================================================
 * LUMEN frontend — Compute-Over-Network player
 *
 * Pipeline per frame:
 *   1. hidden <video> plays a forced-144p stream (server /api/stream)
 *   2. we pull the raw luma plane from a 256px-wide offscreen canvas
 *   3. ONNX Runtime Web (WebGPU EP, falls back WebGL/WASM) hallucinates the
 *      missing detail: 224x224 tile -> 672x672 (3x super-resolution)
 *   4. the SR luma is combined with bilinear-upscaled chroma and pushed to a
 *      visible <canvas>; text (title/subtitles/controls) is drawn as crisp
 *      HTML on top — the neural net never renders text.
 * ========================================================================== */
'use strict';

/* ----------------------------- constants ------------------------------- */
const MODEL_URL = '/models/super-resolution-10.onnx';
const TILE = 224;          // ONNX model fixed input
const SR_OUT = 672;        // 3x model output
const WASM_DIR = '/vendor/onnxruntime-web/';
const LOW_W = 256;         // our internal low-res work width (144p-class)
const API = '';            // same-origin

const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

/* ----------------------------- neural engine ----------------------------- */
const Engine = {
  session: null,
  provider: null,          // 'webgpu' | 'webgl' | 'wasm' | 'off'
  inputName: 'input',
  outputName: 'output',
  skipFrames: 1,

  /* try WebGPU -> WebGL -> WASM; every path is first-request lazy */
  async load() {
    try {
      if (!window.ort) throw new Error('onnxruntime-web failed to load');
      ort.env.wasm.wasmPaths = WASM_DIR;
      ort.env.wasm.numThreads = 1;          // no SharedArrayBuffer needed

      const choices = [];
      if (navigator.gpu) choices.push('webgpu');
      choices.push('webgl', 'wasm');
      for (const ep of choices) {
        try {
          this.session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
          });
          this.provider = ep;
          this.inputName = this.session.inputNames[0];
          this.outputName = this.session.outputNames[0];
          break;
        } catch (e) { /* try next provider */ }
      }
    } catch (e) { console.error('engine:', e); }
    badge();
  },

  get ready() { return !!this.session; },

  async upscaleTile(input) {
    const feed = {};
    feed[this.inputName] = new ort.Tensor('float32', input, [1, 1, TILE, TILE]);
    const out = await this.session.run(feed);
    return out[this.outputName].data;       // Float32Array(SR_OUT*SR_OUT)
  },

  /* adaptive frame pacing: keep ~24fps without melting the battery */
  observe(ms) {
    if (ms > 70) this.skipFrames = Math.min(4, this.skipFrames + 1);
    else if (ms < 30) this.skipFrames = Math.max(1, this.skipFrames - 1);
  },
};

/* ----------------------------- YCbCr math -------------------------------- */
/* BT.601: neural net only sees luma (Y); chroma comes from cheap bilinear. */
function rgbToYCbCr(r, g, b) {
  return [
    0.299 * r + 0.587 * g + 0.114 * b,
    -0.169 * r - 0.331 * g + 0.500 * b + 128,
    0.500 * r - 0.419 * g - 0.081 * b + 128,
  ];
}
const ycbcrToRgb = (y, cb, cr) => {
  const dCb = cb - 128, dCr = cr - 128;
  return [
    y + 1.402 * dCr,
    y - 0.344 * dCb - 0.714 * dCr,
    y + 1.772 * dCb,
  ];
};
const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v | 0);

/* --------------------- frame processor (tiled SR) ------------------------- */
/* Tile math:
 *   224x224 model input. Frames are tiled into up to a 2x2 grid of 224 crops:
 *   landscape 256x144  -> 2 horizontal tiles, vertically center-padded;
 *   portrait  144x256  -> 2 vertical tiles (Short), horizontally center-padded;
 *   smaller frames     -> single tile, edge-replicated padding.
 *   Seams blend with a smoothstep crossfade so the joins are invisible.
 */
const Processor = {
  low: null, lowCtx: null, chroma: null, chromaCtx: null, out: null, outCtx: null,
  W: 0, H: 0, sw: 0, sh: 0, yPlane: null, cbPlane: null, crPlane: null,
  tileInput: null, wsum: null, outBuf: null,

  init(W, H) {
    this.W = W; this.H = H;
    this.sw = W * 3; this.sh = H * 3;

    this.low = document.createElement('canvas');
    this.low.width = W; this.low.height = H;
    this.lowCtx = this.low.getContext('2d', { willReadFrequently: true });

    this.chroma = document.createElement('canvas');
    this.chroma.width = this.sw; this.chroma.height = this.sh;
    this.chromaCtx = this.chroma.getContext('2d', { willReadFrequently: true });

    // the visible canvas is sized to the 3x buffer; CSS stretches it full-bleed
    const out = $('#sr');
    out.width = this.sw; out.height = this.sh;
    this.outCtx = out.getContext('2d');
    this.out = out;

    this.yPlane = new Float32Array(W * H);
    this.cbPlane = new Float32Array(W * H);
    this.crPlane = new Float32Array(W * H);
    this.tileInput = new Float32Array(TILE * TILE);
    this.wsum = new Float32Array(this.sw * this.sh);
    this.outBuf = new Float32Array(this.sw * this.sh);
  },

  /* grab current frame -> planes; returns false when frame not ready */
  capture(video) {
    const { W, H } = this;
    this.lowCtx.drawImage(video, 0, 0, W, H);
    const px = this.lowCtx.getImageData(0, 0, W, H).data;
    const { yPlane, cbPlane, crPlane } = this;
    for (let p = 0, i = 0; i < px.length; i += 4, p++) {
      const [y, cb, cr] = rgbToYCbCr(px[i], px[i + 1], px[i + 2]);
      yPlane[p] = y; cbPlane[p] = cb; crPlane[p] = cr;
    }
    return true;
  },

  /* run the model on every tile and assemble the 3x luma buffer.
   * Tile grid: up to 2x2 tiles (224 crops). Frames smaller than 224 in a
   * dimension are center-padded (edge replicated); larger frames split into
   * two overlapping tiles with smooth crossfades on both seams — this covers
   * landscape (256x144) and portrait Shorts (144x256) alike. */
  async renderLuma() {
    const { W, H, sw, sh } = this;
    const padL = W > TILE ? 0 : ((TILE - W) >> 1);
    const padT = H > TILE ? 0 : ((TILE - H) >> 1);
    const nx = W > TILE ? 2 : 1;
    const ny = H > TILE ? 2 : 1;
    const xs = [0, W - TILE];
    const ys = [0, H - TILE];
    const s3 = (u) => u * u * (3 - 2 * u);        // smoothstep
    const hW = (t, cx) => {
      if (nx === 1) return 1;
      const L = 3 * (W - TILE);
      if (cx < L) return t === 0 ? 1 : 0;
      if (cx >= 672) return t === 0 ? 0 : 1;
      const u = s3((cx - L) / (672 - L));
      return t === 0 ? 1 - u : u;
    };
    const vW = (t, cy) => {
      if (ny === 1) return 1;
      const L = 3 * (H - TILE);
      if (cy < L) return t === 0 ? 1 : 0;
      if (cy >= 672) return t === 0 ? 0 : 1;
      const u = s3((cy - L) / (672 - L));
      return t === 0 ? 1 - u : u;
    };

    this.wsum.fill(0); this.outBuf.fill(0);
    for (let tiy = 0; tiy < ny; tiy++) {
      const y0 = ys[tiy];
      for (let tix = 0; tix < nx; tix++) {
        const x0 = xs[tix];
        const ti = this.tileInput;
        const yPlane = this.yPlane;
        for (let r = 0; r < TILE; r++) {
          const sy = Math.min(H - 1, Math.max(0, y0 + r - padT));
          const rowSrc = sy * W;
          const rowDst = r * TILE;
          for (let c = 0; c < TILE; c++) {
            const sx = Math.min(W - 1, Math.max(0, x0 + c - padL));
            ti[rowDst + c] = yPlane[rowSrc + sx];
          }
        }
        const out = await Engine.upscaleTile(ti);     // Float32 672x672

        const cx0 = 3 * (x0 - padL);
        const cy0 = 3 * (y0 - padT);
        for (let oy = 0; oy < SR_OUT; oy++) {
          const cy = cy0 + oy;
          if (cy < 0 || cy >= sh) continue;
          const fy = vW(tiy, cy);
          if (fy <= 0) continue;
          const rowOut = cy * sw;
          const rowMod = oy * SR_OUT;
          for (let ox = 0; ox < SR_OUT; ox++) {
            const cx = cx0 + ox;
            if (cx < 0 || cx >= sw) continue;
            const f = fy * hW(tix, cx);
            if (f <= 0) continue;
            const v = out[rowMod + ox] * f;
            const i = rowOut + cx;
            this.outBuf[i] += v;
            this.wsum[i] += f;
          }
        }
      }
    }
    if (nx > 1 || ny > 1) {
      for (let i = 0; i < sw * sh; i++) {
        if (this.wsum[i] > 0) this.outBuf[i] /= this.wsum[i];
      }
    }
  },

  /* merge SR luma with bilinear chroma -> visible canvas (single pass) */
  paint() {
    const { sw, sh } = this;
    this.chromaCtx.imageSmoothingEnabled = true;
    this.chromaCtx.imageSmoothingQuality = 'high';
    this.chromaCtx.drawImage(this.low, 0, 0, sw, sh);
    const c = this.chromaCtx.getImageData(0, 0, sw, sh).data;

    const outImg = this.outCtx.createImageData(sw, sh);
    const d = outImg.data;
    const yBuf = this.outBuf;
    // decode Cb/Cr up front, then one YCbCr->RGB pass in a tight loop
    for (let i = 0; i < sw * sh; i++) {
      const cb = -0.169 * c[4 * i] - 0.331 * c[4 * i + 1] + 0.500 * c[4 * i + 2] + 128;
      const cr = 0.500 * c[4 * i] - 0.419 * c[4 * i + 1] - 0.081 * c[4 * i + 2] + 128;
      const [r, g, b] = ycbcrToRgb(yBuf[i], cb, cr);
      d[4 * i] = clamp255(r); d[4 * i + 1] = clamp255(g);
      d[4 * i + 2] = clamp255(b); d[4 * i + 3] = 255;
    }
    this.outCtx.putImageData(outImg, 0, 0);
  },
};

/* ----------------------------- player state ------------------------------ */
const Player = {
  id: null, meta: null, cues: [], cueIdx: 0, ccOn: true,
  raf: 0, lastTs: 0, audioTimer: 0, engineOff: false, f: 0,

  async open(id, t) {
    this.close();
    this.id = id;
    $('#player').hidden = false;
    const boot = $('#bootmsg'); boot.hidden = false;
    try {
      const m = await (await fetch(`${API}/api/manifest/${id}`)).json();
      if (m.error) throw new Error(m.error);
      this.meta = m;

      $('#vititle').textContent = m.title;
      document.title = `${m.title} · Lumen`;

      // Shorts / portrait videos get a 9:16 stage + SHORTS badge
      $('#player').classList.toggle('short', m.aspect === 'portrait' || m.short);
      $('#badge').textContent = m.short ? 'SHORTS · 144p ⇢ SR×3' : '144p ⇢ SR×3';

      const v = $('#v');
      v.src = m.videoUrl;
      v.muted = true;
      await v.play().catch(() => {});

      if (m.audioUrl) {                      // split-stream case
        const a = $('#a');
        a.src = m.audioUrl;
        await a.play().catch(() => {});
        this.audioTimer = setInterval(() => {
          const a2 = $('#a');
          if (!isNaN(a2.duration) && Math.abs(a2.currentTime - v.currentTime) > 1.5)
            a2.currentTime = v.currentTime;
        }, 2000);
      }

      if (t) v.currentTime = t;
      this.renderMeta(m);
      this.loadCues();
      badge(`engine:${Engine.provider || 'loading'}`);
    } catch (e) {
      $('#meta').replaceChildren();
      const p = el('div', 'empty', `Could not load video. ${e.message}`);
      $('#meta').appendChild(p);
      badge(e.message);
    } finally {
      boot.hidden = true;
    }
  },

  async renderMeta(m) {
    const box = $('#meta');
    box.replaceChildren();

    box.appendChild(el('div', 'mtitle', m.title));
    const stats = [fmtViews(m.viewCount), m.uploadDate,
      `${m.short ? 'Shorts' : m.height + 'p'} · ${m.fps || 30}fps · itag ${m.formatId}`].filter(Boolean);
    box.appendChild(el('div', 'vstats', stats.join(' · ')));

    // action row — like / save / share
    const ar = el('div', 'actorow');
    const like = el('button', 'act');
    like.innerHTML = svg(ICON.like) + `<span>${fmtViews(m.likeCount) || ''}</span>`;
    like.onclick = () => {
      const on = like.dataset.on === '1';
      like.dataset.on = on ? '' : '1';
      like.style.color = on ? '' : 'var(--blue)';
      like.style.background = on ? '' : 'rgba(62,166,255,.12)';
    };
    const save = el('button', 'act');
    const saved = getSaved().some((x) => x.id === m.id);
    save.innerHTML = svg(saved ? ICON.saveFilled : ICON.save);
    if (saved) save.style.color = 'var(--blue)';
    save.onclick = () => {
      saveVideo({ id: m.id, title: m.title, author: m.author, thumb: m.thumb || `https://i.ytimg.com/vi/${m.id}/hqdefault.jpg`, views: m.viewCount });
      save.style.color = 'var(--blue)';
      toast('Saved to Watch later');
    };
    const share = el('button', 'act');
    share.innerHTML = svg(ICON.share);
    share.onclick = () => {
      const url = `${location.origin}/#/watch?v=${m.id}`;
      if (navigator.share) navigator.share({ title: m.title, url }).catch(() => {});
      else if (navigator.clipboard) navigator.clipboard.writeText(url).then(() => toast('Link copied')).catch(() => {});
    };
    ar.append(like, save, share);
    box.appendChild(ar);

    // channel row + subscribe
    const ch = el('div', 'channel');
    ch.appendChild(avatarEl(m.author || '?'));
    const info = el('div');
    info.appendChild(el('div', 'cname', m.author || 'Unknown'));
    info.appendChild(el('div', 'csubs', m.authorId ? 'Channel' : ''));
    ch.appendChild(info);
    const sub = el('button', 'subbtn', 'Subscribe');
    const isSub = getSubs().some((s) => s.id === m.authorId);
    if (isSub) { sub.classList.add('on'); sub.textContent = 'Subscribed'; }
    sub.onclick = () => {
      if (!m.authorId) return;
      if (getSubs().some((s) => s.id === m.authorId)) {
        localStorage.lumenSubs = JSON.stringify(getSubs().filter((s) => s.id !== m.authorId));
        sub.classList.remove('on'); sub.textContent = 'Subscribe'; toast('Unsubscribed');
      } else {
        saveSub(m.authorId, m.author);
        sub.classList.add('on'); sub.textContent = 'Subscribed'; toast('Subscribed');
      }
    };
    ch.appendChild(sub);
    box.appendChild(ch);

    if (m.description) box.appendChild(el('div', 'desc', m.description));
    if (m.tags && m.tags.length) {
      const row = el('div', 'tagrow');
      m.tags.slice(0, 8).forEach((t) => row.appendChild(el('button', 'tag', '#' + t)));
      box.appendChild(row);
    }

    // comments
    const chd = el('div', 'commenthead', 'Comments');
    chd.id = 'commenthead';
    box.appendChild(chd);
    const cBox = el('div', 'comments');
    fetch(`${API}/api/comments/${m.id}?n=20`)
      .then((r) => r.json())
      .then((j) => {
        const list = j.comments || [];
        if (!list.length) { cBox.appendChild(el('div', 'empty', 'Comments unavailable.')); return; }
        list.forEach((c) => {
          const d = el('div', 'comment');
          d.appendChild(avatarEl(c.author));
          const b = el('div', 'cb');
          b.appendChild(el('div', 'cm', `${c.author} · ${c.timeText}`));
          b.appendChild(el('div', 'ct', c.text));
          d.appendChild(b);
          cBox.appendChild(d);
        });
      })
      .catch(() => cBox.appendChild(el('div', 'empty', 'Comments unavailable.')));
    box.appendChild(cBox);

    // related
    box.appendChild(el('h2', 'section', 'Recommended'));
    const rel = el('div', 'rellist');
    try {
      const j = await (await fetch(`${API}/api/related/${m.id}`)).json();
      (j.entries || []).slice(0, 12).forEach((r) => {
        const a = el('a', 'relrow');
        a.href = `#/watch?v=${r.id}`;
        const w = el('div', 'thumb');
        const img = el('img'); img.loading = 'lazy'; img.src = r.thumb; img.alt = '';
        w.appendChild(img);
        if (r.durationMs && !r.short) w.appendChild(el('span', 'dur', fmtTime(r.durationMs)));
        a.appendChild(w);
        const t = el('div', 'vtext');
        t.appendChild(el('h3', '', r.title));
        t.appendChild(el('p', '', `${r.author || ''} · ${fmtViews(r.views)} views`));
        a.appendChild(t);
        rel.appendChild(a);
      });
      if (!j.entries?.length) rel.appendChild(el('div', 'empty', 'No related videos found.'));
    } catch { rel.appendChild(el('div', 'empty', 'Could not load recommendations.')); }
    box.appendChild(rel);
  },

  async loadCues() {
    try {
      const j = await (await fetch(`${API}/api/subs/${this.id}?lang=en`)).json();
      this.cues = j.cues || [];
    } catch { this.cues = []; }
  },

  close() {
    if (this.raf) cancelAnimationFrame(this.raf);
    clearInterval(this.audioTimer);
    $('#v').pause(); $('#v').removeAttribute('src'); $('#a').pause(); $('#a').removeAttribute('src');
    this.cues = [];
    $('#player').hidden = true;
    $('#player').classList.remove('short');
  },
};

/* ----------------------------- render loop ------------------------------- */
function startLoop() {
  cancelAnimationFrame(Player.raf);
  const v = $('#v'), out = $('#sr');
  const drawPassthrough = () => {
    // graceful fallback: sharp bilinear from 144p source
    const c = out.getContext('2d');
    if (v.videoWidth) { out.width = v.videoWidth; out.height = v.videoHeight; }
    c.imageSmoothingQuality = 'high';
    c.drawImage(v, 0, 0, out.width, out.height);
  };

  const tick = async (ts) => {
    if (!v.paused && !v.ended && !document.hidden && !Player.engineOff) {
      if (Engine.ready) {
        if ((Player.f++ % Engine.skipFrames) === 0 && v.videoWidth) {
          if (!Processor.W ||
              Processor.W !== Math.round(v.videoWidth) ||
              Processor.H !== Math.round(v.videoHeight)) {
            Processor.init(Math.round(v.videoWidth) || 256,
                           Math.round(v.videoHeight) || 144);
          }
          try {
            const t0 = performance.now();
            if (Processor.capture(v)) await Processor.renderLuma();
            Processor.paint();
            Engine.observe(performance.now() - t0);
          } catch (e) { Player.engineOff = true; drawPassthrough(); }
        }
      } else if (v.videoWidth) { drawPassthrough(); }
    }
    Player.raf = requestAnimationFrame(tick);
  };
  Player.raf = requestAnimationFrame(tick);
}

/* ----------------------------- playback UI -------------------------------- */
function wirePlayer() {
  const v = $('#v'), a = $('#a'), pl = $('#play'), tm = $('#time');
  const pad = (s) => String(s | 0).padStart(2, '0');
  const fmt = (s) => `${pad(s / 60)}:${pad(s % 60)}`;

  v.addEventListener('timeupdate', () => {
    tm.textContent = `${fmt(v.currentTime)} / ${fmt(v.duration || 0)}`;
    const T = v.currentTime;
    let from = Player.cueIdx;
    const cues = Player.cues;
    while (from < cues.length && cues[from].end < T) from++;
    while (from > 0 && cues[from - 1].start > T) from--;
    Player.cueIdx = from;
    const sub = $('#subtitle span');
    const hit = from < cues.length && cues[from].start <= T && cues[from].end >= T;
    if (hit && Player.ccOn) sub.textContent = cues[from].text;
    else sub.textContent = '';
  });
  v.addEventListener('seeking', () => { if (!isNaN(a.duration)) a.currentTime = v.currentTime; });

  // thin seekable progress rail at the very bottom of the stage
  const prog = el('div', 'progress');
  const fill = el('div', 'progfill');
  prog.appendChild(fill);
  $('#stage').appendChild(prog);
  const syncProg = () => {
    const d = v.duration || 0;
    fill.style.width = (d ? (v.currentTime / d) * 100 : 0) + '%';
  };
  v.addEventListener('timeupdate', syncProg);
  v.addEventListener('loadedmetadata', syncProg);
  prog.addEventListener('click', (e) => {
    if (!v.duration) return;
    const r = prog.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width) * v.duration;
    v.currentTime = Math.max(0, Math.min(t, v.duration - 0.1));
    a.currentTime = v.currentTime;     // keep split audio in step
  });
  const setPlayIcon = (playing) => { pl.innerHTML = svg(playing ? ICON.pause : ICON.play, 24, 24); };
  v.addEventListener('play', () => { if (a.paused) a.play().catch(() => {}); setPlayIcon(true); startLoop(); });
  v.addEventListener('pause', () => { cancelAnimationFrame(Player.raf); a.pause(); setPlayIcon(false); });

  pl.onclick = () => (v.paused ? v.play() : v.pause());
  $('#close').onclick = () => Player.close();
  $('#back').onclick = () => { if (history.length > 1) history.back(); else location.hash = '#/'; };
  $('#cc').addEventListener('change', (e) => { Player.ccOn = e.target.checked; });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelAnimationFrame(Player.raf);
    else if (!v.paused) startLoop();
  });

  // Shorts right rail
  const rlike = $('#rlike');
  rlike.onclick = () => {
    const on = rlike.dataset.on === '1';
    rlike.dataset.on = on ? '' : '1';
    rlike.style.color = on ? '#fff' : 'var(--blue)';
  };
  $('#rdislike').onclick = (e) => {
    const on = e.currentTarget.dataset.on === '1';
    e.currentTarget.dataset.on = on ? '' : '1';
    e.currentTarget.style.color = on ? '#fff' : 'var(--blue)';
  };
  $('#rcomment').onclick = () => {
    const c = $('#commenthead');
    if (c) c.scrollIntoView({ behavior: 'smooth', block: 'start' });
    else if (Player.id) location.hash = `#/watch?v=${Player.id}`;
  };
  $('#rshare').onclick = () => {
    const m = Player.meta; if (!m) return;
    const url = `${location.origin}/#/watch?v=${m.id}`;
    if (navigator.share) navigator.share({ title: m.title, url }).catch(() => {});
    else if (navigator.clipboard) navigator.clipboard.writeText(url).then(() => toast('Link copied')).catch(() => {});
  };
}

/* ----------------------------- SPA views --------------------------------- */
function esc(s) { return String(s ?? '').replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c])); }

const fmtViews = (n) => {
  if (n == null) return '';
  return n >= 1e6 ? (n / 1e6).toFixed(1) + 'M'
    : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(n);
};
const fmtTime = (ms) => {
  const s = Math.round((ms || 0) / 1000);
  const p = (x) => String(x).padStart(2, '0');
  return s >= 3600 ? `${p(s / 3600)}:${p((s % 3600) / 60)}:${p(s % 60)}`
    : `${(s / 60) | 0}:${p(s % 60)}`;
};

/* ------------------------- Material-style icons ------------------------- */
const ICON = {
  play: 'M8 5v14l11-7z',
  pause: 'M6 5h4v14H6zM14 5h4v14h-4z',
  like: 'M1 21h4V9H1v12zM23 10c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z',
  dislike: 'M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.58-6.59c.37-.36.59-.86.59-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z',
  share: 'M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z',
  save: 'M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z',
  saveFilled: 'M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z',
  chevUp: 'M7.41 15.41 12 10.83l4.59 4.58L18 14l-6-6-6 6z',
};
const svg = (d, w = 20, h = 20) =>
  `<svg viewBox="0 0 24 24" width="${w}" height="${h}" style="display:inline-block"><path fill="currentColor" d="${d}"/></svg>`;

const AV_COLORS = ['#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399', '#2dd4bf', '#60a5fa', '#818cf8', '#c084fc', '#f472b6'];
const avatarEl = (name) => {
  const c = el('span', 'avatar');
  const s = String(name || '?');
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  c.style.background = AV_COLORS[h % AV_COLORS.length];
  c.textContent = (s[0] || '?').toUpperCase();
  return c;
};

let toastTimer = 0;
function toast(msg) {
  let t = $('#toast');
  if (!t) { t = el('div', 'toast', ''); document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}

const CHIPS = ['All', 'Music', 'Gaming', 'Vlogs', 'Comedy', 'Tech', 'Mr Beast', 'Lo-Fi'];

function card(r) {
  const a = el('a', 'card');
  a.href = `#/watch?v=${r.id}`;
  if (r.short) a.classList.add('card-short');

  const w = el('div', 'thumb');
  const img = el('img'); img.loading = 'lazy'; img.alt = '';
  if (r.short) {
    img.src = `https://i.ytimg.com/vi/${r.id}/oar2.jpg`;  // real vertical Shorts art
    img.onerror = () => { img.onerror = null; img.src = r.thumb || ''; };  // fall back to landscape
  } else {
    img.src = r.thumb;
  }
  w.appendChild(img);
  if (r.live) w.appendChild(el('span', 'livebadge', 'LIVE'));
  else if (r.durationMs && !r.short) w.appendChild(el('span', 'dur', fmtTime(r.durationMs)));
  a.appendChild(w);

  const row = el('div', 'vrow');
  row.appendChild(avatarEl(r.author || '?'));
  const t = el('div', 'vtext');
  t.appendChild(el('h3', '', r.title));
  const meta = [];
  if (r.author) meta.push(r.author);
  if (r.views != null) meta.push(`${fmtViews(r.views)} views`);
  if (r.durationMs && !r.live) meta.push(fmtTime(r.durationMs));
  t.appendChild(el('p', '', meta.join(' · ')));
  row.appendChild(t);
  a.appendChild(row);
  return a;
}

async function homeView() {
  const box = $('#view');
  box.replaceChildren();

  const chips = el('div', 'chips');
  CHIPS.forEach((c, i) => {
    const b = el('button', 'chip', c);
    if (i === 0) b.classList.add('on');
    b.onclick = () => { if (c !== 'All') { location.hash = `#/search?q=${encodeURIComponent(c)}`; } };
    chips.appendChild(b);
  });
  box.appendChild(chips);

  const status = el('div', 'empty', 'Loading your feed…');
  box.appendChild(status);

  try {
    const [home, sh] = await Promise.all([
      fetch(`${API}/api/home`).then((r) => r.json()),
      fetch(`${API}/api/shorts`).then((r) => r.json()),
    ]);
    status.remove();

    if ((sh.entries || []).length) {
      const sb = el('div', 'shortsbox');
      sb.appendChild(el('h2', '', 'Shorts'));
      const shelf = el('div', 'shortshelf');
      sh.entries.forEach((c) => shelf.appendChild(card(c)));
      sb.appendChild(shelf);
      const more = el('a', 'seemore', 'See all Shorts ›');
      more.href = '#/shorts';
      sb.appendChild(more);
      box.appendChild(sb);
    }

    // each home section gets its own heading + grid (structured, YouTube-like)
    const sections = home.sections || [];
    let total = 0;
    for (const sec of sections) {
      const entries = sec.entries || [];
      if (!entries.length) continue;
      total += entries.length;
      box.appendChild(el('h2', 'section', sec.title));
      const g = el('div', 'grid');
      entries.forEach((c) => g.appendChild(card(c)));
      box.appendChild(g);
    }
    if (!total && !(sh.entries || []).length) {
      box.appendChild(el('div', 'empty', 'No recommendations right now — try a search from the top bar.'));
    }
  } catch (e) {
    status.textContent = 'Could not load the feed. ' + e.message;
  }
}

async function shortsView() {
  const box = $('#view');
  box.replaceChildren();
  box.appendChild(el('div', 'empty', 'Loading Shorts…'));
  try {
    const j = await (await fetch(`${API}/api/shorts`)).json();
    box.replaceChildren();
    const g = el('div', 'grid');
    (j.entries || []).forEach((r) => g.appendChild(card(r)));
    if (!g.childElementCount) box.appendChild(el('div', 'empty', 'No Shorts available right now.'));
    else box.appendChild(g);
  } catch (e) { box.replaceChildren(el('div', 'empty', e.message)); }
}

async function searchView(q) {
  const box = $('#view');
  box.replaceChildren();
  const bar = el('div', 'searchbar');
  const inp = el('input');
  inp.type = 'search'; inp.placeholder = 'Search YouTube'; inp.value = q || '';
  inp.setAttribute('autofocus', '');
  const cancel = el('button', 'cancel', 'Cancel');
  cancel.onclick = () => { if (history.length > 1) history.back(); else location.hash = '#/'; };
  bar.appendChild(inp); bar.appendChild(cancel);
  box.appendChild(bar);
  const go = () => {
    const v = inp.value.trim();
    if (v) location.hash = `#/search?q=${encodeURIComponent(v)}`;
    else inp.focus();
  };
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  if (!q) {
    inp.focus();
    const sug = el('div', 'rows');
    CHIPS.slice(1).forEach((c) => {
      const b = el('a', 'rowcard');
      b.href = `#/search?q=${encodeURIComponent(c)}`;
      b.appendChild(avatarEl(c));
      b.appendChild(el('span', 'lbl', c));
      sug.appendChild(b);
    });
    box.appendChild(sug);
    return;
  }
  const status = el('div', 'empty', 'Searching…');
  box.appendChild(status);
  try {
    const j = await (await fetch(`${API}/api/search?q=${encodeURIComponent(q)}`)).json();
    box.replaceChildren(bar);
    if (!j.entries?.length) { box.appendChild(el('div', 'empty', `No results for “${esc(q)}”.`)); return; }
    const g = el('div', 'grid');
    j.entries.forEach((r) => g.appendChild(card(r)));
    box.appendChild(g);
  } catch (e) { box.replaceChildren(bar); status.textContent = e.message; box.appendChild(status); }
}

function subsView() {
  const box = $('#view');
  box.replaceChildren();
  box.appendChild(el('h2', 'section', 'Subscriptions'));
  const subs = getSubs();
  if (!subs.length) {
    box.appendChild(el('div', 'empty', 'No subscriptions yet.\nTap the red SUBSCRIBE button on any channel to keep up here.'));
    return;
  }
  const rows = el('div', 'rows');
  subs.forEach((s) => {
    const r = el('a', 'rowcard');
    r.href = `#/channel?id=${s.id}`;
    r.appendChild(avatarEl(s.name));
    r.appendChild(el('span', 'lbl', s.name));
    rows.appendChild(r);
  });
  box.appendChild(rows);
}

function youView() {
  const box = $('#view');
  box.replaceChildren();
  box.appendChild(el('h2', 'section', 'Watch later'));
  const favs = getSaved();
  if (!favs.length) {
    box.appendChild(el('div', 'empty', 'Nothing saved yet.\nUse the save button on any video and it appears here.'));
    return;
  }
  const g = el('div', 'grid');
  favs.forEach((f) => g.appendChild(card(f)));
  box.appendChild(g);
}

async function channelView(id) {
  const box = $('#view');
  box.replaceChildren();
  box.appendChild(el('div', 'empty', 'Loading channel…'));
  try {
    const j = await (await fetch(`${API}/api/channel/${id}`)).json();
    box.replaceChildren();
    if (j.error) { box.appendChild(el('div', 'empty', j.error)); return; }
    box.appendChild(el('h2', 'section', j.name));
    const g = el('div', 'grid');
    (j.entries || []).forEach((r) => g.appendChild(card(r)));
    box.appendChild(g);
  } catch (e) { box.replaceChildren(el('div', 'empty', e.message)); }
}

/* ----------------------------- routing ----------------------------------- */
const NAVMAP = { '/': 'home', '/shorts': 'shorts', '/saved': 'saved', '/you': 'you', '/subs': 'subs' };
function setNav(path) {
  const key = NAVMAP[path] || (path.startsWith('/watch') || path.startsWith('/search') ? 'home' : (path === '/channel' ? 'subs' : ''));
  document.querySelectorAll('.navbtn').forEach((b) => b.classList.toggle('on', b.dataset.nav === key));
}
function route() {
  const h = location.hash || '#/';
  const [path, query] = h.slice(1).split('?');
  const q = new URLSearchParams(query || '');
  $('#player').hidden = true;
  const view = $('#view');
  document.body.classList.toggle('watch-mode', path === '/watch');
  view.hidden = (path === '/watch');   // on watch, the media layer replaces the feed chrome
  view.replaceChildren();              // drop any stale content from the previous route

  if (path === '/watch') {
    const id = q.get('v');
    if (id) Player.open(id, parseFloat(q.get('t') || '0') || 0);
  } else if (path === '/search') {
    const qq = q.get('q');
    searchView(qq || '');
  } else if (path === '/shorts') {
    shortsView();
  } else if (path === '/channel') {
    channelView(q.get('id'), path);
  } else if (path === '/subs') {
    subsView();
  } else if (path === '/you' || path === '/saved') {
    youView();
  } else {
    homeView();
  }
  setNav(path);
}

/* ----------------------------- persistence -------------------------------- */
const getSaved = () => { try { return JSON.parse(localStorage.lumenSaved || '[]'); } catch { return []; } };
const saveVideo = (v) => {
  const s = getSaved().filter((x) => x.id !== v.id);
  s.unshift({ id: v.id, title: v.title, author: v.author, thumb: v.thumb, views: v.views });
  localStorage.lumenSaved = JSON.stringify(s.slice(0, 60));
};
const getSubs = () => { try { return JSON.parse(localStorage.lumenSubs || '[]'); } catch { return []; } };
const saveSub = (id, name) => {
  if (!id) return;
  const s = getSubs().filter((x) => x.id !== id);
  s.unshift({ id, name });
  localStorage.lumenSubs = JSON.stringify(s.slice(0, 100));
};

/* ----------------------------- glue --------------------------------------- */
function badge(txt) {
  const b = $('#badge');
  if (!b) return;
  if (txt) { b.textContent = txt; return; }
  b.textContent = `144p ⇢ SR×3${Engine.provider ? ' · ' + Engine.provider : ' · engine…'}`;
}

function init() {
  badge();
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
  // warm the neural engine off the critical path (idle callback)
  const warm = () => Engine.load();
  if ('requestIdleCallback' in window) requestIdleCallback(warm, { timeout: 4000 });
  else setTimeout(warm, 1800);

  // top bar
  $('#logo').addEventListener('click', () => { location.hash = '#/'; });
  $('#topsearch').addEventListener('click', () => { location.hash = '#/search'; });
  $('#topdots').addEventListener('click', () => { location.hash = '#/you'; });

  // bottom navigation — buttons carry data-nav (home/shorts/saved/subs/you);
  // map each to its hash route. "saved" routes to /saved which youView() handles.
  const NAVROUTES = { home: '/', shorts: '/shorts', saved: '/saved', subs: '/subs', you: '/you' };
  $('#bottomnav').addEventListener('click', (e) => {
    const b = e.target.closest('.navbtn');
    if (!b) return;
    const next = NAVROUTES[b.dataset.nav];
    if (!next) return;
    if (location.hash === '#' + next) return;   // already there — don't re-render
    location.hash = next;
  });

  // simple hash navigation for any .card link
  document.addEventListener('click', (e) => {
    const c = e.target.closest('.card');
    if (c && c.getAttribute('href')?.startsWith('#')) {
      e.preventDefault();
      location.hash = c.getAttribute('href');
    }
  });

  wirePlayer();
  window.addEventListener('hashchange', route);
  route();
}
document.addEventListener('DOMContentLoaded', init);