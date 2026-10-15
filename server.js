#!/usr/bin/env node
/**
 * LUMEN — Compute-Over-Network YouTube proxy (Node.js/Express)
 * ------------------------------------------------------------------
 * Strategy: YouTube transports *minimum* bytes to the phone.
 *   1. `/api/manifest/:id`  -> yt-dlp format selection: force the lowest
 *                              bandwidth video (144p mp4 if the source has one,
 *                              else the absolute lowest-height stream) plus the
 *                              lowest-bitrate audio (48 kbps m4a), subtitles,
 *                              and full video metadata.
 *   2. `/api/stream/:id`    -> relays ONLY that 144p video track with HTTP
 *                              Range passthrough (seeking stays cheap).
 *   3. `/api/audio/:id`     -> relays the lowest-bitrate audio track.
 *   4. The phone then *computes* the missing detail locally (WebGPU SR).
 *
 * Also Invidious-style endpoints: search, video details, related, comments,
 * subtitles (parsed VTT -> JSON cue tokens), plus the static PWA shell.
 * Runs plain HTTP (LAN testing) and HTTPS (PWA / iPhone over Tailscale).
 */
'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const { spawn } = require('child_process');
const { Readable } = require('stream');
const express = require('express');

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
const PORT = parseInt(process.env.PORT || '8080', 10);
const TLS_PORT = parseInt(process.env.TLSPORT || '8443', 10);
const YTDLP = process.env.YTDLP || 'yt-dlp';
const PUBLIC_DIR = path.join(__dirname, 'public');
const CERTS_DIR = path.join(__dirname, 'certs');
const PRECOMP = path.join(__dirname, '.precomp');   // brotli/gzip mirrors of static files

/* ---- static transfer compression (neural engine is the big one) ---- */
const MIME = {
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
};
const COMPRESSIBLE = /\.(wasm|onnx|js|mjs|css|html|json|xml|svg|txt)$/i;

function compressTo(src, outBase, kind) {
  return new Promise((resolve, reject) => {
    const rs = fs.createReadStream(src);
    const fin = `${outBase}.${kind}`;
    const tmp = `${fin}.tmp`;                       // atomic publish: no partial files served
    const ws = fs.createWriteStream(tmp);
    rs.on('error', reject); ws.on('error', reject);
    const t = kind === 'br'
      ? zlib.createBrotliCompress({ params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 9 } })
      : zlib.createGzip({ level: 6 });
    ws.on('close', () => { fs.renameSync(tmp, fin); resolve(); });
    rs.pipe(t).pipe(ws);
  });
}

/* Walk public/ and write `.br` + `.gz` mirrors into .precomp (background). */
async function precompressStatic() {
  const walk = [];
  (function rec(dir, rel) {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, ent.name);
      const r = rel ? `${rel}/${ent.name}` : ent.name;
      if (ent.isDirectory()) rec(full, r);
      else if (ent.isFile()) walk.push({ full, r });
    }
  })(PUBLIC_DIR, '');
  let n = 0;
  for (const { full, r } of walk) {
    if (!COMPRESSIBLE.test(full)) continue;              // png etc. already dense
    const st = fs.statSync(full);
    if (st.size < 1024) continue;
    const base = path.join(PRECOMP, r);
    const cur = { br: base + '.br', gz: base + '.gz' };
    fs.mkdirSync(path.dirname(base), { recursive: true });
    const stale = (p) => !fs.existsSync(p) || fs.statSync(p).size <= 0 ||
                        fs.statSync(p).mtimeMs < st.mtimeMs;
    if (stale(cur.br)) { await compressTo(full, base, 'br'); n++; }
    if (stale(cur.gz)) { await compressTo(full, base, 'gz'); n++; }
  }
  console.log(`[lumen] precompressed static assets (${n} new files)`);
}

const UA = ('Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 ' +
            '(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36');
const INV_SOURCES = [
  'https://inv.nadeko.net',
  'https://yewtu.be',
  'https://invidious.nerdvpn.de',
  'https://iv.melmac.space',
  'https://invidious.perennialte.ch',
  'https://invidious.f5.si',
];

/* Fetch Invidious public API with a browser UA (many instances block plain fetch) */
async function invidiousGet(path, timeoutMs = 4500, maxTries = INV_SOURCES.length) {
  const hdrs = {
    'User-Agent': UA,
    'Accept': 'application/json',
    'Accept-Language': 'en',
  };
  for (let i = 0; i < Math.min(maxTries, INV_SOURCES.length); i++) {
    const base = INV_SOURCES[i];
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fetch(`${base}${path}`, { headers: hdrs, signal: ctl.signal });
      if (!r.ok) continue;
      const j = await r.json();
      if (j && j.error && !j.comments) continue;
      return { source: base, json: j };
    } catch { /* next instance */ }
    finally { clearTimeout(t); }
  }
  return { source: null, json: null };
}

/* In-memory info cache: id -> {json, at} (yt-dlp -J is slow; cache it) */
const infoCache = new Map();
const INFO_TTL_MS = 10 * 60 * 1000;
/* Channel flat-playlists are big; cache for 20 min */
const chanCache = new Map();
/* Channel /shorts feeds (small), home feed sections, aggregated shorts */
const chanShortCache = new Map();
const homeCache = new Map();
const HOME_TTL_MS = 10 * 60 * 1000;

/* ---- curated shorts-heavy channels for the home "Shorts" shelf ---- */
const SHORTS_CHANNELS = [
  { id: 'UCX6OQ3DkcsbYNE6H8uQQuVA', name: 'MrBeast' },
  { id: 'UCGt7X90Au6BV8rf49BiM6Dg', name: 'Ray William Johnson' },
  { id: 'UCrPseYLGpNygVi34QpGNqpA', name: 'Ludwig' },
  { id: 'UCuAXFkgsw1L7xaCfnd5JJOw', name: 'Rick Astley' },
];
const HOME_SECTIONS = [
  { title: 'Trending now', q: 'trending music videos' },
  { title: 'Going viral', q: 'viral funny video' },
  { title: 'Fresh drops', q: 'new music video' },
];

/* flat yt-dlp playlist entries -> grid card shape (thumbnail from ytimg) */
function flatCards(entries, forceShort = false) {
  return (entries || [])
    .filter((e) => e && e.id && e.title)
    .map((e) => ({
      id: e.id,
      title: e.title,
      author: e.channel || e.uploader || null,
      authorId: e.channel_id || null,
      durationMs: (e.duration || 0) * 1000,
      views: e.view_count ?? null,
      live: !!e.is_live,
      short: forceShort,
      thumb: forceShort
        ? `https://i.ytimg.com/vi/${e.id}/oar2.jpg`   // real vertical Shorts art (1080x1920)
        : `https://i.ytimg.com/vi/${e.id}/mqdefault.jpg`,
    }));
}

async function cachedChannel(id) {
  const hit = chanCache.get(id);
  if (hit && Date.now() - hit.at < 20 * 60 * 1000) return hit.json;
  const json = await ytdlpJson([
    '-J', '--no-warnings', '--flat-playlist', '--socket-timeout', '15',
    `https://www.youtube.com/channel/${id}/videos`,
  ]);
  chanCache.set(id, { json, at: Date.now() });
  return json;
}

/* Tiny async mutex so we never fire 20 yt-dlp processes at once */
let dlpBusy = false;
const dlpQueue = [];

function dlpLock() {
  return new Promise((res) => {
    if (!dlpBusy) { dlpBusy = true; return res(); }
    dlpQueue.push(res);
  });
}
function dlpUnlock() {
  if (dlpQueue.length) { const nxt = dlpQueue.shift(); nxt(); }
  else dlpBusy = false;
}

/* ------------------------------------------------------------------ */
/* yt-dlp helpers                                                      */
/* ------------------------------------------------------------------ */
/** Run yt-dlp, collect stdout, resolve JSON. Throws with stderr on error. */
function ytdlpJson(args, { timeoutMs = 30000, maxBytes = 8 << 20 } = {}) {
  return new Promise(async (resolve, reject) => {
    await dlpLock();
    let out = [];
    let bytes = 0;
    const child = spawn(YTDLP, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, LANG: 'en_US.UTF-8' },
    });
    const timer = setTimeout(() => child.kill('SIGKILL'), timeoutMs);
    child.stdout.on('data', (d) => {
      bytes += d.length;
      if (bytes > maxBytes) { child.kill('SIGKILL'); return; }
      out.push(d);
    });
    let err = [];
    child.stderr.on('data', (d) => {
      const s = String(d);
      if (!/WARNING/i.test(s)) err.push(s);      // warnings are noise
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      dlpUnlock();
      if (code === 0 || (code === 2 && bytes > 0)) {
        try { resolve(JSON.parse(Buffer.concat(out).toString('utf8'))); }
        catch (e) { reject(new Error('yt-dlp returned non-JSON output')); }
      } else {
        reject(new Error(`yt-dlp exit ${code}: ${err.join('').slice(0, 400)}`));
      }
    });
    child.on('error', (e) => { clearTimeout(timer); dlpUnlock(); reject(e); });
  });
}

/** yt-dlp URL for a video id. */
const vidUrl = (id) => `https://www.youtube.com/watch?v=${id}`;
const isValidId = (id) => /^[\w-]{6,20}$/.test(id || '');

/** Fresh (uncached) full info blob for a video. */
async function fetchInfo(id) {
  const cached = infoCache.get(id);
  if (cached && Date.now() - cached.at < INFO_TTL_MS) return cached.json;
  // player_client=web,web_embedded gives the widest compatible format table.
  const json = await ytdlpJson([
    '-J', '--no-warnings', '--no-playlist', '--skip-download',
    '--socket-timeout', '15',
    '--extractor-args', 'youtube:player_client=default,web,web_embedded',
    vidUrl(id),
  ]);
  infoCache.set(id, { json, at: Date.now() });
  return json;
}

/* ------------------------------------------------------------------ */
/* Data-savings stream selection (the heart of Compute-Over-Network)   */
/* ------------------------------------------------------------------ */
function pickStreams(info) {
  const F = info.formats || [];
  const has = (f, k) => (f[k] && f[k].toLowerCase() !== 'none');

  // 1) True combined 144p (3gp/18 in old videos): zero extra request.
  const combined = F
    .filter((f) => has(f, 'vcodec') && has(f, 'acodec') && f.height > 0)
    .sort((a, b) => (a.height - b.height) ||
                     (a.width - b.width) ||
                     ((a.filesize || 0) - (b.filesize || 0)));
  const combined144 = combined.filter((f) => f.height <= 144).at(0);

  // 2) Lowest video-only stream (modern YouTube): we enforce <=144p first,
  //    then accept anything minimal (240p) so the app always has a stream.
  const vOnly = F
    .filter((f) => has(f, 'vcodec') && !has(f, 'acodec') && f.height > 0)
    .sort((a, b) => (a.height - b.height) || ((a.filesize || 0) - (b.filesize || 0)));
  const video = vOnly.find((f) => f.height <= 144) || vOnly.at(0) || combined.at(0);

  // 3) Lowest audio-only: prefer AAC (m4a) for iOS, then opus/vorbis.
  const aOnly = F
    .filter((f) => has(f, 'acodec') && !has(f, 'vcodec'))
    .sort((a, b) => (a.abr || 999) - (b.abr || 999) ||
                    ((a.filesize || 0) - (b.filesize || 0)));
  const audio = aOnly.find((f) => /mp4a/i.test(f.acodec)) || aOnly.at(0);

  return {
    video: video || null,
    audio: audio || null,
    combined144: combined144 || null,
  };
}

/* ------------------------------------------------------------------ */
/* Range-pass-through proxy for YouTube's CDN                          */
/* ------------------------------------------------------------------ */
function makeRelay(getUrl) {
  return async (req, res, next) => {
    const id = req.params.id;
    if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
    try {
      const url = await getUrl(id);
      const headers = { 'User-Agent': UA, Accept: '*/*' };
      if (req.headers.range) headers.Range = req.headers.range;
      // Node's global fetch follows googlevideo redirects natively.
      const upstream = await fetch(url, { headers, redirect: 'follow' });
      if (!upstream.ok && !(req.headers.range && upstream.status === 206)) {
        console.warn('relay upstream', upstream.status, url);
        return res.status(upstream.status > 499 ? 502 : 500).end();
      }
      res.status(upstream.status === 206 ? 206 : 200);
      const pass = ['content-type', 'content-length', 'accept-ranges',
                    'content-range', 'etag'];
      for (const k of pass) {
        const v = upstream.headers.get(k);
        if (v) res.setHeader(k, v);
      }
      if (!res.getHeader('accept-ranges')) res.setHeader('Accept-Ranges', 'bytes');
      const body = Readable.fromWeb(upstream.body);
      body.on('error', () => res.destroy());
      req.on('close', () => body.destroy());
      body.pipe(res);
    } catch (e) {
      console.error('relay error', e.message);
      if (!res.headersSent) res.status(502).json({ error: e.message });
      else res.end();
    }
  };
}

/* ------------------------------------------------------------------ */
/* Express app                                                         */
/* ------------------------------------------------------------------ */
const app = express();
app.disable('x-powered-by');
app.use((req, res, next) => {           // open CORS (Tailscale/LAN origins)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Range');
  if (req.method === 'OPTIONS') return res.end();
  next();
});

/* ---- static PWA shell, with smart caching ---- */
app.use((req, res, next) => {
  // cache-transparent for the shell; long-cache the immutable assets
  if (/^\/(models|vendor|icons)\//.test(req.path)) {
    res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
  } else if (req.path === '/' || req.path === '/index.html') {
    res.setHeader('Cache-Control', 'no-cache');
  }
  next();
});
/* Serve precompressed `.br`/`.gz` mirrors when the client can take them.
   Keeps the model + ORT WASM download (~30+ MB raw) to a fraction. */
app.use((req, res, next) => {
  res.setHeader('Vary', 'Accept-Encoding');
  if (req.method !== 'GET' && req.method !== 'HEAD') return next();
  const p = decodeURIComponent(req.path).replace(/^\/+/, '');
  if (!COMPRESSIBLE.test(p)) return next();
  const ae = req.headers['accept-encoding'] || '';
  /* try encodings in preference order; serve the first mirror that exists */
  for (const [enc, re] of [['br', /\bbr\b/i], ['gz', /\bgzip\b/i]]) {
    if (!re.test(ae)) continue;
    const candidate = path.join(PRECOMP, p) + '.' + enc;
    let ok = false;
    try { ok = fs.existsSync(candidate) && fs.statSync(candidate).size > 0; } catch { /* ignore */ }
    if (!ok) continue;
    res.sendFile(candidate, {
      acceptRanges: false,
      headers: {
        'Content-Encoding': enc,
        'Content-Type': MIME[path.extname(p)] || 'application/octet-stream',
      },
    });
    return;
  }
  next();
});
app.use(express.static(PUBLIC_DIR, { etag: true, index: 'index.html' }));

/* ---- manifest: one yt-dlp -J call -> everything the player needs ---- */
app.get('/api/manifest/:id', async (req, res) => {
  const id = req.params.id;
  if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
  try {
    const info = await fetchInfo(id);
    const { video, audio } = pickStreams(info);
    const has = (f, k) => f && f[k] && f[k].toLowerCase() !== 'none';
    const isCombined = !!(video && has(video, 'acodec') && has(video, 'vcodec'));

    const subs = {};
    const src = info.subtitles && Object.keys(info.subtitles).length
      ? info.subtitles
      : (info.automatic_captions || {});
    for (const [lang, tracks] of Object.entries(src)) {
      const t = tracks.find((x) => x.ext === 'vtt') || tracks[0];
      if (t) subs[lang] = { url: t.url, kind: info.subtitles?.[lang] ? 'manual' : 'auto' };
    }

    const thumbnail = (info.thumbnails || []).at(-1)?.url ||
                      `https://i.ytimg.com/vi/${id}/hqdefault.jpg`;

    const portrait = !!(info.width && info.height && info.height > info.width);
    const dur = info.duration || 0;
    const isShort = portrait && dur > 0 && dur <= 180;  // YouTube Shorts: vertical, ≤3 min

    res.json({
      id,
      title: info.title,
      author: info.channel || info.uploader || 'unknown',
      authorId: info.channel_id,
      description: (info.description || '').slice(0, 2000),
      durationMs: (info.duration || 0) * 1000,
      viewCount: info.view_count || 0,
      likeCount: info.like_count || 0,
      uploadDate: info.upload_date,
      tags: (info.tags || []).slice(0, 12),
      height: video ? (video.height || 144) : 144,
      width: video ? (video.width || 256) : 256,
      fps: video ? (video.fps || 24) : 24,
      formatId: video?.format_id || null,
      isCombined,
      videoUrl: `/api/stream/${id}`,
      audioUrl: isCombined ? null : `/api/audio/${id}`,
      thumb: thumbnail,
      subtitles: subs,
      aspect: portrait ? 'portrait' : 'landscape',
      short: isShort,
      live: !!info.is_live,
      ageLimited: !!info.age_limit,
    });
  } catch (e) {
    res.status(502).json({ error: `youtube fetch failed: ${e.message}`.slice(0, 300) });
  }
});

/* ---- forced-144p video track ---- */
app.get('/api/stream/:id', makeRelay(async (id) => {
  const info = await fetchInfo(id);
  const { video } = pickStreams(info);
  if (video && info.formats.some((f) => f.url === video.url && video.url)) {
    return video.url;
  }
  // Sign a fresh URL in case the cached one expired (signed URLs are short-lived)
  const url = (await ytdlpJson([
    '-f', video ? video.format_id : 'worst', '-g', '--no-warnings',
    vidUrl(id)]));
  return [].concat(url)[0];
}));

/* ---- lowest-bitrate audio track ---- */
app.get('/api/audio/:id', makeRelay(async (id) => {
  const info = await fetchInfo(id);
  const { audio } = pickStreams(info);
  if (audio && audio.url) return audio.url;
  const url = await ytdlpJson(['-f', 'worstaudio/ba', '-g', '--no-warnings', vidUrl(id)]);
  return [].concat(url)[0];
}));

/* ---- subtitles -> raw JSON cue tokens (the "semantic" data layer) ---- */
app.get('/api/subs/:id', async (req, res) => {
  const id = req.params.id;
  const want = (req.query.lang || 'en').toString().toLowerCase();
  if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
  try {
    const info = await fetchInfo(id);
    const table = info.subtitles && Object.keys(info.subtitles).length
      ? info.subtitles : (info.automatic_captions || {});
    let lang = want;
    if (!table[lang]) {
      lang = Object.keys(table).find((k) => k.startsWith(want + '-')) || Object.keys(table)[0];
    }
    const track = table[lang]?.find((t) => t.ext === 'vtt') || table[lang]?.[0];
    if (!track) return res.json({ lang, tracks: Object.keys(table), cues: [] });

    const r = await fetch(track.url);
    const vtt = await r.text();
    res.json({ lang, tracks: Object.keys(table), cues: parseVttCues(vtt) });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

function parseVttCues(vtt) {
  const cues = [];
  const re = /(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})[^\n]*\n([\s\S]*?)(?:\n{2,}|$)/g;
  const ts = (a, b, c, d) => (+a) * 3600 + (+b) * 60 + (+c) + (+d) / 1000;
  let m;
  while ((m = re.exec(vtt))) {
    const text = m[9].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&')
      .split('\n').map((s) => s.trim()).filter(Boolean).join(' ');
    cues.push({ start: ts(m[1], m[2], m[3], m[4]), end: ts(m[5], m[6], m[7], m[8]), text });
  }
  return cues;
}

/* ---- Invidious-style search ---- */
app.get('/api/search', async (req, res) => {
  const q = (req.query.q || '').toString().trim();
  if (!q) return res.status(400).json({ error: 'missing q' });
  try {
    // ytsearch<N> is yt-dlp's stable, no-bot search interface.
    const json = await ytdlpJson([
      '-J', '--no-warnings', '--flat-playlist', '--socket-timeout', '15',
      `ytsearch${Math.min(20, parseInt(req.query.n || '12', 10))}:${q}`,
    ]);
    const out = (json.entries || []).map((e) => ({
      id: e.id,
      title: e.title,
      author: e.channel || e.uploader || null,
      authorId: e.channel_id || null,
      durationMs: (e.duration || 0) * 1000,
      views: e.view_count ?? null,
      live: !!e.is_live,
      thumb: `https://i.ytimg.com/vi/${e.id}/mqdefault.jpg`,
    }));
    res.json({ query: q, entries: out });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

/* ---- home: YouTube-like grid feed (self-hosted search sections, cached) ---- */
app.get('/api/home', async (req, res) => {
  const hit = homeCache.get('feed');
  if (hit && Date.now() - hit.at < HOME_TTL_MS) return res.json(hit.json);
  const sections = [];
  let failures = 0;
  for (const s of HOME_SECTIONS) {
    try {
      const j = await ytdlpJson([
        '-J', '--no-warnings', '--flat-playlist', '--socket-timeout', '15',
        `ytsearch12:${s.q}`,
      ]);
      const entries = flatCards(j.entries);
      if (entries.length) sections.push({ title: s.title, entries });
    } catch { failures++; }
  }
  const out = { sections, failures };
  if (!sections.length && failures === HOME_SECTIONS.length)
    return res.status(502).json({ error: 'all home feed sources failed' });
  homeCache.set('feed', { json: out, at: Date.now() });
  res.json(out);
});

/* ---- popular Shorts shelf: aggregated from curated /shorts tabs ---- */
app.get('/api/shorts', async (req, res) => {
  const hit = homeCache.get('shorts');
  if (hit && Date.now() - hit.at < HOME_TTL_MS) return res.json(hit.json);
  const pool = [];
  for (const ch of SHORTS_CHANNELS) {
    try {
      const j = await ytdlpJson([
        '-J', '--no-warnings', '--flat-playlist', '--playlist-items', '1-12',
        '--socket-timeout', '15',
        `https://www.youtube.com/channel/${ch.id}/shorts`,
      ]);
      pool.push(...flatCards(j.entries, true));
    } catch { /* skip dead channel */ }
  }
  const seen = new Set();
  const out = { entries: pool.filter((e) => (seen.has(e.id) ? false : seen.add(e.id))).slice(0, 30) };
  homeCache.set('shorts', { json: out, at: Date.now() });
  res.json(out);
});

/* ---- a single channel's Shorts (for channel pages) ---- */
app.get('/api/channel/:id/shorts', async (req, res) => {
  const id = req.params.id;
  if (!/^[UC][\w-]{10,}$/.test(id || '')) return res.status(400).json({ error: 'bad channel id' });
  const hit = chanShortCache.get(id);
  if (hit && Date.now() - hit.at < HOME_TTL_MS)
    return res.json({ id, name: hit.name, entries: hit.entries });
  try {
    const json = await ytdlpJson([
      '-J', '--no-warnings', '--flat-playlist', '--playlist-items', '1-24',
      '--socket-timeout', '15',
      `https://www.youtube.com/channel/${id}/shorts`,
    ]);
    const out = { id, name: json.channel || json.title || id, entries: flatCards(json.entries, true) };
    chanShortCache.set(id, { ...out, at: Date.now() });
    res.json(out);
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

/* ---- full video metadata ---- */
app.get('/api/video/:id', async (req, res) => {
  const id = req.params.id;
  if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
  try {
    const info = await fetchInfo(id);
    res.json({
      id,
      title: info.title,
      author: info.channel || info.uploader,
      authorId: info.channel_id,
      thumb: `https://i.ytimg.com/vi/${id}/maxresdefault.jpg`,
      description: info.description || '',
      durationMs: (info.duration || 0) * 1000,
      viewCount: info.view_count || 0,
      likeCount: info.like_count || 0,
      dislikeCount: info.dislike_count ?? null,
      uploadDate: info.upload_date || null,
      tags: info.tags || [],
      categories: info.categories || [],
      chapters: (info.chapters || []).map((c) => ({ start: c.start_time, title: c.title, end: c.end_time })),
    });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

/* ---- related / recommended ----
 * Preferred: Invidious public API. Fallback: this server's own channel
 * uploads feed (yt-dlp), so related support never depends on a third party
 * being up. */
app.get('/api/related/:id', async (req, res) => {
  const id = req.params.id;
  if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
  const { source, json } = await invidiousGet(
    `/api/v1/videos/${id}?fields=recommendedVideos,title,author,authorId`,
    2800, 2);
  if (json && json.recommendedVideos) {
    const entries = json.recommendedVideos
      .filter((v) => v.videoId && v.videoId !== id)
      .map((v) => ({
        id: v.videoId, title: v.title, author: v.author,
        durationMs: (v.lengthSeconds || 0) * 1000,
        views: v.viewCount ?? null,
        thumb: `https://i.ytimg.com/vi/${v.videoId}/mqdefault.jpg`,
      }));
    return res.json({ source, entries });
  }
  // self-hosted fallback: same channel's latest uploads
  try {
    const info = await fetchInfo(id);
    if (info.channel_id) {
      const ch = await cachedChannel(info.channel_id);
      const entries = (ch.entries || []).filter((e) => e.id !== id).slice(0, 12)
        .map((e) => ({
          id: e.id, title: e.title, author: ch.channel || info.channel,
          durationMs: (e.duration || 0) * 1000, views: e.view_count ?? null,
          thumb: `https://i.ytimg.com/vi/${e.id}/mqdefault.jpg`,
        }));
      return res.json({ source: 'channel', entries });
    }
  } catch { /* fall through */ }
  res.json({ source: null, entries: [] });
});

/* ---- channel uploads (Invidious "subscriptions" style) ---- */
app.get('/api/channel/:id', async (req, res) => {
  const id = req.params.id;
  if (!/^[UC][\w-]{10,}$/.test(id || '')) return res.status(400).json({ error: 'bad channel id' });
  try {
    const json = await cachedChannel(id);
    const entries = (json.entries || []).map((e) => ({
      id: e.id, title: e.title,
      author: e.channel || json.channel || null,
      durationMs: (e.duration || 0) * 1000,
      views: e.view_count ?? null,
      thumb: `https://i.ytimg.com/vi/${e.id}/mqdefault.jpg`,
    }));
    res.json({ id, name: json.channel || id, entries });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

/* ---- comments (Invidious public API, best effort; never 5xx the UI) ---- */
app.get('/api/comments/:id', async (req, res) => {
  const id = req.params.id;
  if (!isValidId(id)) return res.status(400).json({ error: 'bad id' });
  const { source, json } = await invidiousGet(
    `/api/v1/comments/${id}?sort_by=top&fields=commentCount,comments`, 6000);
  const comments = Array.isArray(json?.comments) ? json.comments : [];
  res.json({
    source,
    count: comments.length,
    comments: comments.slice(0, Math.min(30, parseInt(req.query.n || '20', 10)))
      .map((c) => ({
        id: c.commentId || c.id, author: c.author, text: c.content || c.text,
        likeCount: c.likeCount || 0, timeText: c.timeText || '',
      })),
  });
});

/* ---- serve CA for iPhone trust bootstrap ---- */
app.get('/ca.crt', (req, res) => {
  const p = path.join(CERTS_DIR, 'ca.crt');
  if (!fs.existsSync(p)) return res.status(404).end();
  res.setHeader('Content-Type', 'application/x-x509-ca-cert');
  res.sendFile(p);
});

/* ---- https optional (only needed for WebGPU/PWA-on-iPhone over a trusted CA) ---- */
app.get('/certs', (req, res) => {
  res.send(`<h2>Lumen TLS (optional)</h2>
  <p>Plain <a href="http://100.87.138.101:8090">http://100.87.138.101:8090</a>
  is the primary address — no certificate needed.</p>
  <p>Only set up HTTPS if you want WebGPU / installable PWA on an iPhone:
  install <a href="/ca.crt">/ca.crt</a> as a trusted profile.</p>`);
});

const httpServer = http.createServer(app);
/* Kick off static precompression in the background (never blocks boot). */
setTimeout(() => precompressStatic().catch((e) => console.error('[lumen] precompress:', e.message)), 500);
httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`Lumen (http)  : http://0.0.0.0:${PORT}`);
  console.log(`       LAN     : http://${lanIp()}:${PORT}`);
  console.log(`       Tailscale: http://${tsIp()}:${PORT}`);
});

/* HTTPS server (PWA + WebGPU need a secure context; iOS over Tailscale) */
const tlsOpts = {
  key: fs.existsSync(path.join(CERTS_DIR, 'lumen.key')) && fs.readFileSync(path.join(CERTS_DIR, 'lumen.key')),
  cert: fs.existsSync(path.join(CERTS_DIR, 'lumen.crt')) && fs.readFileSync(path.join(CERTS_DIR, 'lumen.crt')),
};
if (tlsOpts.key && tlsOpts.cert) {
  https.createServer(tlsOpts, app).listen(TLS_PORT, '0.0.0.0', () => {
    console.log(`Lumen (https, optional) : https://0.0.0.0:${TLS_PORT}`);
  });
} else {
  console.log('TLS disabled (HTTP primary; add certs/ to enable WebGPU-PWA on iPhone)');
}

/* ------------------------------------------------------------------ */
function tsIp() {
  try {
    const ifs = os.networkInterfaces();
    return (ifs.tailscale0 || []).find((a) => a.family === 'IPv4')?.address || '';
  } catch { return ''; }
}
function lanIp() {
  try {
    const ifs = os.networkInterfaces();
    for (const name of Object.keys(ifs)) {
      if (name === 'lo') continue;
      const v4 = (ifs[name] || []).find((a) => a.family === 'IPv4');
      if (v4) return v4.address;
    }
  } catch { }
  return '127.0.0.1';
}