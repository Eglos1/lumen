# Lumen

Alternative YouTube frontend + PWA intended to run on a phone over a slow
(per-byte / capped) cellular connection.

## The idea: compute-over-network

The server-side "proxy" does **two** things:

1. **Transport diet** – it relocates the cheapest available stream:
   - video: **144p or the lowest height available** (AV1 preferred – typically
     ~1.5 MB for a 3–4 min video, i.e. ~56 kbps);
   - audio: separate **lowest-bitrate audio-only** format (m4a).
2. **Metadata injection** – titles, channel, duration, subtitles, search,
   related videos, channel uploads and comments are served as **raw JSON data**
   (never through the neural network).

The phone reconstructs resolution **on the client**:

- the 144p frame is split into Y luma + chroma (YCbCr);
- luma tiles are upscaled **3x on-device** with an ONNX super-resolution model
  running on **WebGPU** (falls back to WebGL → WASM);
- the sharp luma is recombined with bilinearly-upscaled chroma into a visible
  ~768×432 canvas;
- all text (subtitles, UI) is drawn natively in HTML **after** the neural
  pass, so it stays perfectly legible and never burns neural-net GPU.

Active data usage on the wire stays tiny; the expensive lifting happens on
your own device.

---

## Run

```sh
cd lumen
npm install            # once (express, onnxruntime-web)
./run.sh               # start HTTP :8090 + HTTPS :8443
./run.sh stop
```

or manually:

```sh
kill "$(cat /tmp/lumen.pid)" 2>/dev/null
PORT=8090 TLSPORT=8443 nohup node server.js > /tmp/lumen.log 2>&1 &
echo $! > /tmp/lumen.pid
```

| URL | Use |
|---|---|
| `http://localhost:8090` | quick local checks (service worker/WebGPU disabled) |
| `http://100.87.138.101:8090` | **iPhone over Tailscale — primary, no cert needed** |
| `https://100.87.138.101:8443` | optional: only needed for WebGPU+installable PWA on iOS |

Requires: `node` ≥ 20, `yt-dlp`, `ffmpeg` (optional). HTTPS is optional and
uses the certs in `certs/`.

---

## First time on the iPhone

1. Install **Tailscale** from the App Store, log in, connect.
2. Open **`http://100.87.138.101:8090`** in Safari. That's it.

No CA profile, no certificates — the whole app (search, grid home feed,
Shorts, 144p proxy playback) works over plain HTTP. The super-resolution
engine falls back to **WebGL → WASM** on insecure contexts. The only features
that need HTTPS are **WebGPU** rendering and **installable/offline PWA**.

If you ever want those, serve `https://100.87.138.101:8443` and install
`/ca.crt` as a trusted profile (`scripts/make-certs.sh` generated the SANs for
this Tailscale IP).

---

## API

All endpoints are same-origin JSON under `/api/`; `:id` is an 11-char YouTube
video id.

| Endpoint | Description |
|---|---|
| `GET /api/manifest/:id` | **stream plan** – chosen 144p stream(s), subtitles, duration, thumb. Itag 394/160-type 144p video, separate audio url. |
| `GET /api/stream/:id` | range-capable `video/mp4` relay for the low-bitrate 144p video. |
| `GET /api/audio/:id`  | `audio/mp4` relay of the lowest-bitrate audio stream. |
| `GET /api/subs/:id?lang=en` | parsed subtitle cues `{startMs, endMs, text}`. |
| `GET /api/search?q=...` | yt-dlp search results. |
| `GET /api/video/:id` | single video metadata (same as search entries). |
| `GET /api/home` | YouTube-like home grid — sections (`Trending now`, `Going viral`, `Fresh drops`) built from cached yt-searches. |
| `GET /api/shorts` | aggregated Shorts shelf from curated creator `/shorts` tabs (all vertical). |
| `GET /api/channel/:id` | channel uploads (flat playlist, cached 20 min). |
| `GET /api/channel/:id/shorts` | that channel's Shorts tab. |
| `GET /api/related/:id` | Invidious recommended (best effort) → falls back to same-channel uploads. |
| `GET /api/comments/:id?n=20` | Invidious comments (best effort, never 5xx). |
| `GET /ca.crt` | local CA cert (only needed for the optional HTTPS/WebGPU setup). |

Server-side ops:

- yt-dlp runs with `--extractor-args youtube:player_client=default,web,web_embedded`.
- `/api/manifest` results are cached in memory for 10 min; channel feeds 20 min;
  stream relays respond with 206 partial content (seek + resume).
- Five subtitle languages are parsed into `{start, end, text}` cues (seconds).
- Home/Shorts feeds are cached ~10 min and built with `--playlist-items`
  caps so they never crawl a whole channel.

---

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `PORT` | `8090` | HTTP listener (also used for Tailscale HTTP access) |
| `TLSPORT` | `8443` | HTTPS listener for the PWA/iPhone |
| `CERTS_DIR` | `certs/` | `lumen.key`, `lumen.crt`, `ca.crt` |
| model | `public/models/super-resolution-10.onnx` | ONNX SR model (3x, 224→672) |

The SR model is **`super-resolution-10.onnx`** (ONNX Model Zoo / ORT tutorial,
~234 KB, opset 10, input `[1,1,224,224]`, output `[1,1,672,672]`). It ships in
`public/models/` and is cached by the service worker's `HEAVY` prefix cache
alongside every ONNX Runtime WebGPU WASM/JS bundle.

### Transfer compression + client caching (neural engine)

- At boot the server mirrors every compressible static file (`.onnx`, `.wasm`,
  `.js`, `.css`…) into `.precomp/` as **Brotli** and **gzip** variants and
  serves them with the right `Content-Encoding`, honouring `Accept-Encoding`
  (brotli preferred, gzip fallback, atomic rename so partial files are never
  served). The ORT engine goes from **~28 MB → ~4.7 MB (br)** on the wire.
- The service worker caches the whole neural stack **client-side** the first
  time it's fetched (`/models/` and `/vendor/onnxruntime-web/` prefix rules,
  cache-first), so the engine **never downloads twice**. CORE shell updates
  without re-downloading the WASM.

Frontend flow (see `public/app.js`):

1. `Engine.choose()` → `webgpu` → `webgl` → `wasm`; the status dot shows
   `engine:<provider>`.
2. 144p frame → YUV luma; tiles upscaled 3x by the model; luma+chroma
   recombined → visible canvas, CSS 100% of stage, 16:9 (or **9:16 Shorts**).
3. If the engine is still warming (model load), it draws the raw 144p frames
   until the model ready.

### Home feed + Shorts

- Home (`/`) is a YouTube-style grid: a horizontal **Shorts** shelf from the
  curated `/shorts` feed, then sectioned grids from `/api/home`.
- Portrait Shorts are detected from the manifest (`aspect`, `short` from
  `width < height && ≤3 min`) and get a **9:16 stage**, a `SHORTS` badge, and
  portrait shelf cards — the SR pipeline already tiles vertical frames safely
  (single 224-tile, edge padding).

---

## Notes / limitations

- **YouTube format availability varies by client.** If the `web_embedded`
  client reports a **video-only** 144p itag (e.g. 394 has `acodec: none`), the
  manifest picks it plus a separate audio stream; the player plays them
  together. `isCombined` reflects whether the chosen video already carries
  audio.
- Comments/related depend on public Invidious instances; when they block us
  (`403/401`), comments return empty and related falls back to the same
  channel's uploads – the UI still works.
- The app is designed to work over plain **HTTP** (no certs). WebGPU and
  installable PWA need HTTPS, so on insecure contexts ONNX Runtime is pinned to
  `numThreads = 1` (no SharedArrayBuffer) and the engine degrades to
  WebGL → WASM.
- All streams are fetched **on demand**; nothing is re-encoded server-side.
- Legal disclaimer: YouTube is not an open service; this is a personal,
  offline-first viewer for your own device — use it respectfully and within
  your platform/ToS rights.

## Builds

Prebuilt Windows / Linux / macOS artifacts are produced by the GitHub Actions workflow in `.github/workflows/build.yml` on every tag.

## License

MIT
