/* Lumen service worker — app-shell + model + ONNX runtime caching for
   instant boots and offline start. The API layer is always network-first. */
const VERSION = 'lumen-v2';
const CORE = [
  '/',
  '/index.html',
  '/app.css',
  '/app.js',
  '/manifest.json',
  '/icons/icon-32.png',
  '/icons/icon-180.png',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/icon-maskable-512.png',
];
// Big, immutable compute assets (neural model + ONNX Runtime WASM/JS) live
// in their own bucket so the shell can update without re-downloading them.
// Prefix rules, not exact paths: any file added under these dirs later gets
// cached too — no list surgery when we swap ORT versions.
const HEAVY_PREFIXES = ['/models/', '/vendor/onnxruntime-web/'];
const isHeavy = (p) => HEAVY_PREFIXES.some((pre) => p.startsWith(pre));

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(`${VERSION}-core`).then((c) => c.addAll(CORE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;               // pass CDN/API others

  if (e.request.method !== 'GET') return;

  // Heavy static: cache-first (model + ort wasm are slow & immutable). All
  // compressed (br/gz) responses are stored decoded by the browser's fetch,
  // so the cache is codec-agnostic.
  if (isHeavy(url.pathname)) {
    e.respondWith(
      (async () => {
        const hit = await caches.match(e.request);
        if (hit) return hit;
        // range requests (wasm streaming) aren't cache-keyed well -> fetch straight
        if (e.request.headers.get('range')) return fetch(e.request);
        const net = await fetch(e.request);
        if (net.ok) {
          caches.open(`${VERSION}-heavy`).then((c) => c.put(e.request, net.clone()));
        }
        return net;
      })()
    );
    return;
  }

  if (CORE.includes(url.pathname)) {
    e.respondWith(
      fetch(e.request).catch(() => caches.match(e.request))
    );
    return;
  }

  // API: network only (fresh YouTube data), never cache.
  if (url.pathname.startsWith('/api/')) return;
});