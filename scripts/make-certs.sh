#!/usr/bin/env bash
# Generate a local CA + server certificate for Lumen's HTTPS mode so the
# PWA/service-worker/WebGPU stack works from Safari on an iPhone over Tailscale.
#
#   certs/ca.crt     -> install this on the iPhone (Safari -> open /ca.crt -> profile)
#   certs/lumen.crt  -> server cert (SAN covers the Tailscale IP + localhost)
#   certs/lumen.key  -> server private key (keep secret, local only)
#
# Usage: ./scripts/make-certs.sh [tailscale-ip]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p certs && cd certs

TS_IP="${1:-$(ip -4 addr show tailscale0 2>/dev/null | grep -oP 'inet \K[0-9.]+' || echo 127.0.0.1)}"
echo "SAN target: IP:${TS_IP}, DNS:lumen, DNS:localhost"

if [[ ! -f ca.key ]]; then
    openssl genrsa -out ca.key 3072 2>/dev/null
    openssl req -x509 -new -key ca.key -sha256 -days 3650 -subj "/CN=Lumen Local CA" -out ca.crt
    echo "CA created"
fi

openssl req -newkey rsa:2048 -nodes -keyout lumen.key \
    -subj "/CN=lumen" -out lumen.csr 2>/dev/null

openssl x509 -req -in lumen.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out lumen.crt -days 825 -sha256 \
    -extfile <(printf "subjectAltName=IP:%s,DNS:lumen,DNS:localhost\nextendedKeyUsage=serverAuth" "$TS_IP")
rm -f lumen.csr
chmod 600 lumen.key
echo "certs ready in certs/ (ca.crt, lumen.crt, lumen.key)"