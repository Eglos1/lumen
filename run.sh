#!/usr/bin/env bash
# Lumen — start/stop the Express proxy (HTTP :8090 + HTTPS :8443)
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-start}" in
  start)
    if [ -f /tmp/lumen.pid ] && kill -0 "$(cat /tmp/lumen.pid)" 2>/dev/null; then
      echo "Lumen already running (pid $(cat /tmp/lumen.pid)). Stop it first: ./run.sh stop"
      exit 0
    fi
    PORT=8090 TLSPORT=8443 nohup node server.js > /tmp/lumen.log 2>&1 &
    echo $! > /tmp/lumen.pid
    sleep 1
    echo "Lumen started (pid $(cat /tmp/lumen.pid))"
    echo "  iPhone (Tailscale): http://100.87.138.101:8090"
    echo "  https (optional)  : https://100.87.138.101:8443"
    ;;
  stop)
    if [ -f /tmp/lumen.pid ]; then
      kill "$(cat /tmp/lumen.pid)" 2>/dev/null || true
      rm -f /tmp/lumen.pid
      echo "Lumen stopped."
    else
      echo "No pid file — nothing to stop."
    fi
    ;;
  restart) "$0" stop; "$0" start ;;
  *) echo "usage: $0 [start|stop|restart]"; exit 1 ;;
esac