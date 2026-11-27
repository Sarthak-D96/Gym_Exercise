#!/bin/sh
# Share the website running on this laptop at a public HTTPS link (Cloudflare quick tunnel).
#
#   deploy/share_with_tunnel.sh
#
# Starts the site in production mode (gunicorn, DEBUG off, the Neon database from .env, the
# installed model) and a Cloudflare quick tunnel to it. Prints a https://<words>.trycloudflare.com
# link; the camera works there because it is HTTPS. No Cloudflare account needed.
# The link works while this script runs (Ctrl+C stops both) and changes on every start.
set -e
cd "$(dirname "$0")/.."
PORT=${PORT:-7860}

command -v cloudflared >/dev/null || { echo "Install it first: brew install cloudflared"; exit 1; }
[ -x .venv/bin/python ] || { echo "Missing .venv (python -m venv .venv && pip install -r requirements.txt)"; exit 1; }

# A real secret key for production mode, generated once and kept out of git.
SECRET_FILE=deploy/tunnel.secret
if [ ! -s "$SECRET_FILE" ]; then
    .venv/bin/python -c "import secrets; print(secrets.token_urlsafe(50))" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
fi

export PATH="$PWD/.venv/bin:$PATH"
export PORT
export DJANGO_SECRET_KEY="$(cat "$SECRET_FILE")"
export DJANGO_DEBUG=false
export DJANGO_ALLOWED_HOSTS=".trycloudflare.com,localhost,127.0.0.1"
export DJANGO_CSRF_TRUSTED_ORIGINS="https://*.trycloudflare.com"
export RECOGNITION_ALLOW_SIMULATION=false

python manage.py collectstatic --noinput >/dev/null
echo "Starting the website on port $PORT (log: deploy/tunnel-server.log) ..."
sh deploy/start.sh > deploy/tunnel-server.log 2>&1 &
SERVER=$!
TUNNEL=""
cleanup() {
    [ -n "$TUNNEL" ] && kill "$TUNNEL" 2>/dev/null
    kill "$SERVER" 2>/dev/null
    echo "Stopped the website and the tunnel."
}
trap cleanup EXIT INT TERM

i=0
until curl -s -o /dev/null "http://localhost:$PORT/accounts/login/"; do
    i=$((i + 1))
    kill -0 "$SERVER" 2>/dev/null || { echo "The website failed to start:"; tail -20 deploy/tunnel-server.log; exit 1; }
    [ $i -lt 60 ] || { echo "The website didn't start in 60 s; see deploy/tunnel-server.log"; exit 1; }
    sleep 1
done

echo "Opening the Cloudflare tunnel (log: deploy/tunnel.log) ..."
cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" > deploy/tunnel.log 2>&1 &
TUNNEL=$!
URL=""
i=0
while [ -z "$URL" ]; do
    i=$((i + 1))
    kill -0 "$TUNNEL" 2>/dev/null || { echo "The tunnel failed:"; tail -20 deploy/tunnel.log; exit 1; }
    [ $i -lt 60 ] || { echo "No tunnel link after 60 s; see deploy/tunnel.log"; exit 1; }
    sleep 1
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' deploy/tunnel.log | head -1 || true)
done

echo ""
echo "  Public link:  $URL"
echo ""
echo "Share it while this runs. Ctrl+C stops the website and the tunnel."
wait "$TUNNEL"
