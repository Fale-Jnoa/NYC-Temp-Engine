#!/usr/bin/env bash
# Pull the latest code + models and restart the bot service.
# Run on the host after pushing changes:  ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

# DEPLOY.md creates .venv/, but hosts set up by hand may use venv/.
if [ -x .venv/bin/pip ]; then
  VENV=.venv
elif [ -x venv/bin/pip ]; then
  VENV=venv
else
  echo "No virtualenv found (looked for .venv/ and venv/)" >&2
  exit 1
fi

echo "==> git pull"
git pull --ff-only

echo "==> pip install into $VENV/ (in case deps changed)"
"$VENV/bin/pip" install -r requirements.txt --quiet

echo "==> restart service"
sudo systemctl restart knyc-bot

echo "==> status"
systemctl --no-pager --lines=0 status knyc-bot || true
echo "Deployed. Follow logs with:  journalctl -u knyc-bot -f"
