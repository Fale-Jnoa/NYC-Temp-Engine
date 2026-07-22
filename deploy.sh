#!/usr/bin/env bash
# Pull the latest code + models and restart the bot service.
# Run on the host after pushing changes:  ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> git pull"
git pull --ff-only

echo "==> pip install (in case deps changed)"
.venv/bin/pip install -r requirements.txt --quiet

echo "==> restart service"
sudo systemctl restart knyc-bot

echo "==> status"
systemctl --no-pager --lines=0 status knyc-bot || true
echo "Deployed. Follow logs with:  journalctl -u knyc-bot -f"
