# Deploying the bot 24/7 (Google Cloud free tier)

Runs `knyc_discord_bot.py` as an always-on `systemd` service on a Google Cloud
"Always Free" **e2-micro** VM. The bot only makes **outbound** HTTPS calls, so
no inbound firewall rules are needed.

> **Cost heads-up.** The e2-micro *compute* is free, but Google now bills for
> external IPv4 addresses (~$3/mo). To stay truly $0 you can run the VM
> IPv6-only or behind Cloud NAT; otherwise budget a few dollars a month for the
> IP. If that's a dealbreaker, Oracle Cloud Always Free or a ~€4 Hetzner VPS
> avoid the IPv4 charge and deploy with the exact same steps from **§2** on.

## 1. Create the VM

1. <https://console.cloud.google.com> → create/select a project.
2. Enable the **Compute Engine API** (first time only).
3. **Compute Engine → VM instances → Create instance**:
   - **Name:** `knyc-bot`
   - **Region:** `us-west1`, `us-central1`, or `us-east1` — **only these qualify
     for the free tier.**
   - **Series:** E2 · **Machine type:** `e2-micro`
   - **Boot disk:** Ubuntu 24.04 LTS, 30 GB Standard persistent disk
   - Leave the HTTP/HTTPS firewall boxes **unchecked** (no inbound needed)
   - **Create**
4. When it shows a green check, click **SSH** to open a browser terminal. Run
   everything below there.

## 2. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
```

## 3. Swap (e2-micro has only 1 GB RAM — cheap insurance against OOM)

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 4. Clone and install

```bash
git clone https://github.com/Fale-Jnoa/NYC-Temp-Engine.git
cd NYC-Temp-Engine
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## 5. Discord credentials

Create `.env` in the repo directory (it is gitignored — it lives only here):

```bash
cat > .env <<'EOF'
DISCORD_TOKEN=paste_your_token_here
GUILD_ID=paste_your_guild_id_here
EOF
chmod 600 .env
```

## 6. Install as a systemd service

```bash
sudo tee /etc/systemd/system/knyc-bot.service >/dev/null <<EOF
[Unit]
Description=KNYC Temp Engine Discord bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/NYC-Temp-Engine
ExecStart=$HOME/NYC-Temp-Engine/.venv/bin/python -u knyc_discord_bot.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now knyc-bot
```

`Restart=always` + `enable` means it comes back after crashes **and** reboots.

## 7. Verify

```bash
systemctl status knyc-bot --no-pager
journalctl -u knyc-bot -n 40 --no-pager
```

You should see `Logged in as ...` and `Posted — reassessed high ...`. To watch
live: `journalctl -u knyc-bot -f`.

## Updating after you push code or retrained models

```bash
cd ~/NYC-Temp-Engine && chmod +x deploy.sh && ./deploy.sh
```

`deploy.sh` = `git pull` + `pip install -r requirements.txt` + restart. Because
the runtime `nowcast_log.csv` is gitignored, pulls never conflict with the
host's live prediction history.

## Notes

- **Timezone:** the bot computes NY-local time internally (`zoneinfo`), so the
  VM's clock/timezone is irrelevant — the 6:30 AM ET scorecard fires correctly
  regardless of where the VM lives.
- **Secrets:** `.env` is never committed. Edit it directly on the VM; restart
  with `sudo systemctl restart knyc-bot` after any change.
- **New pip dependency:** `deploy.sh` reinstalls requirements each run, so a new
  package in `requirements.txt` is picked up automatically.
- **Two Discord channels required:** `#predictions` and `#score` must exist in
  the guild.
