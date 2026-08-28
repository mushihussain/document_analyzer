# Deploying to a Contabo VPS

A from-scratch runbook for putting this app on a standard Ubuntu VPS (Contabo
or otherwise - nothing here is Contabo-specific beyond how you first get
root access). Ballpark sizing from the app's own resource footprint: at least
**2 vCPU / 4GB RAM**, since the embedding model and OCR engine both load into
memory at startup.

## 0. Before you start

Contabo emails the server's IP and root password (or you set an SSH key at
order time). Pick **Ubuntu 22.04 LTS** if you're choosing the image. Point
your domain's **A record** at the server's IP now - DNS propagation takes a
while and certbot needs it resolved later.

## 1. Initial server access & hardening

```bash
ssh root@YOUR_SERVER_IP
apt update && apt upgrade -y

# non-root user for everything from here on
adduser deploy
usermod -aG sudo deploy
rsync --archive --chown=deploy:deploy ~/.ssh /home/deploy   # copy your SSH key over, if you used one

# basic firewall
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable

exit
ssh deploy@YOUR_SERVER_IP   # continue as this user from now on
```

## 2. System dependencies

```bash
sudo apt install -y python3 python3-venv python3-pip build-essential git nginx
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo apt install -y certbot python3-certbot-nginx
```

`build-essential` covers any pip package that needs to compile; most of this
app's ML dependencies (`onnxruntime`, `opencv-python`) ship prebuilt Linux
wheels so it's mostly a safety net.

## 3. Get the code

```bash
sudo mkdir -p /opt/document-analyzer
sudo chown deploy:deploy /opt/document-analyzer
git clone https://github.com/mushihussain/document_analyzer.git /opt/document-analyzer
cd /opt/document-analyzer
```

## 4. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env
```

In `.env`, set real values for at minimum:

```
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
DOCUMENTS_FOLDER=/opt/document-analyzer/backend/data/documents
VECTOR_DB_PATH=/opt/document-analyzer/backend/data/vector_db
DB_PATH=/opt/document-analyzer/backend/data/app.db
```

Absolute paths matter here - a relative `./data/...` would resolve
differently depending on where systemd launches the process from.

Quick manual test before wiring up systemd:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
# Ctrl+C once you see it start clean
```

Bind to `127.0.0.1`, not `0.0.0.0` - Nginx is the only thing that should talk
to it; the backend itself should never be directly reachable from the
internet.

## 5. Run the backend as a systemd service

```bash
sudo nano /etc/systemd/system/document-analyzer.service
```

```ini
[Unit]
Description=Document Analyzer backend
After=network.target

[Service]
User=deploy
WorkingDirectory=/opt/document-analyzer/backend
ExecStart=/opt/document-analyzer/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

No `--reload` - that flag is dev-only and reloads on every file touch, which
you don't want in production.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now document-analyzer
sudo systemctl status document-analyzer     # should show "active (running)"
```

## 6. Build the frontend

```bash
cd /opt/document-analyzer/frontend
npm ci
npm run build
```

Static files land in `dist/document-analyzer-ui/browser` - that's what Nginx
serves.

## 7. Nginx - serve the frontend, proxy `/api` to the backend

```bash
sudo nano /etc/nginx/sites-available/document-analyzer
```

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    root /opt/document-analyzer/frontend/dist/document-analyzer-ui/browser;
    index index.html;

    client_max_body_size 60m;   # match/exceed MAX_UPLOAD_MB (default 50)

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;   # chat can fail over across 3 LLM providers, ~60s timeout each
    }

    location / {
        try_files $uri /index.html;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/document-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 8. HTTPS

```bash
sudo certbot --nginx -d yourdomain.com
```

Certbot rewrites the Nginx config for TLS and sets up auto-renewal on its
own.

## 9. Verify

Visit `https://yourdomain.com` - register an account, upload a document, ask
it a question. Check backend logs if anything's off:

```bash
sudo journalctl -u document-analyzer -f
```

## Ongoing

- **Data lives in `backend/data/`** (SQLite db, per-user document folders,
  Chroma index) - that's what you back up.
- **Deploying an update:**
  ```bash
  cd /opt/document-analyzer && git pull
  cd backend && .venv/bin/pip install -r requirements.txt && sudo systemctl restart document-analyzer
  cd ../frontend && npm ci && npm run build   # nginx serves the new build immediately, no reload needed
  ```
- **RAM**: this needs ~2 vCPU/4GB minimum - the embedding model and OCR both
  load into memory at startup. If the VPS plan is smaller than that, expect
  it to struggle.

## Frontend API URL

The frontend calls a relative `/api` (see `api.service.ts` / `auth.service.ts`),
which is what the Nginx config above proxies to the backend on the same
origin - not a hardcoded `localhost`. Local development (`ng serve` /
`npm start`) uses `proxy.conf.json` (wired into `angular.json`) to forward
`/api` to `http://localhost:8000` automatically, so no per-environment config
is needed either way.
