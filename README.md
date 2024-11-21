# SwarmChat

SwarmChat is an interactive chat application that lets users converse with a variety of AI personalities, each with their own unique writing style and perspective.

## Features

- Multiple AI personalities including Ernest Hemingway, Thomas Pynchon, Emily Dickinson, and others
- Voice input and text-to-speech capabilities
- Real-time chat interface with typing indicators
- Secure authentication system
- HTTPS/SSL support
- Mobile-responsive design

## Architecture

SwarmChat consists of two main components:
1. A FastAPI backend service that handles chat logic and AI interactions
2. A React frontend for the user interface

### Frontend (React)

The frontend is built with React and uses several modern web technologies:

- **UI Components**: Built using shadcn/ui components
- **State Management**: Uses React hooks for local state
- **Speech Features**: Implements browser Web Speech API for voice input and text-to-speech
- **Styling**: Utilizes Tailwind CSS for responsive design
- **HTTP Client**: Uses the Fetch API for communication with the backend

Key files:
```
frontend/
├── src/
│   ├── components/
│   │   └── SwarmChat.jsx     # Main chat interface
│   ├── lib/
│   │   └── speech-handler.js # Speech recognition and synthesis
│   └── App.jsx
```

### Backend Services

The application requires two systemd services to run:

#### 1. SwarmChat API Service

This service runs the FastAPI backend application.

File: `/etc/systemd/system/swarmchat-api.service`
```ini
[Unit]
Description=SwarmChat FastAPI Application
After=network.target
Wants=network-online.target

[Service]
User=peter
Group=peter
WorkingDirectory=/home/peter/convo
EnvironmentFile=/etc/swarmchat/environment
ExecStart=/home/peter/.virtualenvs/swarm/bin/uvicorn swarm_chat:app --host 127.0.0.1 --port 8000 --log-level info
Restart=always
StandardOutput=journal
StandardError=journal
```

#### 2. SwarmChat Frontend Service

This service serves the React frontend application.

File: `/etc/systemd/system/swarmchat-frontend.service`
```ini
[Unit]
Description=SwarmChat React Frontend
After=network.target
Wants=network-online.target

[Service]
User=peter
Group=users
WorkingDirectory=/home/peter/convo/frontend
Environment=NODE_ENV=production
Environment=PORT=3000
Environment=REACT_APP_API_URL=https://swarmchat.me/api
ExecStart=/usr/bin/npm start
Restart=always
StandardOutput=journal
StandardError=journal
```

### NGINX Configuration

NGINX is used to serve the React frontend and proxy API requests to the backend.

File: `/etc/nginx/sites-available/swarmchat`
```nginx
server {
    listen 443 ssl;
    server_name swarmchat.me www.swarmchat.me;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/swarmchat.me/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/swarmchat.me/privkey.pem;

    # Serve React frontend
    location / {
        alias /home/peter/convo/frontend/build/;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to FastAPI backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/swarmchat.git
cd swarmchat
```

2. Set up the Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
npm run build
```

4. Configure environment:
```bash
sudo mkdir -p /etc/swarmchat
sudo nano /etc/swarmchat/environment
# Add your OpenAI API key:
# OPENAI_API_KEY=your_key_here
```

5. Set up services:
```bash
sudo cp services/swarmchat-api.service /etc/systemd/system/
sudo cp services/swarmchat-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable swarmchat-api swarmchat-frontend
sudo systemctl start swarmchat-api swarmchat-frontend
```

6. Configure NGINX:
```bash
sudo cp nginx/swarmchat /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/swarmchat /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Maintenance

### Logs
- Backend API logs: `sudo journalctl -u swarmchat-api -f`
- Frontend logs: `sudo journalctl -u swarmchat-frontend -f`
- NGINX access logs: `/var/log/nginx/swarmchat.access.log`
- NGINX error logs: `/var/log/nginx/swarmchat.error.log`

### Common Tasks
- Restart services: `sudo systemctl restart swarmchat-api swarmchat-frontend`
- Update frontend:
  ```bash
  cd frontend
  git pull
  npm install
  npm run build
  sudo systemctl restart swarmchat-frontend
  ```
- Check service status: `sudo systemctl status swarmchat-api swarmchat-frontend`

## Security Notes

- The OpenAI API key is stored in `/etc/swarmchat/environment` with 600 permissions
- SSL certificates are managed by Let's Encrypt
- All API requests are proxied through NGINX over HTTPS
- Frontend build files are owned by www-data for NGINX access
