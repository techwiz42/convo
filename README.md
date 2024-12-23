# SwarmChat

**see it in action at https://swarmchat.me**

SwarmChat is an interactive chat application that lets users converse with a variety of AI personalities, each with his or her own unique writing style and perspective.

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

File: `/etc/systemd/system/swarmchat-api-prod.service` (dev is similar)
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

File: `/etc/systemd/system/swarmchat-fe-prod.service` (dev is similar)
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

## Development Pipeline

SwarmChat uses a development/production pipeline with parallel environments:

### Directory Structure
```bash
/home/peter/
├── convo/          # Production environment
└── convo-dev/      # Development environment
```

### Initial Setup

1. Clone the repository twice:
```bash
cd /home/peter
git clone https://github.com/yourusername/swarmchat.git convo
git clone https://github.com/yourusername/swarmchat.git convo-dev
```

2. Set up branches:
```bash
# In production directory
cd /home/peter/convo
git checkout main

# In development directory
cd /home/peter/convo-dev
git checkout -b dev
```

3. Create virtual environments:
```bash
# Production environment
python -m venv ~/.virtualenvs/swarm
source ~/.virtualenvs/swarm/bin/activate
pip install -r requirements.txt

# Development environment
python -m venv ~/.virtualenvs/swarm-dev
source ~/.virtualenvs/swarm-dev/bin/activate
pip install -r requirements.txt
```

### Development Workflow

1. Make changes in the dev environment:
```bash
cd /home/peter/convo-dev
git checkout dev
# Make your changes
git add .
git commit -m "Description of changes"
git push origin dev
```

2. Deploy to production:
```bash
# In production directory
cd /home/peter/convo
git checkout main
git pull origin main
git merge origin/dev
git push origin main
```

### Deployment Script

Create a deployment script (`deploy-to-prod.sh`):
```bash
#!/bin/bash

echo "Deploying to production..."

# Go to dev directory and get latest
cd /home/peter/convo-dev
git checkout dev
git pull origin dev

# Go to prod directory
cd /home/peter/convo
git checkout main
git pull origin main

# Merge dev into main
git merge origin/dev

# Push changes
git push origin main

# Rebuild frontend
cd frontend
npm run build

# Restart services
sudo systemctl restart swarmchat-api-prod
sudo systemctl restart swarmchat-fe-prod

echo "Deployment complete"
```

Make it executable:
```bash
chmod +x deploy-to-prod.sh
```

### Environment Configuration

1. Production environment:
```bash
# /etc/swarmchat/environment
OPENAI_API_KEY=your_key_here
NODE_ENV=production
PORT=3000
REACT_APP_API_URL=https://swarmchat.me/api
```

2. Development environment:
```bash
# /etc/swarmchat/environment.dev
OPENAI_API_KEY=your_key_here
NODE_ENV=development
PORT=3001
REACT_APP_API_URL=https://dev.swarmchat.me/api
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
sudo cp services/swarmchat-fe.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable swarmchat-api swarmchat-fe
sudo systemctl start swarmchat-api swarmchat-fe
```

6. Configure NGINX:
```bash
sudo cp nginx/swarmchat /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/swarmchat /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Testing

SwarmChat includes a comprehensive testing suite using pytest for Python tests.

### Running Tests

```bash
# Run the full test suite
pytest tests/

# Run specific test categories
pytest tests/ -m "security"    # Security tests
pytest tests/ -m "performance" # Performance tests
pytest tests/ -m "integration" # Integration tests
```

### Test Categories

- Unit Tests: Basic functionality testing
- Integration Tests: Testing component interactions
- Security Tests: Testing security features and vulnerabilities
- Performance Tests: Testing system under load
- Chaos Tests: Testing system resilience
- Contract Tests: Validating API contracts
- Property-Based Tests: Testing with generated inputs

### Code Quality

1. Run pylint for PEP-8 compliance:
```bash
pylint swarm_chat.py agents.py tests/*.py
```

2. Run mypy for static type checking:
```bash
mypy swarm_chat.py agents.py tests/*.py
```

### Test Fixtures

The test suite includes fixtures for:
- Session management
- Mock agents
- Rate limiting
- Error scenarios
- Large conversations
- Concurrent operations

### Monitoring Tests

The test suite includes monitoring capabilities:
- Log output validation
- Metrics collection
- Performance benchmarking
- Resource usage tracking

See `tests/test_swarm_chat.py` and `tests/test_swarm_chat_extended.py` for detailed test implementations.

## Maintenance

### Logs
- Backend API logs: `sudo journalctl -u swarmchat-api -f`
- Frontend logs: `sudo journalctl -u swarmchat-fe -f`
- NGINX access logs: `/var/log/nginx/swarmchat.access.log`
- NGINX error logs: `/var/log/nginx/swarmchat.error.log`

### Common Tasks
- Restart services: `sudo systemctl restart swarmchat-api swarmchat-fe`
- Update frontend:
  ```bash
  cd frontend
  git pull
  npm install
  npm run build
  sudo systemctl restart swarmchat-fe
  ```
- Check service status: `sudo systemctl status swarmchat-api swarmchat-fe`

## Security Notes

- The OpenAI API key is stored in `/etc/swarmchat/environment` with 600 permissions
- SSL certificates are managed by Let's Encrypt
- All API requests are proxied through NGINX over HTTPS
- Frontend build files are owned by www-data for NGINX access
