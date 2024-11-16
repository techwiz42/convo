from fastapi import FastAPI, HTTPException, WebSocket, Cookie, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
from swarm import Swarm, Agent
from collections import defaultdict
import secrets
import uvicorn
from contextlib import asynccontextmanager
from agents import (triage_agent, transfer_to_hemmingway, transfer_to_pynchon,
                   transfer_to_dickinson, transfer_to_dale_carnegie,
                   transfer_to_shrink, transfer_to_flapper)
import logging
import sys
from datetime import datetime
import random
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'swarm_chat_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBasic()
server_should_exit = False

def signal_handler(signum, frame):
    global server_should_exit
    logger.info("Shutdown signal received. Cleaning up...")
    server_should_exit = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class UserSession:
    def __init__(self, username: str):
        self.username = username
        self.messages: List[dict] = []
        self.client = Swarm()
        self.agent = triage_agent
        self.lock = asyncio.Lock()
        self.is_first_message = True
        logger.info(f"New session created for user: {username}")

    def get_random_agent_transfer(self):
        """Get a random agent transfer function"""
        transfer_functions = [
            transfer_to_hemmingway,
            transfer_to_pynchon,
            transfer_to_dickinson,
            transfer_to_dale_carnegie,
            transfer_to_shrink,
            transfer_to_flapper
        ]
        selected = random.choice(transfer_functions)
        logger.info(f"Selected random agent transfer: {selected.__name__}")
        return selected()  # Return the agent directly, not the transfer function

class ChatMessage(BaseModel):
    content: str

class SwarmChatManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.tokens: Dict[str, str] = {}  # token -> username mapping
        self.sessions_lock = asyncio.Lock()
        self.tokens_lock = asyncio.Lock()
        logger.info("SwarmChatManager initialized")

    @asynccontextmanager
    async def get_session_safe(self, token: str) -> Optional[UserSession]:
        """Safely get a session with proper locking"""
        try:
            if not token:
                logger.warning("No token provided")
                yield None
                return

            async with self.tokens_lock:
                username = self.tokens.get(token)
                if not username:
                    logger.warning(f"Invalid token attempted: {token[:8]}...")
                    yield None
                    return

            async with self.sessions_lock:
                session = self.sessions.get(username)
                if not session:
                    logger.warning(f"No session found for username: {username}")
                    yield None
                    return

            async with session.lock:
                logger.debug(f"Session acquired for user: {username}")
                try:
                    yield session
                except Exception as e:
                    logger.error(f"Error during session use: {str(e)}")
                    raise
                finally:
                    logger.debug(f"Session released for user: {username}")

        except Exception as e:
            logger.warning(f"Error in get_session_safe: {str(e)}")
            yield None
            return

    async def create_session(self, username: str) -> str:
        """Create a new session with proper locking"""
        try:
            token = secrets.token_urlsafe(32)
            
            async with self.sessions_lock:
                if username not in self.sessions:
                    self.sessions[username] = UserSession(username)
                    logger.info(f"Created new session for user: {username}")
                else:
                    logger.info(f"Retrieved existing session for user: {username}")
            
            async with self.tokens_lock:
                self.tokens[token] = username
                logger.debug(f"Token mapped for user: {username}")
            
            return token

        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Error creating session")

    async def process_message(self, token: str, content: str) -> Optional[str]:
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        async with self.get_session_safe(token) as session:
            if not session:
                logger.warning(f"Invalid session for token: {token[:8]}...")
                raise HTTPException(status_code=401, detail="Invalid session")

            try:
                logger.info(f"Processing message for user {session.username}")
                session.messages.append({
                    "role": "user",
                    "content": content
                })

                # Handle agent selection
                if session.is_first_message:
                    logger.info("Processing first message with triage agent")
                    session.is_first_message = False
                else:
                    # Get a new random agent for this message
                    session.agent = session.get_random_agent_transfer()
                    logger.info(f"Switched to agent: {session.agent.name}")

                response = await asyncio.to_thread(
                    session.client.run,
                    agent=session.agent,
                    messages=session.messages
                )

                if response.messages:
                    latest_message = response.messages[-1]
                    if latest_message.get("role") == "assistant":
                        session.messages = response.messages
                        logger.info("Message history updated")
                        return latest_message.get("content")

                logger.warning("No valid assistant message found in response")
                return None

            except Exception as e:
                logger.error(f"Message processing error: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing message: {str(e)}"
                )

    async def get_user_messages(self, token: str) -> List[dict]:
        """Safely get user messages with proper locking"""
        if not token:
            logger.warning("No token provided for message retrieval")
            raise HTTPException(status_code=401, detail="Not authenticated")
            
        async with self.get_session_safe(token) as session:
            if not session:
                logger.warning(f"Invalid session for token: {token[:8]}...")
                raise HTTPException(status_code=401, detail="Invalid session")
            
            try:
                logger.debug(f"Retrieving messages for user: {session.username}")
                return session.messages.copy()
            except Exception as e:
                logger.error(f"Message retrieval error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving messages: {str(e)}"
                )

chat_manager = SwarmChatManager()

# FastAPI routes
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    try:
        with open("static/index.html") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving chat page: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving chat page")

@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    try:
        logger.info(f"Login attempt: {credentials.username}")
        token = await chat_manager.create_session(credentials.username)
        return {"token": token, "username": credentials.username}
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/chat")
async def send_message(message: ChatMessage, token: str = Cookie(None)):
    if not token:
        logger.warning("Chat attempt without token")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    logger.info(f"Chat message received, token: {token[:8]}...")
    response = await chat_manager.process_message(token, message.content)
    return {"response": response}

@app.get("/history")
async def get_history(token: str = Cookie(None)):
    if not token:
        logger.warning("History request without token")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    logger.debug(f"History requested, token: {token[:8]}...")
    messages = await chat_manager.get_user_messages(token)
    return {"messages": messages}

@app.post("/quit")
async def quit_server():
    logger.info("Quit requested via API")
    global server_should_exit
    server_should_exit = True
    return {"status": "shutting down"}

if __name__ == "__main__":
    logger.info("Starting Swarm Chat server...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("Server shutdown complete")
