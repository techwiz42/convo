import asyncio
import logging
import random
import secrets
from datetime import datetime
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, Cookie, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from agents import (
    triage_agent,
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper,
)
from swarm import Swarm, Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'swarm_chat_{datetime.now().strftime("%Y%m%d")}.log')
    ],
)
logger = logging.getLogger(__name__)

# Set uvicorn access logger to only show startup and HTTP methods
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.handlers = []
uvicorn_logger.addHandler(logging.StreamHandler())
uvicorn_logger.setLevel(logging.WARNING)

app = FastAPI()
security = HTTPBasic()


class ChatMessage(BaseModel):
    """Model for chat messages."""

    content: str


class UserSession:
    """Class to manage user sessions."""

    def __init__(self, username: str):
        self.username: str = username
        self.messages: List[dict] = []
        self.client: Swarm = Swarm()
        self.agent: Agent = triage_agent
        self.lock: asyncio.Lock = asyncio.Lock()
        self.first_message_handled: bool = False
        logger.info(f"New session created for user: {username}")

    def select_random_agent(self) -> Agent:
        """Select and instantiate a random agent."""
        agents = [
            transfer_to_hemmingway,
            transfer_to_pynchon,
            transfer_to_dickinson,
            transfer_to_dale_carnegie,
            transfer_to_shrink,
            transfer_to_flapper,
        ]
        selected_func = random.choice(agents)
        new_agent = selected_func()
        logger.info(f"Selected random agent: {new_agent.name}")
        return new_agent


class SwarmChatManager:
    """Manager class for handling chat sessions."""

    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.tokens: Dict[str, str] = {}
        self.sessions_lock: asyncio.Lock = asyncio.Lock()
        self.tokens_lock: asyncio.Lock = asyncio.Lock()
        logger.info("SwarmChatManager initialized")

    @asynccontextmanager
    async def get_session_safe(
        self, token: str
    ) -> AsyncGenerator[Optional[UserSession], None]:
        """Safely get a session with proper locking."""
        if not token:
            logger.warning("No token provided")
            yield None
            return

        try:
            username: Optional[str] = None
            session: Optional[UserSession] = None

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
                finally:
                    logger.debug(f"Session released for user: {username}")

        except Exception as e:
            logger.error(f"Error in get_session_safe: {str(e)}")
            yield None

    async def create_session(self, username: str) -> str:
        """Create a new session with proper locking."""
        try:
            token: str = secrets.token_urlsafe(32)

            async with self.sessions_lock:
                self.sessions[username] = UserSession(username)

            async with self.tokens_lock:
                self.tokens[token] = username

            return token

        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Error creating session")

    async def process_message(self, token: str, content: str) -> Optional[str]:
        """Process a message and return the response."""
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            async with self.get_session_safe(token) as session:
                if not session:
                    raise HTTPException(status_code=401, detail="Invalid session")

                session.messages.append({"role": "user", "content": content})

                if not session.first_message_handled:
                    logger.info("Processing first message with triage agent")
                    session.first_message_handled = True
                else:
                    session.agent = session.select_random_agent()
                    logger.info(f"Using agent: {session.agent.name}")

                response = await asyncio.to_thread(
                    session.client.run, agent=session.agent, messages=session.messages
                )

                if response.messages:
                    latest_message = response.messages[-1]
                    if latest_message.get("role") == "assistant":
                        session.messages = response.messages
                        return latest_message.get("content")

                return None

        except Exception as e:
            logger.error(f"Message processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing message: {str(e)}"
            )

    async def get_user_messages(self, token: str) -> List[dict]:
        """Safely get user messages with proper locking."""
        if not token:
            logger.warning("No token provided for message retrieval")
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            async with self.get_session_safe(token) as session:
                if not session:
                    logger.warning(f"Invalid session for token: {token[:8]}...")
                    raise HTTPException(status_code=401, detail="Invalid session")

                logger.debug(f"Retrieving messages for session")
                return session.messages.copy()

        except Exception as e:
            logger.error(f"Message retrieval error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error retrieving messages: {str(e)}"
            )


chat_manager = SwarmChatManager()

# FastAPI routes
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_chat_page() -> str:
    """Serve the chat page."""
    try:
        with open("static/index.html") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving chat page: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving chat page")


@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    """Handle user login."""
    try:
        logger.info(f"Login attempt: {credentials.username}")
        token = await chat_manager.create_session(credentials.username)
        return {"token": token, "username": credentials.username}
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/chat")
async def send_message(message: ChatMessage, token: str = Cookie(None)) -> dict:
    """Handle chat messages."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    logger.info("Processing chat message")
    response = await chat_manager.process_message(token, message.content)
    return {"response": response}


@app.get("/history")
async def get_history(token: str = Cookie(None)) -> dict:
    """Get chat history."""
    if not token:
        logger.warning("History request without token")
        raise HTTPException(status_code=401, detail="Not authenticated")

    logger.debug("Retrieving chat history")
    messages = await chat_manager.get_user_messages(token)
    return {"messages": messages}


if __name__ == "__main__":
    print("Starting Swarm Chat server...")
    config = uvicorn.Config(
        app, host="0.0.0.0", port=8000, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        print("Server shutdown complete")
