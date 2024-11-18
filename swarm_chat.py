"""FastAPI-based chat application that manages multi-agent
   conversations using a swarm architecture."""

import asyncio
import logging
import random
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator

# Third-party imports
import uvicorn  # pylint: disable=import-error
from fastapi import FastAPI, HTTPException, Depends  # pylint: disable=import-error
from fastapi.responses import HTMLResponse  # pylint: disable=import-error
from fastapi.staticfiles import StaticFiles  # pylint: disable=import-error
from fastapi.security import HTTPBasic, HTTPBasicCredentials  # pylint: disable=import-error
from pydantic import BaseModel  # pylint: disable=import-error
from swarm import Swarm, Agent  # pylint: disable=import-error

# Local imports
from agents import (
    triage_agent,
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper,
    transfer_to_bullwinkle,
    transfer_to_yogi_berra,
    transfer_to_yogi_bhajan,
    transfer_to_mencken
)

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

#pylint: disable=R0903
class ChatMessage(BaseModel):
    """Model for chat messages."""
    content: str
 
    def __str__(self) -> str:
        """String representation of chat message."""
        return self.content

#pylint: disable=R0903
class UserSession:
    """Class to manage user sessions."""

    def __init__(self, username: str):
        self.username: str = username
        self.messages: List[dict] = []
        self.client: Swarm = Swarm()
        self.agent: Agent = triage_agent
        self.lock: asyncio.Lock = asyncio.Lock()
        self.first_message_handled: bool = False
        logger.info("New session created for user: %s", username)

    def select_random_agent(self) -> Agent:
        """Select and instantiate a random agent."""
        agents = [
            transfer_to_hemmingway,
            transfer_to_pynchon,
            transfer_to_dickinson,
            transfer_to_dale_carnegie,
            transfer_to_shrink,
            transfer_to_flapper,
            transfer_to_bullwinkle,
            transfer_to_yogi_berra,
            transfer_to_mencken,
            transfer_to_yogi_bhajan
        ]
        selected_func = random.choice(agents)
        new_agent = selected_func()
        logger.info("Selected random agent: %s", new_agent.name)
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
                    logger.warning("Invalid token attempted: %s...", token[:8])
                    yield None
                    return

            async with self.sessions_lock:
                session = self.sessions.get(username)
                if not session:
                    logger.warning("No session found for username: %s", username)
                    yield None
                    return

            async with session.lock:
                logger.debug("Session acquired for user: %s", username)
                try:
                    yield session
                finally:
                    logger.debug("Session released for user: %s", username)

        except Exception as e:
            logger.error("Error in get_session_safe: %s", str(e))
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
            logger.error("Session creation failed: %s", str(e))
            raise HTTPException(status_code=500, detail="Error creating session") from e

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
                    logger.info("Using agent: %s", session.agent.name)

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
            logger.error("Message processing error: %s", str(e), exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing message: {str(e)}"
            ) from e

    async def get_user_messages(self, token: str) -> List[dict]:
        """Safely get user messages with proper locking."""
        if not token:
            logger.warning("No token provided for message retrieval")
            raise HTTPException(status_code=401, detail="Not authenticated")

        try:
            async with self.get_session_safe(token) as session:
                if not session:
                    logger.warning("Invalid session for token: %s...", token[:8])
                    raise HTTPException(status_code=401, detail="Invalid session")

                logger.debug("Retrieving messages for session")
                return session.messages.copy()

        except Exception as e:
            logger.error("Message retrieval error: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Error retrieving messages: {str(e)}"
            ) from e


chat_manager = SwarmChatManager()

# FastAPI routes
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_chat_page() -> str:
    """Serve the chat page."""
    try:
        with open("static/index.html", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error("Error serving chat page: %s", str(e))
        raise HTTPException(status_code=500, detail="Error serving chat page") from e


@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    """Handle user login."""
    try:
        logger.info("Login attempt: %s", credentials.username)
        token = await chat_manager.create_session(credentials.username)
        return {"token": token, "username": credentials.username}
    except Exception as e:
        logger.error("Login failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Login failed") from e


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
