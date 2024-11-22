"""FastAPI-based chat application that manages multi-agent conversations using a swarm architecture."""

import asyncio
import logging
import random
import secrets
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from logging.handlers import RotatingFileHandler
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from swarm import Swarm, Agent

from agents import (
    create_moderator,
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper,
    transfer_to_bullwinkle,
    transfer_to_yogi_berra,
    transfer_to_yogi_bhajan,
    transfer_to_mencken,
    get_agent_functions
)

# Create logging directory if it doesn't exist
LOG_DIR = "/var/log/swarm"
os.makedirs(LOG_DIR, exist_ok=True)

# Add this before the app starts
print("OpenAI API Key status:", "Present" if os.getenv("OPENAI_API_KEY") else "Missing")
print("Key starts with:", os.getenv("OPENAI_API_KEY")[:4] + "..." if os.getenv("OPENAI_API_KEY") else "None")

# Configure main application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        RotatingFileHandler(
            os.path.join(LOG_DIR, "swarm_chat.log"),
            maxBytes=10_485_760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    ],
)
logger = logging.getLogger(__name__)

# Configure access logger
access_logger = logging.getLogger("swarm.access")
access_logger.setLevel(logging.INFO)
access_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "access.log"),
    maxBytes=10_485_760,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
access_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
access_handler.setFormatter(access_formatter)
access_logger.addHandler(access_handler)

# Configure error logger
error_logger = logging.getLogger("swarm.error")
error_logger.setLevel(logging.ERROR)
error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    maxBytes=10_485_760,
    backupCount=5,
    encoding='utf-8'
)
error_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    'Exception: %(exc_info)s'
)
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Set uvicorn access logger
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.handlers = []
uvicorn_access_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "uvicorn_access.log"),
    maxBytes=10_485_760,
    backupCount=5,
    encoding='utf-8'
)
uvicorn_access_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
uvicorn_logger.addHandler(uvicorn_access_handler)
uvicorn_logger.setLevel(logging.INFO)

# Pydantic Models
class ChatMessage(BaseModel):
    """Model for chat messages."""
    content: str
    def __str__(self) -> str:
        return self.content

class TokenResponse(BaseModel):
    """Model for token responses."""
    token: str
    username: str

class MessageResponse(BaseModel):
    """Model for chat message responses."""
    response: Optional[str]

class HistoryResponse(BaseModel):
    """Model for chat history responses."""
    messages: List[Dict[str, str]]

app = FastAPI()
security = HTTPBasic()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://swarmchat.me:3000",
                   "http://swaarmchat.me",
                   "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

class UserSession:
    """Class to manage user sessions."""
    
    GREETINGS = [
        "Hello {name}! I'm the moderator. What would you like to discuss today?",
        "Welcome {name}! I'm here to guide our conversation. What's on your mind?", 
        "Greetings {name}! I'm your moderator for today. How can I assist you?",
        "Hi {name}! I'll be moderating our chat today. What would you like to explore?",
        "Good to see you, {name}! I'm the moderator. What shall we talk about?"
    ]

    def __init__(self, username: str):
        self.username: str = username
        self.messages: List[dict] = []
        self.client: Swarm = Swarm()
        self.agent: Agent = create_moderator(username)
        self.lock: asyncio.Lock = asyncio.Lock()
        self.first_message_sent: bool = False
        logger.info("New session created for user: %s", username)

    async def send_first_message(self) -> Optional[str]:
        """Send initial moderator message."""
        try:
            if not self.first_message_sent:
                initial_message = random.choice(self.GREETINGS).format(name=self.username)
                self.messages.append({"role": "assistant", "content": initial_message})
                self.first_message_sent = True
                return initial_message
            return None
        except Exception as e:
            logger.error("Error sending first message: %s", str(e))
            return None

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
        new_agent = selected_func(self.username)
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

    async def log_access(self, token: str, request: Request, message_type: str, content: str):
        """Log access information"""
        try:
            async with self.tokens_lock:
                username = self.tokens.get(token)
                if not username:
                    return

            client_ip = request.client.host if request.client else "unknown"
            log_message = (
                f"IP: {client_ip} | "
                f"User: {username} | "
                f"Type: {message_type} | "
                f"Content: {content}"
            )
            access_logger.info(log_message)
        except Exception as e:
            logger.error(f"Error logging access: {str(e)}")

    async def get_token_username(self, token: str) -> Optional[str]:
        """Get username associated with token."""
        async with self.tokens_lock:
            return self.tokens.get(token)

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
            username = await self.get_token_username(token)
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
        """Create a new session with proper locking and detailed logging."""
        try:
            logger.info("Starting session creation for user: %s", username)
            token: str = secrets.token_urlsafe(32)

            logger.debug("Created token for user %s: %s...", username, token[:8])

            async with self.sessions_lock:
                logger.debug("Acquired sessions lock for user: %s", username)
                try:
                    session = UserSession(username)
                    self.sessions[username] = session
                    logger.debug("Created UserSession object for user: %s", username)
                
                    init_message = await session.send_first_message()
                    logger.debug("Sent first message for user %s: %s", username, init_message)
                except Exception as session_error:
                    logger.error(
                        "Error during session object creation for user %s: %s",
                        username,
                        str(session_error),
                        exc_info=True
                    )
                    raise

            async with self.tokens_lock:
                logger.debug("Acquired tokens lock for user: %s", username)
                self.tokens[token] = username

            logger.info("Successfully created session for user: %s", username)
            return token

        except Exception as e:
            logger.error(
                "Session creation failed for user %s: %s",
                username,
                str(e),
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error creating session: {str(e)}"
            ) from e

    async def process_message(
        self, token: str, content: str, request: Request
    ) -> Optional[str]:
        """Process a message and return the response."""
        try:
            await self.log_access(token, request, "prompt", content)

            async with self.get_session_safe(token) as session:
                if not session:
                    raise HTTPException(status_code=401, detail="Invalid session")

                session.messages.append({"role": "user", "content": content})
                
                if session.first_message_sent:
                    session.agent = session.select_random_agent()
                    logger.info("Using agent: %s", session.agent.name)

                response = await asyncio.to_thread(
                    session.client.run, agent=session.agent, messages=session.messages
                )

                if response.messages:
                    latest_message = response.messages[-1]
                    if latest_message.get("role") == "assistant":
                        session.messages = response.messages
                        response_content = latest_message.get("content")
                        await self.log_access(token, request, "response", response_content)
                        return response_content

                return None

        except Exception as e:
            logger.error("Message processing error: %s", str(e), exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error processing message: {str(e)}"
            ) from e

chat_manager = SwarmChatManager()

# FastAPI routes
@app.post("/api/login", response_model=TokenResponse)
async def login(credentials: HTTPBasicCredentials = Depends(security)) -> TokenResponse:
    """Handle user login."""
    try:
        logger.info("Login attempt: %s", credentials.username)
        
        if not os.access(LOG_DIR, os.W_OK):
            logger.error("Log directory %s is not writable", LOG_DIR)
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: Log directory not writable"
            )
            
        try:
            token = await chat_manager.create_session(credentials.username)
            logger.info("Session created successfully for user: %s", credentials.username)
            return TokenResponse(token=token, username=credentials.username)
        except Exception as session_error:
            logger.error(
                "Session creation failed for user %s: %s",
                credentials.username,
                str(session_error),
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Session creation failed: {str(session_error)}"
            ) from session_error
            
    except Exception as e:
        logger.error(
            "Login failed for user %s: %s",
            credentials.username,
            str(e),
            exc_info=True
        )
        error_detail = f"Login failed: {str(e)}"
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=error_detail) from e

async def get_token_from_auth(request: Request) -> str:
    """Extract token from Authorization header."""
    auth = request.headers.get('Authorization')
    if not auth or not auth.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )
    return auth.split(' ')[1]

@app.post("/api/chat", response_model=MessageResponse)
async def send_message(
    message: ChatMessage,
    request: Request,
    token: str = Depends(get_token_from_auth)
) -> MessageResponse:
    """Handle chat messages."""
    logger.info("Processing chat message")
    response = await chat_manager.process_message(token, message.content, request)
    return MessageResponse(response=response)

@app.get("/api/history", response_model=HistoryResponse)
async def get_history(token: str = Depends(get_token_from_auth)) -> HistoryResponse:
    """Get chat history."""
    logger.debug("Retrieving chat history")
    async with chat_manager.get_session_safe(token) as session:
        if not session:
            raise HTTPException(status_code=401, detail="Invalid session")
        return HistoryResponse(messages=session.messages)

@app.get("/api/debug")
async def debug_info():
    return {
        "env_vars": {k: v[:4] + "..." if k == "OPENAI_API_KEY" else v
                     for k, v in os.environ.items()},
        "working_directory": os.getcwd(),
        "user": os.getuid(),
        "group": os.getgid(),
        "python_path": sys.path,
    }

if __name__ == "__main__":
    logger.info("Starting Swarm Chat server...")
    try:
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutting down by user request...")
    except Exception as e:
        error_logger.error("Server crashed", exc_info=True)
        raise
    finally:
        logger.info("Server shutdown complete")
