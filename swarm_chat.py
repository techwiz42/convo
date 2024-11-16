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
import random

app = FastAPI()
security = HTTPBasic()

class UserSession:
    def __init__(self, username: str):
        self.username = username
        self.messages: List[dict] = []
        self.client = Swarm()
        authors = ["Hemmingway", 
                   "Pynchon", 
                   "Emily Dickenson", 
                   "Dale Carnegie", 
                   "A Freudian Psychoanalyst", 
                   "A flapper from the 1920s"]
        author = authors[random.randint(0,len(authors))]
        print(f"AUTHOR: {author}")
        self.agent = Agent(
            name="Agent",
            instructions=f"""
                          You are a helpful but charmingly inept agent.
                          Respond as if you were {author}.
                          End your response by enquiring about some aspect of the
                          user's life or something the user has revealed in the 
                          chat history. Do not begin your response with 'Ah' or anything like it. 
                         """
        )
        self.lock = asyncio.Lock()  # Lock for this specific user's session

class ChatMessage(BaseModel):
    content: str

class SwarmChatManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.tokens: Dict[str, str] = {}  # token -> username mapping
        self.sessions_lock = asyncio.Lock()
        self.tokens_lock = asyncio.Lock()

    @asynccontextmanager
    async def get_session_safe(self, token: str) -> Optional[UserSession]:
        """Safely get a session with proper locking"""
        async with self.tokens_lock:
            username = self.tokens.get(token)
            if not username:
                yield None
                return

        async with self.sessions_lock:
            session = self.sessions.get(username)
            if not session:
                yield None
                return

        async with session.lock:
            yield session

    async def create_session(self, username: str) -> str:
        """Create a new session with proper locking"""
        token = secrets.token_urlsafe(32)
        
        # First, handle the session creation/retrieval
        async with self.sessions_lock:
            if username not in self.sessions:
                self.sessions[username] = UserSession(username)
        
        # Then, handle the token mapping
        async with self.tokens_lock:
            self.tokens[token] = username
        
        return token

    async def process_message(self, token: str, content: str) -> Optional[str]:
        async with self.get_session_safe(token) as session:
            if not session:
                raise HTTPException(status_code=401, detail="Invalid session")

            # Add user message to this user's message history
            session.messages.append({
                "role": "user",
                "content": content
            })

            # Use this user's dedicated Swarm client and agent
            try:
                response = await asyncio.to_thread(
                    session.client.run,
                    agent=session.agent,
                    messages=session.messages
                )

                # Update this user's message history and agent state
                session.messages = response.messages
                session.agent = response.agent

                # Get the last agent message
                for message in reversed(response.messages):
                    if message.get("content") and message.get("sender") == "Agent":
                        return message["content"]
                return None

            except Exception as e:
                # Log the error and raise a proper HTTP exception
                print(f"Error processing message: {str(e)}")
                raise HTTPException(status_code=500, detail="Error processing message")

    async def get_user_messages(self, token: str) -> List[dict]:
        """Safely get user messages with proper locking"""
        async with self.get_session_safe(token) as session:
            if not session:
                raise HTTPException(status_code=401, detail="Invalid session")
            return session.messages.copy()  # Return a copy to prevent external modifications

chat_manager = SwarmChatManager()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    with open("static/index.html") as f:
        return f.read()

@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    # In a real app, you'd verify credentials against a database
    token = await chat_manager.create_session(credentials.username)
    return {"token": token, "username": credentials.username}

@app.post("/chat")
async def send_message(message: ChatMessage, token: str = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    response = await chat_manager.process_message(token, message.content)
    return {"response": response}

@app.get("/history")
async def get_history(token: str = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    messages = await chat_manager.get_user_messages(token)
    return {"messages": messages}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
