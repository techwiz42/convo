# tests/conftest.py

import pytest
import asyncio
from typing import AsyncGenerator, Dict, List
from unittest.mock import Mock
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes
class MockAgent:
    def __init__(self, name="test_agent", instructions="test instructions"):
        self.name = name
        self.instructions = instructions
        self.functions = []

class MockSwarm:
    def run(self, agent: MockAgent, messages: List[Dict]) -> Mock:
        """Mock the Swarm's run method"""
        return Mock(messages=[
            *messages,
            {"role": "assistant", "content": "test response"}
        ])

class MockSession:
    def __init__(self, username: str):
        self.username = username
        self.messages: List[Dict] = []
        self.agent = MockAgent()
        self.client = MockSwarm()
        self.lock = asyncio.Lock()
        self.first_message_handled = False

class MockChatManager:
    def __init__(self):
        self.sessions: Dict[str, MockSession] = {}
        self.tokens: Dict[str, str] = {}
        self.sessions_lock = asyncio.Lock()
        self.tokens_lock = asyncio.Lock()
        self.rate_limits: Dict[str, float] = {}
        self.max_sessions = 1000
        self.max_message_size = 8192

    async def create_session(self, username: str) -> str:
        """Create a new session for a user"""
        if len(self.sessions) >= self.max_sessions:
            raise ValueError("Maximum session limit reached")
        
        async with self.sessions_lock:
            if username in self.sessions:
                raise ValueError(f"Session already exists for user: {username}")
            
            session = MockSession(username)
            self.sessions[username] = session
            token = f"test_token_{username}"
            self.tokens[token] = username
            return token

    async def get_session_safe(self, token: str) -> MockSession:
        """Safely get a session by token"""
        if not token:
            return None
        username = self.tokens.get(token)
        if not username:
            return None
        return self.sessions.get(username)

    async def process_message(self, token: str, content: str) -> str:
        """Process a message from a user"""
        if not token:
            raise ValueError("No token provided")
        
        if not content or content.isspace():
            raise ValueError("Empty or whitespace-only message")
            
        if len(content) > self.max_message_size:
            raise ValueError("Message exceeds maximum length")
            
        session = await self.get_session_safe(token)
        if not session:
            raise ValueError("Invalid session")
            
        # Rate limiting check
        current_time = time.time()
        if token in self.rate_limits:
            last_time = self.rate_limits[token]
            if current_time - last_time < 0.1:  # 10 messages per second limit
                raise ValueError("Rate limit exceeded")
        
        self.rate_limits[token] = current_time
        
        # Process message
        async with session.lock:
            session.messages.append({
                "role": "user",
                "content": content
            })
            
            response = session.client.run(
                agent=session.agent,
                messages=session.messages
            )
            
            if response.messages:
                return response.messages[-1].get("content")
            return None

    def get_current_agent(self, token: str) -> MockAgent:
        """Get the current agent for a session"""
        username = self.tokens.get(token)
        if username and username in self.sessions:
            return self.sessions[username].agent
        return None

    async def close_session(self, token: str) -> None:
        """Close a session"""
        username = self.tokens.get(token)
        if username:
            async with self.sessions_lock:
                self.sessions.pop(username, None)
            async with self.tokens_lock:
                self.tokens.pop(token, None)
            self.rate_limits.pop(token, None)

# Event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

# Main fixtures
@pytest.fixture
def chat_manager():
    """Provide a mock chat manager instance."""
    return MockChatManager()

@pytest.fixture
async def test_session(chat_manager):
    """Provide a test session and token."""
    token = await chat_manager.create_session("test_user")
    yield token, chat_manager.sessions["test_user"]
    await chat_manager.close_session(token)

@pytest.fixture
async def multiple_sessions(chat_manager):
    """Provide multiple test sessions."""
    tokens = []
    for i in range(5):
        token = await chat_manager.create_session(f"test_user_{i}")
        tokens.append(token)
    yield tokens
    for token in tokens:
        await chat_manager.close_session(token)

@pytest.fixture
async def error_session(chat_manager):
    """Provide a session for error testing."""
    token = await chat_manager.create_session("error_test_user")
    session = chat_manager.sessions["error_test_user"]
    session.client.run = Mock(side_effect=RuntimeError("Test error"))
    yield token, session
    await chat_manager.close_session(token)

@pytest.fixture
async def rate_limited_session(chat_manager):
    """Provide a session for rate limit testing."""
    token = await chat_manager.create_session("rate_test_user")
    chat_manager.rate_limits[token] = time.time()
    yield token
    await chat_manager.close_session(token)

@pytest.fixture
async def filled_session(chat_manager):
    """Provide a session with existing messages."""
    token = await chat_manager.create_session("filled_test_user")
    session = chat_manager.sessions["filled_test_user"]
    
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"}
    ]
    session.messages.extend(test_messages)
    yield token, session
    await chat_manager.close_session(token)

@pytest.fixture
async def max_sessions(chat_manager):
    """Fill the session manager to capacity."""
    tokens = []
    for i in range(chat_manager.max_sessions):
        token = await chat_manager.create_session(f"max_test_user_{i}")
        tokens.append(token)
    yield
    for token in tokens:
        await chat_manager.close_session(token)

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "stress: mark test as a stress test")
    config.addinivalue_line("markers", "reliability: mark test as a reliability test")
