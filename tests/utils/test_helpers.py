# tests/utils/test_helpers.py

import random
import string
import asyncio
import time
from typing import List, Dict, Optional, Tuple
import psutil
import os
import statistics
from dataclasses import dataclass
from contextlib import contextmanager
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class TestSession:
    """Test session data container"""
    user_id: str
    token: str
    messages: List[Dict]
    start_time: float
    last_activity: float

class PerformanceMetrics:
    """Container for performance test metrics"""
    def __init__(self):
        self.response_times: List[float] = []
        self.error_counts: int = 0
        self.memory_samples: List[float] = []
        self.start_time: float = time.time()
    
    def add_response_time(self, response_time: float):
        self.response_times.append(response_time)
    
    def add_memory_sample(self, memory_mb: float):
        self.memory_samples.append(memory_mb)
    
    def get_statistics(self) -> Dict:
        """Calculate performance statistics"""
        if not self.response_times:
            return {}
        
        return {
            "avg_response_time": statistics.mean(self.response_times),
            "max_response_time": max(self.response_times),
            "min_response_time": min(self.response_times),
            "response_time_stddev": statistics.stdev(self.response_times) if len(self.response_times) > 1 else 0,
            "total_errors": self.error_counts,
            "avg_memory_usage": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "max_memory_usage": max(self.memory_samples) if self.memory_samples else 0,
            "total_duration": time.time() - self.start_time
        }

def generate_large_message(size: int) -> str:
    """Generate a message of specified size in bytes"""
    chunks = []
    remaining_size = size
    
    # Add some realistic content patterns
    patterns = [
        "Hello! ", "How are you? ", "This is a test message. ",
        "We need to make this message longer. ", "Testing large content. "
    ]
    
    while remaining_size > 0:
        if remaining_size > 50 and random.random() < 0.3:
            # Add a pattern occasionally
            pattern = random.choice(patterns)
            chunks.append(pattern)
            remaining_size -= len(pattern)
        else:
            # Add random characters
            chunk_size = min(remaining_size, 100)
            chunk = ''.join(random.choices(
                string.ascii_letters + string.digits + ' .,!?', 
                k=chunk_size
            ))
            chunks.append(chunk)
            remaining_size -= chunk_size
    
    return ''.join(chunks)

async def create_concurrent_users(
    num_users: int,
    session_duration: int = 300
) -> List[TestSession]:
    """Create specified number of concurrent test users"""
    current_time = time.time()
    users = []
    
    for i in range(num_users):
        users.append(TestSession(
            user_id=f'test_user_{i}',
            token=f'test_token_{i}_{random.randint(1000, 9999)}',
            messages=[],
            start_time=current_time,
            last_activity=current_time
        ))
    
    return users

def monitor_memory_usage() -> float:
    """Monitor current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

async def simulate_network_delay(
    min_delay: float = 0.01,
    max_delay: float = 0.5
) -> None:
    """Simulate network latency"""
    delay = random.uniform(min_delay, max_delay)
    await asyncio.sleep(delay)

@contextmanager
def create_test_session(
    user_id: str,
    token: Optional[str] = None
) -> TestSession:
    """Create a test session with cleanup"""
    session = TestSession(
        user_id=user_id,
        token=token or f"test_token_{random.randint(1000, 9999)}",
        messages=[],
        start_time=time.time(),
        last_activity=time.time()
    )
    
    try:
        yield session
    finally:
        # Cleanup session data
        session.messages.clear()

async def simulate_user_behavior(
    session: TestSession,
    duration: int = 60,
    message_interval: Tuple[float, float] = (1.0, 5.0),
    error_probability: float = 0.1
) -> PerformanceMetrics:
    """Simulate realistic user behavior"""
    metrics = PerformanceMetrics()
    end_time = time.time() + duration
    
    while time.time() < end_time:
        try:
            # Simulate user thinking time
            await asyncio.sleep(random.uniform(*message_interval))
            
            # Generate message
            if random.random() < error_probability:
                # Generate an invalid message occasionally
                message = generate_large_message(10000)  # Exceeds size limit
            else:
                message = generate_large_message(random.randint(10, 500))
            
            # Record metrics
            start_time = time.time()
            session.messages.append({
                "content": message,
                "timestamp": start_time
            })
            
            # Simulate processing time
            await simulate_network_delay()
            
            # Record response time
            response_time = time.time() - start_time
            metrics.add_response_time(response_time)
            
            # Monitor memory
            metrics.add_memory_sample(monitor_memory_usage())
            
            session.last_activity = time.time()
            
        except Exception as e:
            metrics.error_counts += 1
            logger.error(f"Error in user simulation: {str(e)}")
    
    return metrics

def generate_test_messages() -> List[str]:
    """Generate a variety of test messages"""
    return [
        # Normal messages
        "Hello, how are you?",
        "Testing a normal message",
        
        # Empty and whitespace
        "",
        "   ",
        "\n\t\r",
        
        # Special characters
        "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`",
        
        # Unicode and emoji
        "Hello 你好 Привет שָׁלוֹם",
        "👋 Hello! 🌟 Testing emoji 🚀",
        
        # HTML/Script injection attempts
        "<script>alert('xss')</script>",
        "<img src='x' onerror='alert(1)'>",
        
        # SQL injection attempts
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        
        # Large messages
        "A" * 1000,
        "B" * 5000,
        
        # Mixed content
        "Regular text with 你好 and 👋 and <script>",
        
        # JSON content
        json.dumps({"key": "value", "numbers": [1, 2, 3]}),
        
        # Markdown content
        "# Header\n## Subheader\n* List item\n```code block```"
    ]

def create_load_test_profile(
    duration: int = 3600,
    max_users: int = 1000,
    ramp_up_time: int = 300
) -> Dict:
    """Create a load test profile"""
    return {
        "duration": duration,
        "max_users": max_users,
        "ramp_up_time": ramp_up_time,
        "user_creation_rate": max_users / ramp_up_time,
        "think_time_range": (1.0, 5.0),
        "error_injection_rate": 0.1,
        "message_size_range": (10, 5000)
    }

def validate_response(
    response: str,
    expected_patterns: Optional[List[str]] = None,
    excluded_patterns: Optional[List[str]] = None
) -> bool:
    """Validate response content"""
    if not response:
        return False
    
    if expected_patterns:
        if not all(pattern in response for pattern in expected_patterns):
            return False
    
    if excluded_patterns:
        if any(pattern in response for pattern in excluded_patterns):
            return False
    
    return True

async def cleanup_test_data(sessions: List[TestSession]) -> None:
    """Clean up test data"""
    for session in sessions:
        session.messages.clear()
    await asyncio.sleep(0)  # Yield control to event loop

def log_test_metrics(metrics: PerformanceMetrics, test_name: str) -> None:
    """Log test metrics"""
    stats = metrics.get_statistics()
    logger.info(f"Test Results for {test_name}:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")

class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_user_messages(num_messages: int) -> List[Dict]:
        """Generate a series of user messages"""
        messages = []
        for i in range(num_messages):
            messages.append({
                "content": generate_large_message(random.randint(10, 500)),
                "timestamp": time.time() + i
            })
        return messages
    
    @staticmethod
    def generate_error_cases() -> List[Dict]:
        """Generate error test cases"""
        return [
            {"type": "overflow", "content": generate_large_message(10000)},
            {"type": "invalid_chars", "content": "\0\1\2\3\4\5"},
            {"type": "empty", "content": ""},
            {"type": "injection", "content": "<script>alert('xss')</script>"},
            {"type": "unicode_error", "content": "Hello \ud800World"}  # Invalid Unicode
        ]

if __name__ == "__main__":
    # Example usage
    async def main():
        users = await create_concurrent_users(10)
        for user in users:
            metrics = await simulate_user_behavior(user, duration=10)
            log_test_metrics(metrics, f"User {user.user_id}")
        await cleanup_test_data(users)
    
    asyncio.run(main())
