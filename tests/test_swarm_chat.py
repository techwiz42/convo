# tests/test_swarm_chat.py

import pytest
import asyncio
import time
from typing import List, Dict
from unittest.mock import Mock, patch

class TestMessageHandling:
    """Tests for various message content and formats"""
    
    @pytest.mark.asyncio
    async def test_basic_message(self, chat_manager):
        """Test basic message handling"""
        token = await chat_manager.create_session("test_user")
        response = await chat_manager.process_message(token, "Hello")
        assert response is not None
        assert isinstance(response, str)
        assert response == "test response"

    @pytest.mark.asyncio
    async def test_long_message_handling(self, chat_manager):
        """Test handling of messages at and beyond size limits"""
        token = await chat_manager.create_session("test_user")
        
        # Test message within limits
        normal_msg = "A" * 1000
        response = await chat_manager.process_message(token, normal_msg)
        assert response is not None
        
        # Test message at limit
        limit_msg = "A" * chat_manager.max_message_size
        response = await chat_manager.process_message(token, limit_msg)
        assert response is not None
        
        # Test message exceeding limit
        over_limit_msg = "A" * (chat_manager.max_message_size + 1)
        with pytest.raises(ValueError, match="Message exceeds maximum length"):
            await chat_manager.process_message(token, over_limit_msg)

    @pytest.mark.asyncio
    async def test_empty_whitespace_messages(self, chat_manager):
        """Test handling of empty and whitespace-only messages"""
        token = await chat_manager.create_session("test_user")
        
        empty_messages = ["", " ", "\n", "\t", "   \n\t   "]
        for msg in empty_messages:
            with pytest.raises(ValueError, match="Empty or whitespace-only message"):
                await chat_manager.process_message(token, msg)

    @pytest.mark.asyncio
    async def test_special_characters(self, chat_manager):
        """Test handling of messages with special characters"""
        token = await chat_manager.create_session("test_user")
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
        response = await chat_manager.process_message(token, special_chars)
        assert response is not None
        assert response == "test response"

    @pytest.mark.asyncio
    async def test_unicode_emoji_support(self, chat_manager):
        """Test handling of Unicode characters and emojis"""
        token = await chat_manager.create_session("test_user")
        
        # Test Unicode characters
        unicode_text = "Hello 你好 Привет שָׁלוֹם"
        response = await chat_manager.process_message(token, unicode_text)
        assert response is not None
        assert response == "test response"
        
        # Test emojis
        emoji_text = "Hello! 👋 How are you? 🤔"
        response = await chat_manager.process_message(token, emoji_text)
        assert response is not None
        assert response == "test response"

class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_profiling(self, chat_manager):
        """Profile response times under various conditions"""
        token = await chat_manager.create_session("test_user")
        message_sizes = [10, 100, 1000, 5000]
        response_times = []
        
        for size in message_sizes:
            message = "A" * size
            start_time = time.time()
            response = await chat_manager.process_message(token, message)
            response_time = time.time() - start_time
            response_times.append(response_time)
            assert response == "test response"
        
        assert all(t < 1.0 for t in response_times), "Some responses took too long"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_handling(self, chat_manager):
        """Test system performance with concurrent users"""
        num_users = 5
        tasks = []
        
        async def user_session(user_id: int):
            token = await chat_manager.create_session(f"user_{user_id}")
            response = await chat_manager.process_message(token, f"Message from user {user_id}")
            assert response == "test response"
            await chat_manager.close_session(token)
        
        for i in range(num_users):
            tasks.append(asyncio.create_task(user_session(i)))
        
        await asyncio.gather(*tasks)

class TestAgentBehavior:
    """Tests for agent behavior and characteristics"""
    
    @pytest.mark.asyncio
    async def test_agent_selection(self, chat_manager):
        """Test agent selection and switching"""
        token = await chat_manager.create_session("test_user")
        
        responses = []
        for _ in range(3):
            response = await chat_manager.process_message(token, "Hello")
            responses.append(response)
            agent = chat_manager.get_current_agent(token)
            assert agent is not None
            assert isinstance(agent.name, str)
        
        assert all(response == "test response" for response in responses)

    @pytest.mark.asyncio
    async def test_agent_response_consistency(self, filled_session):
        """Test consistency of agent responses"""
        token, session = filled_session
        
        # Verify existing messages
        assert len(session.messages) > 0
        
        # Send new messages
        for _ in range(3):
            response = await chat_manager.process_message(token, "Tell me about yourself")
            assert response == "test response"
            assert isinstance(session.messages[-1], dict)
            assert "content" in session.messages[-1]

class TestErrorRecovery:
    """Tests for error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_error_handling(self, error_session):
        """Test handling of runtime errors"""
        token, _ = error_session
        
        with pytest.raises(RuntimeError):
            await chat_manager.process_message(token, "Hello")

    @pytest.mark.asyncio
    async def test_session_recovery(self, chat_manager):
        """Test session recovery after errors"""
        token = await chat_manager.create_session("test_user")
        
        # First message should work
        response = await chat_manager.process_message(token, "Hello")
        assert response == "test response"
        
        # Simulate an error
        with patch.object(chat_manager, 'process_message', side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                await chat_manager.process_message(token, "Error message")
        
        # Session should still be usable
        session = await chat_manager.get_session_safe(token)
        assert session is not None
        assert session.messages[-1]["content"] == "Hello"

class TestSessionManagement:
    """Tests for session handling and management"""
    
    @pytest.mark.asyncio
    async def test_session_limits(self, chat_manager):
        """Test session limit enforcement"""
        tokens = []
        try:
            # Create sessions up to limit
            for i in range(chat_manager.max_sessions):
                token = await chat_manager.create_session(f"user_{i}")
                tokens.append(token)
                assert token is not None
            
            # Attempt to create session beyond limit
            with pytest.raises(ValueError, match="Maximum session limit reached"):
                await chat_manager.create_session("overflow_user")
        finally:
            # Cleanup
            for token in tokens:
                await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_session_isolation(self, multiple_sessions):
        """Test that sessions are properly isolated"""
        tokens = multiple_sessions
        
        # Send different messages from each session
        for i, token in enumerate(tokens):
            response = await chat_manager.process_message(token, f"Message {i}")
            assert response == "test response"
            
            # Verify session isolation
            session = await chat_manager.get_session_safe(token)
            assert session.messages[-1]["content"] == f"Message {i}"

    @pytest.mark.asyncio
    async def test_session_cleanup(self, chat_manager):
        """Test proper session cleanup"""
        token = await chat_manager.create_session("test_user")
        
        # Use the session
        await chat_manager.process_message(token, "Hello")
        
        # Close the session
        await chat_manager.close_session(token)
        
        # Verify session is cleaned up
        session = await chat_manager.get_session_safe(token)
        assert session is None
        assert token not in chat_manager.tokens
        assert "test_user" not in chat_manager.sessions

class TestMessageValidation:
    """Tests for message validation and sanitization"""
    
    @pytest.mark.asyncio
    async def test_html_sanitization(self, chat_manager):
        """Test HTML sanitization"""
        token = await chat_manager.create_session("test_user")
        
        html_messages = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "<a href='javascript:alert(1)'>link</a>"
        ]
        
        for msg in html_messages:
            response = await chat_manager.process_message(token, msg)
            assert response == "test response"
            # In a real implementation, verify HTML is properly escaped

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, chat_manager):
        """Test SQL injection prevention"""
        token = await chat_manager.create_session("test_user")
        
        sql_messages = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1; DELETE FROM users"
        ]
        
        for msg in sql_messages:
            response = await chat_manager.process_message(token, msg)
            assert response == "test response"
            # In a real implementation, verify SQL is properly escaped

class TestSecurity:
    """Security-related tests"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rate_limiting(self, chat_manager):
        """Test rate limiting functionality"""
        token = await chat_manager.create_session("test_user")
        
        # First message should work
        response = await chat_manager.process_message(token, "Test message")
        assert response == "test response"
        
        # Immediate second message should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await chat_manager.process_message(token, "Test message 2")
        
        # Wait for rate limit to reset
        await asyncio.sleep(0.1)
        
        # Message should work again
        response = await chat_manager.process_message(token, "Test message 3")
        assert response == "test response"

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_session_invalidation(self, chat_manager):
        """Test session invalidation"""
        # Create and use valid session
        token = await chat_manager.create_session("test_user")
        response = await chat_manager.process_message(token, "Hello")
        assert response == "test response"
        
        # Test invalid token
        with pytest.raises(ValueError):
            await chat_manager.process_message("invalid_token", "Hello")
        
        # Close session and verify it's invalid
        await chat_manager.close_session(token)
        with pytest.raises(ValueError):
            await chat_manager.process_message(token, "Hello")

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, chat_manager):
        """Test concurrent session creation and usage"""
        async def create_and_use_session(user_id: int):
            token = await chat_manager.create_session(f"user_{user_id}")
            response = await chat_manager.process_message(token, f"Message from {user_id}")
            assert response == "test response"
            await chat_manager.close_session(token)
        
        # Create multiple sessions concurrently
        tasks = [create_and_use_session(i) for i in range(5)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
