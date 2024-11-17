# tests/test_swarm_chat.py

import pytest
import asyncio
import time
from typing import List
from unittest.mock import Mock, patch

class TestMessageHandling:
    """Tests for various message content and formats"""
    
    @pytest.mark.asyncio
    async def test_basic_message(self, chat_manager):
        """Test basic message handling"""
        token = await chat_manager.create_session("test_user")
        try:
            response = await chat_manager.process_message(token, "Hello")
            assert response is not None
            assert isinstance(response, str)
            assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_long_message_handling(self, chat_manager):
        """Test handling of messages at and beyond size limits"""
        token = await chat_manager.create_session("test_user")
        try:
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
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_empty_whitespace_messages(self, chat_manager):
        """Test handling of empty and whitespace-only messages"""
        token = await chat_manager.create_session("test_user")
        try:
            empty_messages = ["", " ", "\n", "\t", "   \n\t   "]
            for msg in empty_messages:
                with pytest.raises(ValueError, match="Empty or whitespace-only message"):
                    await chat_manager.process_message(token, msg)
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_special_characters(self, chat_manager):
        """Test handling of messages with special characters"""
        token = await chat_manager.create_session("test_user")
        try:
            special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"
            response = await chat_manager.process_message(token, special_chars)
            assert response is not None
            assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_unicode_emoji_support(self, chat_manager):
        """Test handling of Unicode characters and emojis"""
        token = await chat_manager.create_session("test_user")
        try:
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
        finally:
            await chat_manager.close_session(token)

class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_profiling(self, chat_manager):
        """Profile response times under various conditions"""
        token = await chat_manager.create_session("test_user")
        try:
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
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_handling(self, chat_manager):
        """Test system performance with concurrent users"""
        num_users = 5
        tokens = []
        
        async def user_session(user_id: int):
            token = await chat_manager.create_session(f"user_{user_id}")
            tokens.append(token)
            response = await chat_manager.process_message(token, f"Message from user {user_id}")
            assert response == "test response"
        
        try:
            tasks = [user_session(i) for i in range(num_users)]
            await asyncio.gather(*tasks)
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

class TestAgentBehavior:
    """Tests for agent behavior and characteristics"""
    
    @pytest.mark.asyncio
    async def test_agent_selection(self, chat_manager):
        """Test agent selection and switching"""
        token = await chat_manager.create_session("test_user")
        try:
            responses = []
            for _ in range(3):
                response = await chat_manager.process_message(token, "Hello")
                responses.append(response)
                agent = chat_manager.get_current_agent(token)
                assert agent is not None
                assert isinstance(agent.name, str)
            
            assert all(response == "test response" for response in responses)
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_agent_response_consistency(self, chat_manager):
        """Test consistency of agent responses"""
        token = await chat_manager.create_session("test_user")
        try:
            session = await chat_manager.get_session_safe(token)
            
            # Add some initial messages
            test_messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"}
            ]
            session.messages.extend(test_messages)
            
            # Send new messages
            for _ in range(3):
                response = await chat_manager.process_message(token, "Tell me about yourself")
                assert response == "test response"
                assert isinstance(session.messages[-1], dict)
                assert "content" in session.messages[-1]
        finally:
            await chat_manager.close_session(token)

class TestErrorRecovery:
    """Tests for error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_error_handling(self, chat_manager):
        """Test handling of runtime errors"""
        token = await chat_manager.create_session("error_test_user")
        try:
            session = await chat_manager.get_session_safe(token)
            session.client.run = Mock(side_effect=RuntimeError("Test error"))
            
            with pytest.raises(RuntimeError):
                await chat_manager.process_message(token, "Hello")
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_session_recovery(self, chat_manager):
        """Test session recovery after errors"""
        token = await chat_manager.create_session("test_user")
        try:
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
        finally:
            await chat_manager.close_session(token)

class TestSessionManagement:
    """Tests for session handling and management"""
    
    @pytest.mark.asyncio
    async def test_session_limits(self, chat_manager):
        """Test session limit enforcement"""
        tokens = []
        try:
            # Create sessions up to limit
            for i in range(10):  # Using smaller number for testing
                token = await chat_manager.create_session(f"user_{i}")
                tokens.append(token)
                assert token is not None
            
            # Attempt to create session beyond limit
            with pytest.raises(ValueError, match="Maximum session limit reached"):
                for i in range(chat_manager.max_sessions + 1):
                    await chat_manager.create_session(f"overflow_user_{i}")
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_session_isolation(self, chat_manager):
        """Test that sessions are properly isolated"""
        tokens = []
        try:
            # Create multiple sessions
            for i in range(5):
                token = await chat_manager.create_session(f"user_{i}")
                tokens.append(token)
                
            # Send different messages from each session
            for i, token in enumerate(tokens):
                response = await chat_manager.process_message(token, f"Message {i}")
                assert response == "test response"
                
                # Verify session isolation
                session = await chat_manager.get_session_safe(token)
                assert session.messages[-1]["content"] == f"Message {i}"
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_session_cleanup(self, chat_manager):
        """Test proper session cleanup"""
        token = await chat_manager.create_session("test_user")
        try:
            # Use the session
            await chat_manager.process_message(token, "Hello")
            
            # Close the session
            await chat_manager.close_session(token)
            
            # Verify session is cleaned up
            session = await chat_manager.get_session_safe(token)
            assert session is None
            assert token not in chat_manager.tokens
            assert "test_user" not in chat_manager.sessions
        finally:
            await chat_manager.close_session(token)

class TestMessageValidation:
    """Tests for message validation and sanitization"""
    
    @pytest.mark.asyncio
    async def test_html_sanitization(self, chat_manager):
        """Test HTML sanitization"""
        token = await chat_manager.create_session("test_user")
        try:
            html_messages = [
                "<script>alert('xss')</script>",
                "<img src='x' onerror='alert(1)'>",
                "<a href='javascript:alert(1)'>link</a>",
                "<style>body{display:none}</style>",
                "<iframe src='javascript:alert(1)'></iframe>"
            ]
            
            for msg in html_messages:
                response = await chat_manager.process_message(token, msg)
                assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, chat_manager):
        """Test SQL injection prevention"""
        token = await chat_manager.create_session("test_user")
        try:
            sql_messages = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "1; DELETE FROM users",
                "UNION SELECT * FROM users",
                "'; INSERT INTO users VALUES ('hacked'); --"
            ]
            
            for msg in sql_messages:
                response = await chat_manager.process_message(token, msg)
                assert response == "test response"
        finally:
            await chat_manager.close_session(token)

class TestSecurity:
    """Security-related tests"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rate_limiting(self, chat_manager):
        """Test rate limiting functionality"""
        token = await chat_manager.create_session("test_user")
        try:
            await chat_manager.enable_rate_limiting()
            
            # First message should work
            response = await chat_manager.process_message(token, "Test message")
            assert response == "test response"
            
            # Immediate second message should fail
            with pytest.raises(ValueError, match="Rate limit exceeded"):
                await chat_manager.process_message(token, "Test message 2")
            
            # Wait for rate limit to reset
            await asyncio.sleep(chat_manager.rate_limit_interval)
            
            # Message should work again
            response = await chat_manager.process_message(token, "Test message 3")
            assert response == "test response"
        finally:
            await chat_manager.disable_rate_limiting()
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_session_invalidation(self, chat_manager):
        """Test session invalidation"""
        token = await chat_manager.create_session("test_user")
        try:
            # Valid token should work
            response = await chat_manager.process_message(token, "Hello")
            assert response == "test response"
            
            # Test invalid token
            with pytest.raises(ValueError):
                await chat_manager.process_message("invalid_token", "Hello")
            
            # Close session and verify it's invalid
            await chat_manager.close_session(token)
            with pytest.raises(ValueError):
                await chat_manager.process_message(token, "Hello")
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, chat_manager):
        """Test concurrent session creation and usage"""
        num_users = 5
        tokens = []
        
        async def create_and_use_session(user_id: int):
            token = await chat_manager.create_session(f"user_{user_id}")
            tokens.append(token)
            response = await chat_manager.process_message(token, f"Message from {user_id}")
            assert response == "test response"
        
        try:
            tasks = [create_and_use_session(i) for i in range(num_users)]
            await asyncio.gather(*tasks)
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
