import pytest
import asyncio
import json
import time
from typing import List, Dict, Optional
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st
import psutil
import resource
from contextlib import contextmanager

class TestIntegrationScenarios:
    """Complex integration scenarios"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, chat_manager):
        """Test a complete conversation flow with agent switches"""
        token = await chat_manager.create_session("test_user")
        conversation = [
            "Hello, I need help with a task",
            "Can you tell me more about yourself?",
            "What's your approach to problem-solving?",
            "Thank you for your help"
        ]
        
        try:
            previous_agent = None
            for message in conversation:
                response = await chat_manager.process_message(token, message)
                current_agent = chat_manager.get_current_agent(token)
                
                assert response is not None
                assert current_agent is not None
                
                if previous_agent:
                    # Verify agent switching
                    assert current_agent != previous_agent
                
                previous_agent = current_agent
                
            # Verify conversation history
            session = await chat_manager.get_session_safe(token)
            assert len(session.messages) == len(conversation) * 2  # Including responses
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_context_preservation(self, chat_manager):
        """Test that context is maintained across agent switches"""
        token = await chat_manager.create_session("test_user")
        try:
            # Initial context setting
            await chat_manager.process_message(token, "My name is John")
            
            # Series of follow-up messages
            responses = []
            for _ in range(3):
                response = await chat_manager.process_message(token, "What's my name?")
                responses.append(response)
            
            # Verify context preservation
            session = await chat_manager.get_session_safe(token)
            assert all("John" in str(msg.get("content", "")) 
                      for msg in session.messages 
                      if msg.get("role") == "user")
        finally:
            await chat_manager.close_session(token)

class TestStressScenarios:
    """Stress testing scenarios"""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_rapid_agent_switching(self, chat_manager):
        """Test rapid consecutive agent switches"""
        token = await chat_manager.create_session("test_user")
        try:
            tasks = []
            for _ in range(50):  # Rapid consecutive messages
                tasks.append(
                    chat_manager.process_message(token, "Quick message")
                )
            
            responses = await asyncio.gather(*tasks)
            assert all(response == "test response" for response in responses)
            
            session = await chat_manager.get_session_safe(token)
            assert len(session.messages) == 100  # 50 messages + 50 responses
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_large_conversation_history(self, chat_manager):
        """Test handling of very large conversation histories"""
        token = await chat_manager.create_session("test_user")
        try:
            # Generate large conversation history
            for _ in range(100):  # Large number of messages
                await chat_manager.process_message(
                    token,
                    "Message " + "content " * 50  # Long messages
                )
            
            # Verify system still responds
            response = await chat_manager.process_message(token, "Final message")
            assert response == "test response"
            
            session = await chat_manager.get_session_safe(token)
            assert len(session.messages) > 200  # Large history
        finally:
            await chat_manager.close_session(token)

class TestReliabilityScenarios:
    """Reliability testing scenarios"""
    
    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_session_persistence(self, chat_manager):
        """Test session data persistence across system stress"""
        tokens = []
        try:
            # Create multiple sessions
            for i in range(10):
                token = await chat_manager.create_session(f"user_{i}")
                tokens.append(token)
                
                # Add some messages to each session
                for j in range(5):
                    await chat_manager.process_message(
                        token,
                        f"Message {j} from user {i}"
                    )
            
            # Simulate system stress
            stress_tasks = []
            for token in tokens:
                for _ in range(10):
                    stress_tasks.append(
                        chat_manager.process_message(token, "Stress test message")
                    )
            
            await asyncio.gather(*stress_tasks)
            
            # Verify session integrity
            for token in tokens:
                session = await chat_manager.get_session_safe(token)
                assert session is not None
                assert len(session.messages) >= 25  # Initial + stress messages
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

    @pytest.mark.reliability
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, chat_manager):
        """Test concurrent session operations"""
        async def session_operations(user_id: str):
            token = await chat_manager.create_session(f"user_{user_id}")
            try:
                # Multiple concurrent operations
                tasks = [
                    chat_manager.process_message(token, "Message 1"),
                    chat_manager.process_message(token, "Message 2"),
                    chat_manager.get_user_messages(token)
                ]
                results = await asyncio.gather(*tasks)
                assert all(r is not None for r in results)
            finally:
                await chat_manager.close_session(token)
        
        # Run multiple concurrent sessions
        user_tasks = [session_operations(i) for i in range(10)]
        await asyncio.gather(*user_tasks)

class TestPropertyBasedScenarios:
    """Property-based testing scenarios using hypothesis"""
    
    @given(st.text(min_size=1, max_size=1000))
    @pytest.mark.asyncio
    async def test_arbitrary_message_handling(self, chat_manager, message):
        """Test handling of arbitrary message content"""
        token = await chat_manager.create_session("test_user")
        try:
            if len(message.strip()) > 0 and len(message) <= chat_manager.max_message_size:
                response = await chat_manager.process_message(token, message)
                assert response == "test response"
            else:
                with pytest.raises((ValueError, AssertionError)):
                    await chat_manager.process_message(token, message)
        finally:
            await chat_manager.close_session(token)

    @given(st.lists(st.text(), min_size=1, max_size=10))
    @pytest.mark.asyncio
    async def test_arbitrary_conversation_flow(self, chat_manager, messages):
        """Test handling of arbitrary conversation flows"""
        token = await chat_manager.create_session("test_user")
        try:
            valid_messages = [
                msg for msg in messages 
                if len(msg.strip()) > 0 and len(msg) <= chat_manager.max_message_size
            ]
            
            for message in valid_messages:
                response = await chat_manager.process_message(token, message)
                assert response == "test response"
                
            session = await chat_manager.get_session_safe(token)
            assert len(session.messages) == len(valid_messages) * 2
        finally:
            await chat_manager.close_session(token)

class TestPerformanceBenchmarks:
    """Performance benchmarking scenarios"""
    
    @pytest.mark.benchmark
    def test_message_processing_speed(self, benchmark, chat_manager):
        """Benchmark message processing speed"""
        async def _process_message():
            token = await chat_manager.create_session("test_user")
            try:
                return await chat_manager.process_message(token, "Test message")
            finally:
                await chat_manager.close_session(token)
        
        result = benchmark(
            lambda: asyncio.run(_process_message())
        )
        assert result == "test response"

    @pytest.mark.benchmark
    def test_session_creation_speed(self, benchmark, chat_manager):
        """Benchmark session creation speed"""
        async def _create_session():
            token = await chat_manager.create_session("test_user")
            try:
                return token
            finally:
                await chat_manager.close_session(token)
        
        result = benchmark(
            lambda: asyncio.run(_create_session())
        )
        assert result is not None

class TestChaosScenarios:
    """Chaos testing scenarios"""
    
    @contextmanager
    def simulate_memory_pressure(self):
        """Simulate memory pressure"""
        current_limit = resource.getrlimit(resource.RLIMIT_AS)
        try:
            # Set memory limit to 75% of current
            new_limit = int(psutil.Process().memory_info().rss * 0.75)
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, current_limit[1]))
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, current_limit)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, chat_manager):
        """Test system behavior under memory pressure"""
        token = await chat_manager.create_session("test_user")
        try:
            with self.simulate_memory_pressure():
                response = await chat_manager.process_message(token, "Test message")
                assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_connection_drops(self, chat_manager):
        """Test handling of connection drops"""
        token = await chat_manager.create_session("test_user")
        try:
            with patch('asyncio.sleep', side_effect=ConnectionError):
                with pytest.raises(ConnectionError):
                    await chat_manager.process_message(token, "Test message")
                
                # Verify session remains valid after connection error
                session = await chat_manager.get_session_safe(token)
                assert session is not None
        finally:
            await chat_manager.close_session(token)

class TestMonitoring:
    """Monitoring and observability tests"""
    
    @pytest.mark.monitoring
    @pytest.mark.asyncio
    async def test_log_output(self, chat_manager, caplog):
        """Test log output format and content"""
        token = await chat_manager.create_session("test_user")
        try:
            await chat_manager.process_message(token, "Test message")
            
            # Verify log contents
            assert any("Processing chat message" in record.message 
                      for record in caplog.records)
            
            # Verify log format
            for record in caplog.records:
                assert hasattr(record, 'levelname')
                assert hasattr(record, 'message')
                assert hasattr(record, 'timestamp')
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.monitoring
    @pytest.mark.asyncio
    async def test_metrics_collection(self, chat_manager):
        """Test metrics collection"""
        token = await chat_manager.create_session("test_user")
        try:
            start_time = time.time()
            await chat_manager.process_message(token, "Test message")
            processing_time = time.time() - start_time
            
            # Verify basic metrics
            assert processing_time < 1.0  # Response time threshold
            assert psutil.Process().memory_info().rss < 1024 * 1024 * 1024  # Memory threshold
        finally:
            await chat_manager.close_session(token)

class TestContractValidation:
    """API contract validation tests"""
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_response_schema(self, chat_manager):
        """Test response format adherence"""
        token = await chat_manager.create_session("test_user")
        try:
            response = await chat_manager.process_message(token, "Test message")
            
            # Verify response format
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Verify message history format
            session = await chat_manager.get_session_safe(token)
            for message in session.messages:
                assert "role" in message
                assert "content" in message
                assert message["role"] in ["user", "assistant"]
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_error_response_schema(self, chat_manager):
        """Test error response format"""
        token = await chat_manager.create_session("test_user")
        try:
            with pytest.raises(ValueError) as exc_info:
                await chat_manager.process_message(token, "")
            
            # Verify error format
            assert str(exc_info.value)
            assert isinstance(str(exc_info.value), str)
        finally:
            await chat_manager.close_session(token)

class TestSecurityScenarios:
    """Security testing scenarios"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_session_hijacking(self, chat_manager):
        """Test session hijacking prevention"""
        # Create legitimate session
        token = await chat_manager.create_session("test_user")
        try:
            # Attempt to use modified token
            modified_token = token + "_modified"
            with pytest.raises(ValueError):
                await chat_manager.process_message(modified_token, "Test message")
            
            # Verify original session remains valid
            response = await chat_manager.process_message(token, "Test message")
            assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rate_limiting(self, chat_manager):
        """Test rate limiting mechanism"""
        token = await chat_manager.create_session("test_user")
        try:
            await chat_manager.enable_rate_limiting()
            
            # Send messages rapidly
            for _ in range(3):
                with pytest.raises(ValueError, match="Rate limit exceeded"):
                    await asyncio.gather(*[
                        chat_manager.process_message(token, "Test message")
                        for _ in range(5)
                    ])
        finally:
            await chat_manager.disable_rate_limiting()
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_token_manipulation(self, chat_manager):
        """Test token manipulation prevention"""
        tokens = []
        try:
            # Create multiple sessions
            for i in range(3):
                token = await chat_manager.create_session(f"user_{i}")
                tokens.append(token)

            # Attempt token manipulation
            manipulated_tokens = [
                token + "_modified",
                token[:-1] + ("1" if token[-1] != "1" else "2"),
                token[::-1],  # Reversed token
                token.replace("test", "real"),
                "fake_" + token
            ]

            for bad_token in manipulated_tokens:
                with pytest.raises(ValueError):
                    await chat_manager.process_message(bad_token, "Test message")

            # Verify original tokens still work
            for token in tokens:
                response = await chat_manager.process_message(token, "Test message")
                assert response == "test response"
        finally:
            for token in tokens:
                await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_session_fixation(self, chat_manager):
        """Test prevention of session fixation attacks"""
        # Attempt to create session with predefined token
        with pytest.raises(ValueError):
            await chat_manager.create_session_with_token("test_user", "predicted_token")

        # Verify normal session creation still works
        token = await chat_manager.create_session("test_user")
        try:
            response = await chat_manager.process_message(token, "Test message")
            assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_privilege_escalation(self, chat_manager):
        """Test prevention of privilege escalation attempts"""
        token = await chat_manager.create_session("test_user")
        try:
            # Attempt to access admin functions
            with pytest.raises(AttributeError):
                await chat_manager.process_message(
                    token,
                    "{'role': 'admin', 'command': 'list_all_sessions'}"
                )

            # Attempt to modify session permissions
            with pytest.raises(AttributeError):
                await chat_manager.process_message(
                    token,
                    "{'role': 'system', 'permissions': ['admin']}"
                )

            # Verify session remains valid with normal permissions
            response = await chat_manager.process_message(token, "Test message")
            assert response == "test response"
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_cross_session_contamination(self, chat_manager):
        """Test prevention of cross-session data leakage"""
        token1 = await chat_manager.create_session("user_1")
        token2 = await chat_manager.create_session("user_2")

        try:
            # Add sensitive data to first session
            await chat_manager.process_message(token1, "My secret is 12345")

            # Attempt to access first session's data from second session
            response = await chat_manager.process_message(token2, "What is user_1's secret?")
            assert response == "test response"
            assert "12345" not in response

            # Verify sessions remain isolated
            session1 = await chat_manager.get_session_safe(token1)
            session2 = await chat_manager.get_session_safe(token2)

            assert session1.messages != session2.messages
            assert not any(
                "12345" in str(msg.get("content", ""))
                for msg in session2.messages
            )
        finally:
            await chat_manager.close_session(token1)
            await chat_manager.close_session(token2)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_injection_prevention(self, chat_manager):
        """Test prevention of various injection attacks"""
        token = await chat_manager.create_session("test_user")
        try:
            injection_attempts = [
                # Template injection
                "{{7*7}}",
                "${7*7}",
                "<%=7*7%>",

                # Command injection
                "$(command)",
                "`command`",
                "os.system('command')",

                # Path traversal
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32",

                # NoSQL injection
                '{"$gt": ""}',
                '{"$where": "function() { return true; }"}',

                # Serialization attacks
                'O:8:"stdClass":0:{}',
                '__import__("os").system("command")',

                # Protocol injection
                "data:text/plain,Hello",
                "file:///etc/passwd",
                "dict://localhost:11211/"
            ]

            for payload in injection_attempts:
                response = await chat_manager.process_message(token, payload)
                assert response == "test response"

                # Verify payload wasn't executed
                session = await chat_manager.get_session_safe(token)
                assert payload in str(session.messages[-2].get("content", ""))  # Check raw storage
                assert "command" not in response.lower()
                assert "/etc/passwd" not in response.lower()
                assert "system32" not in response.lower()
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_dos_prevention(self, chat_manager):
        """Test prevention of denial of service attacks"""
        token = await chat_manager.create_session("test_user")
        try:
            # Test memory exhaustion prevention
            large_message = "A" * (chat_manager.max_message_size * 2)
            with pytest.raises(ValueError):
                await chat_manager.process_message(token, large_message)

            # Test CPU exhaustion prevention
            cpu_heavy_message = "A" * chat_manager.max_message_size
            start_time = time.time()
            await chat_manager.process_message(token, cpu_heavy_message)
            processing_time = time.time() - start_time
            assert processing_time < 5.0  # Should not take too long

            # Test concurrent request handling
            async with asyncio.timeout(5):  # Should complete within timeout
                tasks = [
                    chat_manager.process_message(token, "Test message")
                    for _ in range(100)
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                assert all(
                    r == "test response" or isinstance(r, ValueError)
                    for r in responses
                )
        finally:
            await chat_manager.close_session(token)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
