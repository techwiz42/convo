# tests/test_swarm_chat_extended.py

import pytest
import asyncio
from typing import List, Dict
import json
from unittest.mock import Mock, patch

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

class TestEdgeCases:
    """Edge case testing"""
    
    @pytest.mark.asyncio
    async def test_message_boundary_conditions(self, chat_manager):
        """Test message boundary conditions"""
        token = await chat_manager.create_session("test_user")
        try:
            # Test exact size limit
            msg_at_limit = "x" * chat_manager.max_message_size
            response = await chat_manager.process_message(token, msg_at_limit)
            assert response == "test response"
            
            # Test one character below limit
            msg_below_limit = "x" * (chat_manager.max_message_size - 1)
            response = await chat_manager.process_message(token, msg_below_limit)
            assert response == "test response"
            
            # Test one character above limit
            msg_above_limit = "x" * (chat_manager.max_message_size + 1)
            with pytest.raises(ValueError):
                await chat_manager.process_message(token, msg_above_limit)
        finally:
            await chat_manager.close_session(token)

    @pytest.mark.asyncio
    async def test_json_message_handling(self, chat_manager):
        """Test handling of JSON formatted messages"""
        token = await chat_manager.create_session("test_user")
        try:
            json_messages = [
                json.dumps({"type": "greeting", "content": "Hello"}),
                json.dumps({"type": "query", "content": "How are you?"}),
                json.dumps({"type": "command", "action": "do_something"}),
                # Invalid JSON but valid string
                "{invalid_json: content}",
                # Nested JSON
                json.dumps({
                    "outer": {
                        "inner": {
                            "deep": "content"
                        }
                    }
                })
            ]
            
            for msg in json_messages:
                response = await chat_manager.process_message(token, msg)
                assert response == "test response"
        finally:
            await chat_manager.close_session(token)
