import unittest
from unittest.mock import Mock, patch
import asyncio
import logging

# Import your modules here
from async_qa_service import AsyncEnhancedMultiUserQuestionAnswerCLI
from model_implementations import create_model
from user_knowledge_base import UserKnowledgeBase
from text_analysis import TextAnalyzer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestUserKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = UserKnowledgeBase("test_user")

    def test_add_knowledge(self):
        self.kb.add_knowledge("test_topic", "test_content")
        self.assertIn("test_topic", self.kb.knowledge)
        self.assertEqual(self.kb.knowledge["test_topic"], "test_content")

    def test_get_relevant_knowledge(self):
        self.kb.add_knowledge("topic1", "content1")
        self.kb.add_knowledge("topic2", "content2")
        relevant = self.kb.get_relevant_knowledge("topic1")
        self.assertIn("content1", relevant)

class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    def test_analyze_sentiment(self):
        sentiment = self.analyzer.analyze_sentiment("I love this!")
        self.assertGreater(sentiment, 0)

    def test_basic_grammar_check(self):
        score = self.analyzer.basic_grammar_check("This is a correct sentence.")
        self.assertGreater(score, 0.5)

class TestModelImplementations(unittest.TestCase):
    @patch('model_implementations.GPT2LanguageModel')
    def test_create_model_gpt2(self, mock_gpt2):
        model = create_model('gpt2', './models')
        self.assertIsNotNone(model)
        mock_gpt2.assert_called_once()

    @patch('model_implementations.T5LanguageModel')
    def test_create_model_t5(self, mock_t5):
        model = create_model('t5', './models')
        self.assertIsNotNone(model)
        mock_t5.assert_called_once()

class TestAsyncEnhancedMultiUserQuestionAnswerCLI(unittest.TestCase):
    def setUp(self):
        self.model_mock = Mock()
        self.cli = AsyncEnhancedMultiUserQuestionAnswerCLI(self.model_mock)

    @patch('user_knowledge_base.UserKnowledgeBase')
    def test_process_user_input(self, mock_kb):
        mock_kb.return_value.get_relevant_knowledge.return_value = []
        self.model_mock.generate_response.return_value = "model response"
        
        async def run_test():
            response = await self.cli.process_user_input("user1", "test question")
            self.assertEqual(response, "model response")
            self.assertEqual(self.model_mock.generate_response.call_count, 3)
            self.model_mock.generate_response.assert_has_calls([
                unittest.mock.call('Given the context: \n\nPlease provide a response to the following: test question\n\nResponse:', temperature=0.7, top_p=0.9, max_new_tokens=100),
                unittest.mock.call('Given the context: \n\nPlease provide a response to the following: test question\n\nResponse:', temperature=1.0, top_p=1.0, max_new_tokens=150),
                unittest.mock.call('Given the context: \n\nPlease provide a response to the following: test question\n\nResponse:', temperature=0.5, top_p=0.8, max_new_tokens=200)
            ])

        asyncio.run(run_test())

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.model = create_model('gpt2', './models')
        self.cli = AsyncEnhancedMultiUserQuestionAnswerCLI(self.model)

    @patch('builtins.input', side_effect=['user1', 'What is Python?', 'exit'])
    @patch('builtins.print')
    def test_cli_interaction(self, mock_print, mock_input):
        async def run_cli():
            try:
                await self.cli.run()
            except Exception as e:
                logger.error(f"Exception in CLI: {str(e)}")
                raise

        async def run_test():
            # Start the CLI in a separate task
            cli_task = asyncio.create_task(run_cli())
            
            # Wait for a short time to allow the CLI to start
            await asyncio.sleep(0.1)
            
            # Stop the CLI
            self.cli.stop()
            
            # Wait for the CLI task to complete
            try:
                await asyncio.wait_for(cli_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.error("CLI did not stop within the expected time")
            except Exception as e:
                logger.error(f"Unexpected exception: {str(e)}")
            
            mock_print.assert_any_call("Starting Async Enhanced Multi-User Question Answer CLI")
            mock_print.assert_any_call("CLI stopped.")

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
