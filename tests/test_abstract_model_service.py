import unittest
from unittest.mock import MagicMock, patch
import torch
import json
import os
import tempfile
from transformers import PreTrainedTokenizer
from model_implementations import T5LanguageModel, BERTLanguageModel, GPT2LanguageModel#, RoBERTaLanguageModel
from abstract_model_service import EnhancedMultiUserQuestionAnswerCLI, UserKnowledgeBase, Conversation

class TestAbstractedQACLI(unittest.TestCase):

    def setUp(self):
        self.model_factory = lambda user_id: MagicMock()
        self.cli = EnhancedMultiUserQuestionAnswerCLI(self.model_factory)
        self.user_id = "test_user"

    def test_get_or_create_user(self):
        user_data = self.cli.get_or_create_user(self.user_id)
        self.assertIn(self.user_id, self.cli.users)
        self.assertIsInstance(user_data["model"], MagicMock)
        self.assertIsInstance(user_data["knowledge_base"], UserKnowledgeBase)
        self.assertIsInstance(user_data["conversation"], Conversation)

    def test_tokenize(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.input_ids = torch.tensor([1, 2, 3])
        self.cli.users[self.user_id] = {"model": MagicMock(get_tokenizer=lambda: mock_tokenizer)}
        
        result = self.cli.tokenize(f"{self.user_id}: Hello, world!")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.tolist(), [1, 2, 3])

    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.filter_questions')
    def test_generate_questions(self, mock_filter):
        mock_model = MagicMock()
        mock_model.generate_response.return_value = "Generated question?"
        self.cli.users[self.user_id] = {"model": mock_model}
        mock_filter.return_value = ["Filtered question?"]

        questions = self.cli.generate_questions(f"{self.user_id}: Some context", 0.0)
        self.assertIsInstance(questions, list)
        self.assertEqual(questions, ["Filtered question?"])

    def test_filter_questions(self):
        questions = ["Is this a valid question?", "Invalid", "Is this another valid question?"]
        filtered = self.cli.filter_questions(questions)
        self.assertEqual(len(filtered), 2)
        self.assertIn("Is this a valid question?", filtered)
        self.assertIn("Is this another valid question?", filtered)

    def test_is_valid_question(self):
        self.assertTrue(self.cli.is_valid_question("Is this a valid question?"))
        self.assertFalse(self.cli.is_valid_question("invalid"))
        self.assertFalse(self.cli.is_valid_question("This is not a question"))

    def test_extract_entities(self):
        text = "The quick brown fox jumps over the lazy dog"
        entities = self.cli.extract_entities(text)
        self.assertIn("fox", entities)
        self.assertIn("dog", entities)

    def test_generate_entity_questions(self):
        questions = self.cli.generate_entity_questions("AI")
        self.assertEqual(len(questions), 3)
        self.assertTrue(all("AI" in q for q in questions))

    def test_select_question(self):
        self.cli.users[self.user_id] = {"conversation": Conversation(self.user_id)}
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        selected = self.cli.select_question(questions, 0.0, self.user_id)
        self.assertIn(selected, questions)

    def test_adjust_question_for_sentiment(self):
        question = "Is this a test?"
        positive = self.cli.adjust_question_for_sentiment(question, 0.1)
        negative = self.cli.adjust_question_for_sentiment(question, -0.1)
        neutral = self.cli.adjust_question_for_sentiment(question, 0.0)
        self.assertIn("interesting", positive.lower())
        self.assertIn("challenging", negative.lower())
        self.assertEqual(neutral, question)

    def test_analyze_sentiment(self):
        text = "I love this test!"
        sentiment = self.cli.analyze_sentiment(text)
        self.assertGreater(sentiment, 0)

    def test_determine_input_type(self):
        self.cli.users[self.user_id] = {"conversation": Conversation(self.user_id)}
        self.assertEqual(self.cli.determine_input_type("Is this a question?", self.user_id), "question")
        self.assertEqual(self.cli.determine_input_type("This is a statement.", self.user_id), "statement")

    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.get_or_create_user')
    def test_handle_user_question(self, mock_get_user):
        mock_model = MagicMock()
        mock_model.generate_response.return_value = "Generated answer"
        mock_get_user.return_value = {
            "model": mock_model,
            "knowledge_base": MagicMock(),
            "conversation": Conversation(self.user_id)
        }
        response = self.cli.handle_user_question("Test question?", self.user_id)
        self.assertIn("Generated answer", response)
        mock_model.fine_tune.assert_called_once()

    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.get_or_create_user')
    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.generate_questions')
    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.select_question')
    def test_handle_user_answer(self, mock_select, mock_generate, mock_get_user):
        mock_model = MagicMock()
        mock_model.generate_response.return_value = "Generated response"
        mock_get_user.return_value = {
            "model": mock_model,
            "knowledge_base": MagicMock(),
            "conversation": Conversation(self.user_id)
        }
        mock_generate.return_value = ["Generated question?"]
        mock_select.return_value = "Selected question?"
        
        response = self.cli.handle_user_answer("Test answer", self.user_id)
        self.assertIn("Thank you for your answer", response)
        self.assertIn("Selected question?", response)
        mock_model.fine_tune.assert_called_once()

    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.get_or_create_user')
    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.generate_questions')
    @patch('abstract_model_service.EnhancedMultiUserQuestionAnswerCLI.select_question')
    def test_handle_user_statement(self, mock_select, mock_generate, mock_get_user):
        mock_model = MagicMock()
        mock_model.generate_response.return_value = "Generated response"
        mock_get_user.return_value = {
            "model": mock_model,
            "knowledge_base": MagicMock(),
            "conversation": Conversation(self.user_id)
        }
        mock_generate.return_value = ["Generated question?"]
        mock_select.return_value = "Selected question?"
        
        response = self.cli.handle_user_statement("Test statement", self.user_id)
        self.assertIn("Interesting point", response)
        self.assertIn("Selected question?", response)
        mock_model.fine_tune.assert_called_once()

class TestLanguageModels(unittest.TestCase):

    def _test_model(self, model):
        # Test generate_response
        response = model.generate_response(self.test_input)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

        # Test fine_tune
        try:
            model.fine_tune(self.test_input, self.test_target)
        except Exception as e:
            self.fail(f"Fine-tuning failed with error: {str(e)}")

        # Test save and load
        model.save(self.test_path)
        self.assertTrue(model.load(self.test_path))

        # Test get_tokenizer
        tokenizer = model.get_tokenizer()
        self.assertIsInstance(tokenizer, PreTrainedTokenizer)
    def setUp(self):
        self.user_id = "test_user"
        self.test_input = "This is a test input."
        self.test_target = "This is a test target."
        self.test_path = tempfile.mkdtemp()

    def test_t5_model(self):
        model = T5LanguageModel(self.user_id)
        self._test_model(model)

    def test_bert_model(self):
        model = BERTLanguageModel(self.user_id)
        self._test_model(model)

    def test_gpt2_model(self):
        model = GPT2LanguageModel(self.user_id)
        self._test_model(model)
    """ 
    def test_roberta_model(self):
        model = RoBERTaLanguageModel(self.user_id)
        self._test_model(model)
    """
    def _test_model(self, model):
        # Test generate_response
        response = model.generate_response(self.test_input)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

        # Test fine_tune
        model.fine_tune(self.test_input, self.test_target)

        # Test save and load
        model.save(self.test_path)
        self.assertTrue(model.load(self.test_path))

        # Test get_tokenizer
        tokenizer = model.get_tokenizer()
        self.assertIsInstance(tokenizer, PreTrainedTokenizer)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_path)

if __name__ == '__main__':
    unittest.main()
