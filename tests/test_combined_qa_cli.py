import unittest
from unittest.mock import MagicMock, patch
import torch
import json
import os
from combined_qa_cli import EnhancedMultiUserQuestionAnswerCLI, UserKnowledgeBase, UserSpecificModel, Conversation

class TestEnhancedMultiUserQuestionAnswerCLI(unittest.TestCase):

    def setUp(self):
        self.cli = EnhancedMultiUserQuestionAnswerCLI()
        # Initialize UserKnowledgeBase with some sample data
        self.user_id = "test_user"
        self.cli.get_or_create_user(self.user_id)
        self.cli.users[self.user_id]["knowledge_base"].add_knowledge("sample_topic", "This is a sample piece of knowledge.")
        self.cli.users[self.user_id]["knowledge_base"].add_knowledge("another_topic", "This is another sample piece of knowledge.")

    def test_get_or_create_user(self):
        user_id = "new_test_user"
        user_data = self.cli.get_or_create_user(user_id)
        self.assertIn(user_id, self.cli.users)
        self.assertIsInstance(user_data["model"], UserSpecificModel)
        self.assertIsInstance(user_data["knowledge_base"], UserKnowledgeBase)
        self.assertIsInstance(user_data["conversation"], Conversation)

    def test_tokenize(self):
        text = f"{self.user_id}: Hello, world!"
        tokenized = self.cli.tokenize(text)
        self.assertIsInstance(tokenized, torch.Tensor)

    @patch('combined_qa_cli.UserSpecificModel.generate_response')
    def test_generate_questions(self, mock_generate):
        mock_generate.return_value = "Generated question?"
        questions = self.cli.generate_questions(f"{self.user_id}: Some context", 0.0)
        self.assertIsInstance(questions, list)
        self.assertTrue(all(q.endswith('?') for q in questions))

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
        self.assertEqual(self.cli.determine_input_type("Is this a question?", self.user_id), "question")
        self.assertEqual(self.cli.determine_input_type("This is a statement.", self.user_id), "statement")

    @patch('combined_qa_cli.UserSpecificModel.generate_response')
    @patch('combined_qa_cli.UserSpecificModel.fine_tune')
    def test_handle_user_question(self, mock_fine_tune, mock_generate):
        mock_generate.return_value = "Generated answer"
        response = self.cli.handle_user_question("Test question?", self.user_id)
        self.assertIn("Generated answer", response)
        mock_fine_tune.assert_called_once()

    @patch('combined_qa_cli.UserSpecificModel.generate_response')
    @patch('combined_qa_cli.UserSpecificModel.fine_tune')
    def test_handle_user_answer(self, mock_fine_tune, mock_generate):
        mock_generate.return_value = "Generated response"
        response = self.cli.handle_user_answer("Test answer", self.user_id)
        self.assertIn("Thank you for your answer", response)
        mock_fine_tune.assert_called_once()

    @patch('combined_qa_cli.UserSpecificModel.generate_response')
    @patch('combined_qa_cli.UserSpecificModel.fine_tune')
    def test_handle_user_statement(self, mock_fine_tune, mock_generate):
        mock_generate.return_value = "Generated response"
        response = self.cli.handle_user_statement("Test statement", self.user_id)
        self.assertIn("Interesting point", response)
        mock_fine_tune.assert_called_once()

    @patch('combined_qa_cli.UserSpecificModel.save')
    @patch('combined_qa_cli.UserKnowledgeBase.save')
    def test_save_user_data(self, mock_kb_save, mock_model_save):
        self.cli.save_user_data(self.user_id)
        mock_model_save.assert_called_once()
        mock_kb_save.assert_called_once()
        self.assertTrue(os.path.exists(f"{self.user_id}_conversation.json"))
        os.remove(f"{self.user_id}_conversation.json")

    @patch('combined_qa_cli.UserSpecificModel.load')
    @patch('combined_qa_cli.UserKnowledgeBase.load')
    def test_load_user_data(self, mock_kb_load, mock_model_load):
        mock_model_load.return_value = True
        mock_kb_load.return_value = True
        result = self.cli.load_user_data(self.user_id)
        self.assertTrue(result)
        mock_model_load.assert_called_once()
        mock_kb_load.assert_called_once()

    @patch('combined_qa_cli.EnhancedMultiUserQuestionAnswerCLI.handle_user_question')
    def test_process_user_input_question(self, mock_handle_question):
        mock_handle_question.return_value = "Processed question response"
        response = self.cli.process_user_input(self.user_id, "Is this a test question?")
        self.assertEqual(response, "Processed question response")
        mock_handle_question.assert_called_once()

    @patch('combined_qa_cli.EnhancedMultiUserQuestionAnswerCLI.handle_user_statement')
    def test_process_user_input_statement(self, mock_handle_statement):
        mock_handle_statement.return_value = "Processed statement response"
        response = self.cli.process_user_input(self.user_id, "This is a test statement.")
        self.assertEqual(response, "Processed statement response")
        mock_handle_statement.assert_called_once()

if __name__ == '__main__':
    unittest.main()
