import unittest
from unittest.mock import Mock, patch
from qa_cli_v6 import QuestionAnswerCLI

class TestQuestionAnswerCLI(unittest.TestCase):

    def setUp(self):
        # Mock the T5 model and tokenizer
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        self.mock_quality_model = Mock()
        self.mock_quality_tokenizer = Mock()

        with patch('qa_cli_v6.T5Tokenizer.from_pretrained', return_value=self.mock_tokenizer), \
             patch('qa_cli_v6.T5ForConditionalGeneration.from_pretrained', return_value=self.mock_model), \
             patch('qa_cli_v6.BertForSequenceClassification.from_pretrained', return_value=self.mock_quality_model), \
             patch('qa_cli_v6.BertTokenizer.from_pretrained', return_value=self.mock_quality_tokenizer):
            self.cli = QuestionAnswerCLI()

    def test_is_valid_question(self):
        # Test valid questions
        self.assertTrue(self.cli.is_valid_question("What is the capital of France?"))
        self.assertTrue(self.cli.is_valid_question("How does photosynthesis work?"))

        # Test invalid questions
        self.assertFalse(self.cli.is_valid_question("Too short?"))
        self.assertFalse(self.cli.is_valid_question("This is not a question"))
        self.assertFalse(self.cli.is_valid_question("What about the 's?"))
        self.assertFalse(self.cli.is_valid_question("What are your thoughts on the?"))

    def test_adjust_question_for_sentiment(self):
        # Test positive sentiment
        self.assertEqual(
            self.cli.adjust_question_for_sentiment("What is your favorite color?", 0.1),
            "That's interesting! What is your favorite color?"
        )

        # Test negative sentiment
        self.assertEqual(
            self.cli.adjust_question_for_sentiment("What challenges are you facing?", -0.1),
            "I understand this might be challenging. What challenges are you facing?"
        )

        # Test neutral sentiment
        self.assertEqual(
            self.cli.adjust_question_for_sentiment("How old are you?", 0),
            "How old are you?"
        )

    @patch('qa_cli_v6.random.choice')
    def test_generate_entity_question(self, mock_random_choice):
        mock_random_choice.return_value = "What is {entity}?"
        
        # Test PERSON entity
        self.assertEqual(
                self.cli.generate_entity_question("Albert Einstein", "PERSON", 0),
            "What is Albert Einstein?"
        )

        # Test LOCATION entity
        self.assertEqual(
            self.cli.generate_entity_question("Paris", "GPE", 0),
            "What is Paris?"
        )

        # Test ORGANIZATION entity
        self.assertEqual(
            self.cli.generate_entity_question("NASA", "ORGANIZATION", 0),
            "What is NASA?"
        )

    def test_analyze_sentiment(self):
        # Test positive sentiment
        self.assertGreater(self.cli.analyze_sentiment("I love this!"), 0)

        # Test negative sentiment
        self.assertLess(self.cli.analyze_sentiment("I hate this."), 0)

        # Test neutral sentiment
        self.assertAlmostEqual(self.cli.analyze_sentiment("The sky is blue."), 0, delta=0.1)

    def test_tokenize(self):
        self.mock_tokenizer.return_value.input_ids.to.return_value = "mocked_input_ids"
        result = self.cli.tokenize("Test input")
        self.assertEqual(result, "mocked_input_ids")
        self.mock_tokenizer.assert_called_once_with("Test input", return_tensors="pt", max_length=512, truncation=True)

    @patch('qa_cli_v6.QuestionAnswerCLI.generate_question')
    @patch('qa_cli_v6.QuestionAnswerCLI.analyze_sentiment')
    @patch('qa_cli_v6.QuestionAnswerCLI.generate_answers')
    def test_interactive_session(self, mock_generate_answers, mock_analyze_sentiment, mock_generate_question):
        mock_generate_question.return_value = "What is your favorite color?"
        mock_analyze_sentiment.return_value = 0.5
        mock_generate_answers.return_value = "That's an interesting choice!"

        # Mock input function
        with patch('builtins.input', side_effect=["John", "I like blue", "exit"]):
            self.cli.interactive_session()

        # Check if methods were called
        mock_generate_question.assert_called()
        mock_analyze_sentiment.assert_called_with("I like blue")
        mock_generate_answers.assert_called()

if __name__ == '__main__':
    unittest.main()
