import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_cli_v8 import QuestionAnswerCLI

class TestEnhancedQuestionAnswerCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "t5-small"  # Using a smaller model for faster testing
        cls.qa_cli = QuestionAnswerCLI(cls.model_name)

    def test_determine_input_type_question(self):
        input_text = "What is the capital of France?"
        input_type = self.qa_cli.determine_input_type(input_text)
        self.assertEqual(input_type, "question")

    def test_determine_input_type_answer(self):
        self.qa_cli.last_question = "What is the capital of France?"
        input_text = "The capital of France is Paris."
        input_type = self.qa_cli.determine_input_type(input_text)
        self.assertEqual(input_type, "answer")

    def test_determine_input_type_statement(self):
        input_text = "I like to travel to different countries."
        input_type = self.qa_cli.determine_input_type(input_text)
        self.assertEqual(input_type, "statement")

    def test_handle_user_question(self):
        question = "What is the tallest mountain in the world?"
        context = "We were discussing geography and natural landmarks."
        response = self.qa_cli.handle_user_question(question, context)
        self.assertTrue(response.startswith("To answer your question:"))
        self.assertIsNone(self.qa_cli.last_question)

    def test_handle_user_answer(self):
        self.qa_cli.last_question = "What is your favorite color?"
        answer = "My favorite color is blue because it reminds me of the sky."
        context = "We were discussing personal preferences."
        response = self.qa_cli.handle_user_answer(answer, context)
        self.assertTrue(response.startswith("Thank you for your answer."))
        self.assertIsNotNone(self.qa_cli.last_question)
        self.assertNotEqual(self.qa_cli.last_question, "What is your favorite color?")

    def test_handle_user_statement(self):
        statement = "I enjoy hiking in the mountains during summer."
        context = "We were discussing hobbies and outdoor activities."
        response = self.qa_cli.handle_user_statement(statement, context)
        self.assertTrue(response.startswith("Interesting point."))
        self.assertIsNotNone(self.qa_cli.last_question)

    def test_interactive_session(self):
        # Mock the input function to simulate user interactions
        user_inputs = [
            "Alice",  # User name
            "Let's talk about space exploration.",  # Initial context
            "What was the first moon landing mission?",  # User question
            "That's fascinating. Humans have come a long way in space technology.",  # User statement
            "Yes, the International Space Station is a great example of international cooperation.",  # User answer
            "exit"  # End the session
        ]
        
        def mock_input(prompt):
            return user_inputs.pop(0)

        with unittest.mock.patch('builtins.input', mock_input):
            with unittest.mock.patch('builtins.print') as mock_print:
                self.qa_cli.interactive_session()

        # Check that the correct number of interactions occurred
        self.assertEqual(mock_print.call_count, 9)  # Adjust this number based on your exact print statements

        # Check that the session ended correctly
        mock_print.assert_any_call("\nThank you for the conversation, Alice!")

    def test_generate_questions(self):
        context = "The Apollo 11 mission in 1969 was the first to land humans on the Moon."
        sentiment = 0.0
        questions = self.qa_cli.generate_questions(context, sentiment)
        self.assertGreater(len(questions), 0)
        self.assertLessEqual(len(questions), 5)

        for question in questions:
            self.assertIsInstance(question, str)
            self.assertTrue(question.endswith('?'))

        relevant_keywords = ['Apollo', '11', 'mission', '1969', 'Moon', 'land']
        questions_contain_keywords = any(
            any(keyword.lower() in question.lower() for keyword in relevant_keywords)
            for question in questions
        )
        self.assertTrue(questions_contain_keywords, "None of the questions contain relevant keywords")

    def test_generate_answer(self):
        question = "Who were the first astronauts to walk on the Moon?"
        context = "The Apollo 11 mission in 1969 was the first to land humans on the Moon. Neil Armstrong and Buzz Aldrin were part of this historic mission."

        answer = self.qa_cli.generate_answer(question, context)

        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

        relevant_keywords = ['Neil Armstrong', 'Buzz Aldrin', 'Apollo 11', 'astronauts']
        self.assertTrue(any(keyword.lower() in answer.lower() for keyword in relevant_keywords),
                        f"Answer '{answer}' doesn't contain any relevant keywords")

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

if __name__ == '__main__':
    unittest.main()
