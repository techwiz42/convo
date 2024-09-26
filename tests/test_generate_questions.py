import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_cli_v7 import QuestionAnswerCLI

class TestGenerateQuestions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This setup will be run once for the entire test class
        cls.model_name = "t5-small"  # Using a smaller model for faster testing
        cls.qa_cli = QuestionAnswerCLI(cls.model_name)

    def test_generate_questions(self):
        context = "The Eiffel Tower, located in Paris, France, was completed in 1889. It stands at a height of 324 meters and was the tallest man-made structure in the world for 41 years."
        sentiment = 0.00        
        questions = self.qa_cli.generate_questions(context, sentiment)
        
        # Assert that we get the expected number of questions
        self.assertEqual(len(questions), 5)  # Assuming the function generates 5 questions
        
        # Check that each question is a non-empty string
        for question in questions:
            self.assertIsInstance(question, str)
            self.assertTrue(len(question) > 0)
        
        # Check that the questions are related to the context
        relevant_keywords = ['Eiffel', 'Tower', 'Paris', 'France', '1889', '324', 'meters', 'tallest']
        for question in questions:
            self.assertTrue(any(keyword.lower() in question.lower() for keyword in relevant_keywords))
        
        # Check that the questions are different from each other
        self.assertEqual(len(set(questions)), len(questions), "All questions should be unique")

    def test_generate_questions_empty_context(self):
        context = None
        sentiment = 0.00    
        with self.assertRaises(ValueError):
            self.qa_cli.generate_questions(context, sentiment)

    def test_generate_questions_long_context(self):
        context = "A" * 1000  # A very long context
        sentiment = 0.95 
        questions = self.qa_cli.generate_questions(context, sentiment)
        
        # Assert that we still get questions even with a long context
        self.assertTrue(len(questions) > 0)

if __name__ == '__main__':
    unittest.main()
