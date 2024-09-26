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
        #relevant_keywords = ['Eiffel', 'Tower', 'Paris', 'France', '1889', '324', 'meters', 'tallest']
        #for question in questions:
        #    self.assertTrue(any(keyword.lower() in question.lower() for keyword in relevant_keywords))

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


class TestGenerateAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This setup will be run once for the entire test class
        cls.model_name = "t5-small"  # Using a smaller model for faster testing
        cls.qa_cli = QuestionAnswerCLI(cls.model_name)

    def test_generate_answers(self):
        question = "What is the capital of France?"
        context = "France is a country in Western Europe. Its capital city is Paris, which is known for its iconic Eiffel Tower and Louvre museum."

        answers = self.qa_cli.generate_answers(question, context)

        # Assert that we get the expected number of answers
        self.assertEqual(len(answers), 5, "Should generate 5 answers")

        # Check that each answer is a non-empty string
        for answer in answers:
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 0)

        # Check that the answers are related to the question and context
        relevant_keywords = ['Paris', 'capital', 'France']
        for answer in answers:
            self.assertTrue(any(keyword.lower() in answer.lower() for keyword in relevant_keywords),
                            f"Answer '{answer}' doesn't contain any relevant keywords")

        # Check that all answers are unique
        self.assertEqual(len(set(answers)), 5, "All answers should be unique")

        # Print answers for inspection
        print("Generated answers:")
        for i, answer in enumerate(answers, 1):
            print(f"{i}. {answer}")

    def test_generate_answers_no_context(self):
        question = "What is the speed of light?"
        context = ""

        answers = self.qa_cli.generate_answers(question, context)
        # Assert that we still get answers even without context
        self.assertTrue(len(answers) > 0)

        # Check that the answers are related to the question
        # relevant_keywords = ['light', 'speed']
        # for answer in answers:
        #    self.assertTrue(any(keyword.lower() in answer.lower() for keyword in relevant_keywords))

    def test_generate_answers_long_input(self):
        question = "What are the main themes of the novel?"
        context = " The novel explores themes of love, loss, and redemption."

        answers = self.qa_cli.generate_answers(question, context)

        # Assert that we still get answers even with a long context
        self.assertTrue(len(answers) > 0)

        # Check that the answers are related to the themes mentioned at the end of the context
        relevant_keywords = ['love', 'loss', 'redemption']
        for answer in answers:
            print(f"answer: {answer}")
            self.assertTrue(any(keyword.lower() in answer.lower() for keyword in relevant_keywords))

if __name__ == '__main__':
    unittest.main()
