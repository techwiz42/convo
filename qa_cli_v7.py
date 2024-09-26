import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class QuestionAnswerCLI:
    def __init__(self, model_name: str = "t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        self.sia = SentimentIntensityAnalyzer()
        self.previous_questions = set()
        self.user_name = ""
        self.sentiment_cache = {}
        
        print("All models loaded successfully!")

    @lru_cache(maxsize=1000)
    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

    def generate_question(self, context: str, sentiment: float) -> str:
        # Step 1: Generate potential topics
        topic_input = f"Extract key topics from: {context}"
        topic_ids = self.tokenize(topic_input)
        
        topic_outputs = self.model.generate(
            topic_ids,
            max_length=64,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        topics = [self.tokenizer.decode(output, skip_special_tokens=True) for output in (topic_outputs if isinstance(topic_outputs, list) else [topic_outputs])]
        print("Generated topics:", topics)
        
        # Step 2: Generate questions based on topics
        for topic in topics:
            question_input = f"Generate a question about {topic}. The question must end with a question mark."
            question_ids = self.tokenize(question_input)
            
            question_outputs = self.model.generate(
                question_ids,
                max_length=64,
                num_return_sequences=10,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9
            )
            
            questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in (question_outputs if isinstance(question_outputs, list) else [question_outputs])]
            print(f"Generated questions for topic '{topic}':", questions)
            
            filtered_questions = self.filter_questions(questions)
            if filtered_questions:
                selected_question = self.select_non_repetitive_question(filtered_questions, sentiment)
                if selected_question:
                    return selected_question

        print("Failed to generate a valid question. Falling back to default questions.")
        return self.generate_fallback_question(context, sentiment)

    def filter_questions(self, questions):
        return [q for q in questions if self.is_valid_question(q)]

    def is_valid_question(self, question):
        if len(question.split()) < 4:
            return False
        if not question.endswith('?'):
            return False
        if not re.match(r'^[A-Z]', question):
            return False
        return True

    def select_non_repetitive_question(self, questions: list, sentiment: float) -> str:
        for question in questions:
            if question not in self.previous_questions:
                self.previous_questions.add(question)
                return self.adjust_question_for_sentiment(question, sentiment)
        return None

    def adjust_question_for_sentiment(self, question: str, sentiment: float) -> str:
        if sentiment > 0.05:
            return f"That's interesting! {question}"
        elif sentiment < -0.05:
            return f"I understand this might be challenging. {question}"
        else:
            return question

    def extract_nouns(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
        return nouns

    def generate_fallback_question(self, context: str, sentiment: float) -> str:
        sentences = sent_tokenize(context)
        if not sentences:
            return "Can you provide more information about this topic?"

        all_nouns = []
        for sentence in sentences:
            all_nouns.extend(self.extract_nouns(sentence))

        if all_nouns:
            random_noun = random.choice(all_nouns)
            noun_questions = [
                f"Can you tell me more about {random_noun}?",
                f"What are your thoughts on {random_noun}?",
                f"How does {random_noun} relate to this topic?",
                f"What's the significance of {random_noun} in this context?",
                f"How would you explain {random_noun} to someone new to this topic?"
            ]
            question = random.choice(noun_questions)
        else:
            general_questions = [
                "What are your thoughts on this topic?",
                "How does this relate to your experiences?",
                "Can you elaborate on this further?",
                "What implications do you see from this?",
                "How might this concept evolve in the future?",
                "What questions does this raise for you?"
            ]
            question = random.choice(general_questions)

        return self.adjust_question_for_sentiment(question, sentiment)

    def analyze_sentiment(self, text: str) -> float:
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        sentiment = self.sia.polarity_scores(text)['compound']
        self.sentiment_cache[text] = sentiment
        return sentiment

    def generate_answers(self, question: str, context: str) -> str:
        input_text = f"answer question: {question} context: {context}"
        input_ids = self.tokenize(input_text)

        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in (outputs if isinstance(outputs, list) else [outputs])]
        sentiments = [abs(self.analyze_sentiment(answer)) for answer in answers]
        
        return answers[sentiments.index(max(sentiments))]

    def interactive_session(self) -> None:
        self.user_name = input("Please enter your name: ")
        print(f"\nNice to meet you, {self.user_name}!")
        
        context = input("Enter the initial context (or press Enter to start with a blank context): ")
        if not context:
            print("\nLet's start our Q&A session.")
            print("I will ask questions, and you can respond to each question or type 'exit' to end the session.\n")
        else:
            print("\nStarting Q&A session. I will ask questions based on the context you provided.")
            print("You can respond to each question or type 'exit' to end the session.\n")

        sentiment = 0  # Start with neutral sentiment

        while True:
            question = self.generate_question(context, sentiment)
            print(f"AI: {question}")

            user_input = input(f"{self.user_name}: ")
            if user_input.lower() == 'exit':
                break

            sentiment = self.analyze_sentiment(user_input)
            print(f"Detected sentiment: {sentiment:.2f}")

            ai_answer = self.generate_answers(question, context + " " + user_input)
            print(f"AI: {ai_answer}")

            # Filter out 'AI' and 'Answer' from the context
            filtered_context = re.sub(r'\b(AI|Answer|Response|Question):', '', context)
            context = f"{filtered_context} {user_input}"

        print(f"\nThank you for the conversation, {self.user_name}!")
        print("\nFinal context:")
        print(context)

def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5-base", help="Name of the pre-trained model to use")
    args = parser.parse_args()

    try:
        cli = QuestionAnswerCLI(args.model)
        cli.interactive_session()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
