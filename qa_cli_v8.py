import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
import re
import random
import argparse
import json
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class QuestionAnswerCLI:
    def __init__(self, model_name: str = "t5-small"):
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
        self.last_question = None
        self.conversation_history = ""
        
        print("All models loaded successfully!")

    @lru_cache(maxsize=1000)
    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

    def generate_questions(self, context, sentiment):
        # Generate multiple sets of questions with different prompts
        prompts = [
            f"generate questions: {context}",
            f"ask about: {context}",
            f"what would you like to know about: {context}",
            f"create a list of questions based on: {context}",
            f"inquire about: {context}"
        ]
        
        all_questions = []
        for prompt in prompts:
            input_ids = self.tokenize(prompt)
            
            outputs = self.model.generate(
                input_ids,
                max_length=128,
                num_return_sequences=3,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            questions = [f"{q}?" if not q.endswith('?') else q for q in questions]
            all_questions.extend(self.filter_questions(questions))
        
        # If we still don't have any valid questions, generate some based on extracted entities
        if not all_questions:
            entities = self.extract_entities(context)
            for entity in entities:
                all_questions.extend(self.generate_entity_questions(entity))
        
        # Remove duplicates and limit to 5 questions
        all_questions = list(set(all_questions))[:5]
        
        # If we still don't have any questions, use improved fallback questions
        if not all_questions:
            all_questions = self.generate_improved_fallback_questions(context)
        
        return all_questions

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

    def extract_entities(self, text):
        words = word_tokenize(text)
        tagged = pos_tag(words)
        entities = [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('JJ')]
        return list(set(entities))  # Remove duplicates

    def generate_entity_questions(self, entity):
        templates = [
            f"What is the significance of {entity}?",
            f"How does {entity} relate to the topic?",
            f"Can you elaborate on {entity}?",
            f"What are the key aspects of {entity}?",
            f"How would you describe {entity} to someone unfamiliar with it?"
        ]
        return random.sample(templates, min(3, len(templates)))

    def generate_improved_fallback_questions(self, context):
        general_questions = [
            "What are the main points we've discussed so far?",
            "How does this information relate to broader concepts in this field?",
            "What potential implications or applications can we derive from this?",
            "Are there any alternative perspectives or theories we should consider?",
            "How has our understanding of this topic evolved over time?",
            "What questions remain unanswered in this area of study?",
            "How might this information be relevant to current events or future developments?",
            "What are some potential challenges or limitations associated with this topic?",
            "How does this compare to similar concepts or ideas in other fields?",
            "What further research or exploration would be beneficial in this area?"
        ]
        return random.sample(general_questions, 5)

    def select_question(self, questions: list, sentiment: float) -> str:
        for question in questions:
            if question not in self.previous_questions:
                self.previous_questions.add(question)
                return self.adjust_question_for_sentiment(question, sentiment)
        # If all questions have been used, reset the previous_questions set and try again
        self.previous_questions.clear()
        return self.select_question(questions, sentiment)

    def adjust_question_for_sentiment(self, question: str, sentiment: float) -> str:
        if sentiment > 0.05:
            return f"That's interesting! {question}"
        elif sentiment < -0.05:
            return f"I understand this might be challenging. {question}"
        else:
            return question

    def analyze_sentiment(self, text: str) -> float:
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        sentiment = self.sia.polarity_scores(text)['compound']
        self.sentiment_cache[text] = sentiment
        return sentiment

    def generate_answer(self, question, context):
        max_context_length = 384
        input_text = f"question: {question} context: {context[:max_context_length]}"
        input_ids = self.tokenize(input_text)
    
        output = self.model.generate(
            input_ids,
            max_length=128,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.replace(question, "").strip()

    def determine_input_type(self, user_input: str) -> str:
        if user_input.endswith('?'):
            return "question"
        
        if self.last_question:
            input_words = set(word_tokenize(user_input.lower()))
            question_words = set(word_tokenize(self.last_question.lower()))
            common_words = input_words.intersection(question_words)
            if len(common_words) >= 2:
                return "answer"
        
        return "statement"

    def handle_user_question(self, question: str, context: str) -> str:
        answer = self.generate_answer(question, context)
        self.last_question = None
        return f"To answer your question: {answer}"

    def handle_user_answer(self, answer: str, context: str) -> str:
        sentiment = self.analyze_sentiment(answer)
        follow_up_questions = self.generate_questions(context + " " + answer, sentiment)
        selected_question = self.select_question(follow_up_questions, sentiment)
        self.last_question = selected_question
        return f"Thank you for your answer. {selected_question}"

    def handle_user_statement(self, statement: str, context: str) -> str:
        sentiment = self.analyze_sentiment(statement)
        questions = self.generate_questions(context + " " + statement, sentiment)
        selected_question = self.select_question(questions, sentiment)
        self.last_question = selected_question
        return f"Interesting point. {selected_question}"

    def save_session(self):
        session_data = {
            "user_name": self.user_name,
            "conversation_history": self.conversation_history,
            "previous_questions": list(self.previous_questions)
        }
        with open(f"{self.user_name}_session.json", "w") as f:
            json.dump(session_data, f)

    def load_session(self, user_name):
        try:
            with open(f"{user_name}_session.json", "r") as f:
                session_data = json.load(f)
            self.user_name = session_data["user_name"]
            self.conversation_history = session_data["conversation_history"]
            self.previous_questions = set(session_data["previous_questions"])
            return True
        except FileNotFoundError:
            return False

    def interactive_session(self) -> None:
        self.user_name = input("Please enter your name: ")
        if self.load_session(self.user_name):
            print(f"\nWelcome back, {self.user_name}! I've loaded your previous conversation.")
            print("\nHere's a summary of our last conversation:")
            print(self.conversation_history[-500:])  # Print last 500 characters as a summary
        else:
            print(f"\nNice to meet you, {self.user_name}!")
            self.conversation_history = input("Enter the initial context (or press Enter to start with a blank context): ")
            if not self.conversation_history:
                print("\nLet's start our conversation.")
            else:
                print("\nStarting our conversation based on the context you provided.")
        
        print("You can respond to each question, ask your own questions, or make statements. Type 'exit' to end the session.\n")

        while True:
            user_input = input(f"{self.user_name}: ")
            if user_input.lower() == 'exit':
                break

            input_type = self.determine_input_type(user_input)
            
            if input_type == "question":
                response = self.handle_user_question(user_input, self.conversation_history)
            elif input_type == "answer":
                response = self.handle_user_answer(user_input, self.conversation_history)
            else:  # statement
                response = self.handle_user_statement(user_input, self.conversation_history)

            print(f"AI: {response}")
            self.conversation_history += f" {user_input} {response}"

        print(f"\nThank you for the conversation, {self.user_name}!")
        print("Saving your session...")
        self.save_session()
        print("Session saved. You can continue from where you left off next time!")

# ... (QADataset and fine_tune_model functions remain the same)

def main():
    parser = argparse.ArgumentParser(description="Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5-small", help="Name of the pre-trained model to use")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune the model before starting the session")
    args = parser.parse_args()

    try:
        qa_cli = QuestionAnswerCLI(args.model)

        if args.fine_tune:
            print("Fine-tuning the model...")
            # Add your fine-tuning data here
            train_questions = ["What is the capital of France?", "Who wrote 'Romeo and Juliet'?"]
            train_contexts = ["France is a country in Western Europe.", "William Shakespeare was an English playwright."]
            train_answers = ["The capital of France is Paris.", "William Shakespeare wrote 'Romeo and Juliet'."]
            
            qa_cli.model = fine_tune_model(qa_cli.model, qa_cli.tokenizer, train_questions, train_contexts, train_answers)
            print("Fine-tuning complete.")

        qa_cli.interactive_session()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
