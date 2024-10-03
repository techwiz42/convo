import threading
import random
import re
import json
import os
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from user_knowledge_base import UserKnowledgeBase
from model_implementations import Conversation
from abstract_language_model import AbstractLanguageModel

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class EnhancedMultiUserQuestionAnswerCLI:
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.users = {}
        self.user_lock = threading.Lock()
        self.sia = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        self.question_history = set()
        self.max_question_history = 100

    def run(self):
        print("Welcome to the Enhanced Multi-User Fine-Tuned Question-Answer System!")
        
        while True:
            user_id = input("Please enter your name (or 'exit' to quit): ").strip()
            
            if user_id.lower() == 'exit':
                break

            user_data = self.get_or_create_user(user_id)
            
            if self.load_user_data(user_id):
                print(f"\nWelcome back, {user_id}! I've loaded your personalized model and previous conversation.")
                print("\nHere's a summary of our last conversation:")
                print(user_data["conversation"].conversation_history[-500:])  # Print last 500 characters as a summary
            else:
                print(f"\nNice to meet you, {user_id}! I'm creating a personalized model for you.")
                user_data["conversation"].conversation_history = input("Please provide some context or a topic you'd like to discuss (or press Enter to start with a blank slate): ")

            print("\nYou can start asking questions or making statements. Type 'exit' to end the session.")

            while True:
                user_input = input(f"\n{user_id}: ").strip()
                
                if user_input.lower() == 'exit':
                    break

                response = self.process_user_input(user_id, user_input)
                print(f"AI: {response}")

            print(f"\nThank you for the conversation, {user_id}!")
            self.save_user_data(user_id)

        print("\nThank you for using the Enhanced Multi-User Fine-Tuned Question-Answer System!")

    def get_or_create_user(self, user_id: str) -> Dict:
        with self.user_lock:
            if user_id not in self.users:
                self.users[user_id] = {
                    "model": self.model_factory(user_id),
                    "knowledge_base": UserKnowledgeBase(user_id),
                    "conversation": Conversation(user_id)
                }
                self.users[user_id]["knowledge_base"].add_knowledge(
                    "initial", 
                    "Welcome to our conversation system. This knowledge base stores important information from our interactions."
                )
            return self.users[user_id]

    def generate_questions(self, context: str, sentiment: float) -> List[str]:
        entities = self.extract_entities(context)
        questions = []

        # Generate questions based on entities
        for entity in entities:
            questions.extend(self.generate_entity_questions(entity))

        # Generate questions based on the overall context
        questions.extend(self.generate_context_questions(context))

        # Filter out previously asked questions
        new_questions = [q for q in questions if q not in self.question_history]

        # Update question history
        self.question_history.update(new_questions[:5])  # Add up to 5 new questions to history
        if len(self.question_history) > self.max_question_history:
            self.question_history = set(list(self.question_history)[-self.max_question_history:])

        print("\nDEBUG - Generated questions:")
        for q in new_questions:
            print(f"  - {q}")

        return new_questions[:5] if new_questions else ["Can you provide more information or context?"]

    def extract_entities(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entities.append(' '.join(c[0] for c in chunk))
        return entities

    def generate_entity_questions(self, entity: str) -> List[str]:
        templates = [
            f"Can you elaborate on {entity}?",
            f"How does {entity} relate to our discussion?",
            f"What's your perspective on {entity}?",
            f"Could you provide more details about {entity}?",
            f"How would you describe the importance of {entity}?"
        ]
        return random.sample(templates, min(2, len(templates)))

    def generate_context_questions(self, context: str) -> List[str]:
        words = word_tokenize(context.lower())
        question_starters = [
            "How would you describe",
            "Can you explain",
            "What's your view on",
            "How does this relate to",
            "What are the implications of"
        ]
        questions = []
        for _ in range(3):
            if len(words) > 2:
                phrase = " ".join(random.sample(words, 2))
                starter = random.choice(question_starters)
                questions.append(f"{starter} {phrase}?")
        return questions

    def select_question(self, questions: List[str], sentiment: float, user_id: str) -> str:
        if not questions:
            return "Can you provide more information or context?"
        
        selected_question = random.choice(questions)
        if sentiment > 0.05:
            return f"That's interesting. {selected_question}"
        elif sentiment < -0.05:
            return f"I see. {selected_question}"
        else:
            return selected_question

    def analyze_sentiment(self, text: str) -> float:
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        sentiment = self.sia.polarity_scores(text)['compound']
        self.sentiment_cache[text] = sentiment
        return sentiment

    def determine_input_type(self, user_input: str, user_id: str) -> str:
        conversation = self.get_or_create_user(user_id)['conversation']
        if user_input.endswith('?'):
            return "question"
        
        if conversation.last_question:
            input_words = set(word_tokenize(user_input.lower()))
            question_words = set(word_tokenize(conversation.last_question.lower()))
            common_words = input_words.intersection(question_words)
            if len(common_words) >= 2:
                return "answer"
        
        return "statement"

    def handle_user_question(self, question: str, user_id: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        relevant_knowledge = knowledge_base.get_relevant_knowledge(question)
        context = " ".join(relevant_knowledge)

        print(f"\nDEBUG - Handling user question: {question}")
        print(f"DEBUG - Relevant context: {context}")

        input_text = f"Given the context: {context}\n\nPlease provide a detailed answer to the following question: {question}\n\nAnswer:"

        print(f"DEBUG - Input to model: {input_text}")

        # Generate multiple responses with different parameters
        responses = []
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_length": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_length": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_length": 200}
        ]

        for i, params in enumerate(generation_params):
            response = model.generate_response(input_text, **params)
            print(f"DEBUG - Generated response {i+1} (params: {params}): {response}")
            if not response.strip().endswith('?') and response.strip() != question.strip():
                responses.append(response)

        if responses:
            # Select the most diverse response
            answer = max(responses, key=lambda x: self.calculate_diversity_score(x, responses))
        else:
            answer = f"I apologize, but I'm having trouble generating a specific answer to your question: '{question}'. Could you please rephrase or provide more context?"

        print(f"DEBUG - Selected answer: {answer}")

        model.fine_tune(input_text, answer)
        knowledge_base.add_knowledge(question, answer)

        conversation.last_question = None
        return answer

    def calculate_diversity_score(self, response: str, all_responses: List[str]) -> float:
        """Calculate a diversity score for a response compared to all other responses."""
        words = set(response.lower().split())
        other_words = set(" ".join(all_responses).lower().split())
        unique_words = words - (words & other_words)
        return len(unique_words) / len(words) if words else 0

    def handle_user_answer(self, answer: str, user_id: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        sentiment = self.analyze_sentiment(answer)
        follow_up_questions = self.generate_questions(conversation.conversation_history + " " + answer, sentiment)
        selected_question = self.select_question(follow_up_questions, sentiment, user_id)

        input_text = f"context: {answer} question: {selected_question}"

        print(f"\nDEBUG - Handling user answer: {answer}")
        print(f"DEBUG - Input to model: {input_text}")

        # Generate multiple responses with different parameters
        responses = []
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_length": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_length": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_length": 200}
        ]

        for i, params in enumerate(generation_params):
            response = model.generate_response(input_text, **params)
            print(f"DEBUG - Generated response {i+1} (params: {params}): {response}")
            responses.append(response)

        # Select the most diverse response
        selected_response = max(responses, key=lambda x: self.calculate_diversity_score(x, responses))
        print(f"DEBUG - Selected response: {selected_response}")

        model.fine_tune(input_text, selected_response)
        knowledge_base.add_knowledge(answer, selected_response)

        conversation.last_question = selected_question
        return f"{selected_response} {selected_question}"

    def handle_user_statement(self, statement: str, user_id: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        sentiment = self.analyze_sentiment(statement)
        questions = self.generate_questions(conversation.conversation_history + " " + statement, sentiment)
        selected_question = self.select_question(questions, sentiment, user_id)

        input_text = f"context: {statement} question: {selected_question}"

        print(f"\nDEBUG - Handling user statement: {statement}")
        print(f"DEBUG - Input to model: {input_text}")

        # Generate multiple responses with different parameters
        responses = []
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_length": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_length": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_length": 200}
        ]

        for i, params in enumerate(generation_params):
            response = model.generate_response(input_text, **params)
            print(f"DEBUG - Generated response {i+1} (params: {params}): {response}")
            responses.append(response)

        # Select the most diverse response
        selected_response = max(responses, key=lambda x: self.calculate_diversity_score(x, responses))
        print(f"DEBUG - Selected response: {selected_response}")

        model.fine_tune(input_text, selected_response)
        knowledge_base.add_knowledge(statement, selected_response)

        conversation.last_question = selected_question
        return f"{selected_response} {selected_question}"

    def process_user_input(self, user_id: str, user_input: str) -> str:
        input_type = self.determine_input_type(user_input, user_id)

        print(f"\nDEBUG - Processing user input: {user_input}")
        print(f"DEBUG - Determined input type: {input_type}")

        if input_type == "question":
            response = self.handle_user_question(user_input, user_id)
        elif input_type == "answer":
            response = self.handle_user_answer(user_input, user_id)
        else:  # statement
            response = self.handle_user_statement(user_input, user_id)

        user_data = self.get_or_create_user(user_id)
        user_data["conversation"].conversation_history += f" User: {user_input} AI: {response}"
        
        print(f"DEBUG - Final response: {response}")
        return response

    def save_user_data(self, user_id: str):
        user_data = self.users[user_id]
        user_data["model"].save(f"{user_id}_model")
        user_data["knowledge_base"].save()
        with open(f"{user_id}_conversation.json", "w") as f:
            json.dump({
                "conversation_history": user_data["conversation"].conversation_history,
                "previous_questions": list(user_data["conversation"].previous_questions),
                "last_question": user_data["conversation"].last_question
            }, f)

    def load_user_data(self, user_id: str) -> bool:
        user_data = self.get_or_create_user(user_id)
        model_loaded = user_data["model"].load(f"{user_id}_model")
        kb_loaded = user_data["knowledge_base"].load()
        if os.path.exists(f"{user_id}_conversation.json"):
            with open(f"{user_id}_conversation.json", "r") as f:
                conversation_data = json.load(f)
                user_data["conversation"].conversation_history = conversation_data["conversation_history"]
                user_data["conversation"].previous_questions = set(conversation_data["previous_questions"])
                user_data["conversation"].last_question = conversation_data["last_question"]
        return model_loaded and kb_loaded
