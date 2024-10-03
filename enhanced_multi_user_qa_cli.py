import threading
import random
import re
import json
import os
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from user_knowledge_base import UserKnowledgeBase
from conversation import Conversation
from abstract_language_model import AbstractLanguageModel

class EnhancedMultiUserQuestionAnswerCLI:
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.users = {}
        self.user_lock = threading.Lock()
        self.sia = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}

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

    @lru_cache(maxsize=1000)
    def tokenize(self, text: str):
        user_model = self.get_or_create_user(text.split(':')[0].strip())['model']
        return user_model.get_tokenizer()(text, return_tensors="pt", max_length=512, truncation=True).input_ids

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
        prompts = [
            f"generate diverse questions: {context}",
            f"ask about different aspects of: {context}",
            f"what would you like to know more about: {context}",
            f"create a list of varied questions based on: {context}",
            f"inquire about various aspects of: {context}"
        ]
        
        all_questions = []
        user_model = self.get_or_create_user(context.split(':')[0].strip())['model']
        for prompt in prompts:
            output = user_model.generate_response(prompt)
            questions = output.split('\n')  # Assume the model generates a list of questions
            questions = [f"{q}?" if not q.endswith('?') else q for q in questions]
            all_questions.extend(self.filter_questions(questions))
        
        if not all_questions:
            entities = self.extract_entities(context)
            for entity in entities:
                all_questions.extend(self.generate_entity_questions(entity))
        
        all_questions = list(set(all_questions))  # Remove duplicates
        random.shuffle(all_questions)  # Shuffle for variety
        
        if not all_questions:
            all_questions = self.generate_improved_fallback_questions(context)
        
        return all_questions[:5] if all_questions else []  # Return up to 5 questions or an empty list

    def filter_questions(self, questions: List[str]) -> List[str]:
        return [q for q in questions if self.is_valid_question(q)]

    def is_valid_question(self, question: str) -> bool:
        if len(question.split()) < 4:
            return False
        if not question.endswith('?'):
            return False
        if not re.match(r'^[A-Z]', question):
            return False
        return True

    def extract_entities(self, text: str) -> List[str]:
        words = word_tokenize(text)
        tagged = pos_tag(words)
        entities = [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('JJ')]
        return list(set(entities))

    def generate_entity_questions(self, entity: str) -> List[str]:
        templates = [
            f"What is the significance of {entity}?",
            f"How does {entity} relate to the topic?",
            f"Can you elaborate on {entity}?",
            f"What are the key aspects of {entity}?",
            f"How would you describe {entity} to someone unfamiliar with it?"
        ]
        return random.sample(templates, min(3, len(templates)))

    def generate_improved_fallback_questions(self, context: str) -> List[str]:
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

    def select_question(self, questions: List[str], sentiment: float, user_id: str) -> str:
        conversation = self.get_or_create_user(user_id)['conversation']
        
        # Filter out questions that have been asked recently
        new_questions = [q for q in questions if q not in conversation.previous_questions]
        
        if new_questions:
            selected_question = random.choice(new_questions)
            self.add_to_previous_questions(conversation, selected_question)
            return self.adjust_question_for_sentiment(selected_question, sentiment)
        
        # If all questions have been asked, reset the previous questions and try again
        self.reset_previous_questions(conversation)
        return self.select_question(questions, sentiment, user_id)

    def add_to_previous_questions(self, conversation, question):
        conversation.previous_questions.add(question)
        if len(conversation.previous_questions) > 20:  # Keep only the last 20 questions
            conversation.previous_questions.pop()

    def reset_previous_questions(self, conversation):
        conversation.previous_questions.clear()

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

        input_text = f"context: {context} question: {question}"
        answer = model.generate_response(input_text)

        model.fine_tune(input_text, answer)
        knowledge_base.add_knowledge(question, answer)

        conversation.last_question = None
        return f"To answer your question: {answer}"

    def handle_user_answer(self, answer: str, user_id: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        sentiment = self.analyze_sentiment(answer)
        follow_up_questions = self.generate_questions(conversation.conversation_history + " " + answer, sentiment)
        selected_question = self.select_question(follow_up_questions, sentiment, user_id)

        input_text = f"context: {answer} question: {selected_question}"
        response = model.generate_response(input_text)

        model.fine_tune(input_text, response)
        knowledge_base.add_knowledge(answer, response)

        conversation.last_question = selected_question
        return f"Thank you for your answer. {selected_question}"

    def handle_user_statement(self, statement: str, user_id: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        sentiment = self.analyze_sentiment(statement)
        questions = self.generate_questions(conversation.conversation_history + " " + statement, sentiment)
        selected_question = self.select_question(questions, sentiment, user_id)

        input_text = f"context: {statement} question: {selected_question}"
        response = model.generate_response(input_text)

        model.fine_tune(input_text, response)
        knowledge_base.add_knowledge(statement, response)

        conversation.last_question = selected_question
        return f"Interesting point. {selected_question}"

    def process_user_input(self, user_id: str, user_input: str) -> str:
        input_type = self.determine_input_type(user_input, user_id)

        if input_type == "question":
            response = self.handle_user_question(user_input, user_id)
        elif input_type == "answer":
            response = self.handle_user_answer(user_input, user_id)
        else:  # statement
            response = self.handle_user_statement(user_input, user_id)

        user_data = self.get_or_create_user(user_id)
        user_data["conversation"].conversation_history += f" {user_input} {response}"
        
        # If no question was generated, add a generic prompt
        if not response.endswith('?'):
            response += " Is there anything else you'd like to discuss?"
        
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
