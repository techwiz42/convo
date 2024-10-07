import threading
import random
from typing import Dict
import nltk
from nltk.tokenize import word_tokenize
import json
import os
from user_knowledge_base import UserKnowledgeBase
from model_implementations import Conversation
from abstract_language_model import AbstractLanguageModel
from text_analysis import TextAnalyzer
import traceback

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
        self.text_analyzer = TextAnalyzer()
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

                try:
                    response = self.process_user_input(user_id, user_input)
                    print(f"AI: {response}")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    print("Please try again with a different input.")

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
            return self.users[user_id]

    def process_user_input(self, user_id: str, user_input: str) -> str:
        user_data = self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        relevant_knowledge = knowledge_base.get_relevant_knowledge(user_input)
        context = " ".join(relevant_knowledge)

        print(f"\nDEBUG - Processing user input: {user_input}")

        input_text = f"Given the context: {context}\n\nPlease provide a response to the following: {user_input}\n\nResponse:"

        print(f"DEBUG - Input to model: {input_text}")
        print(f"DEBUG - Input type: {type(input_text)}")

        # Generate multiple responses with different parameters
        responses = []
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_new_tokens": 200}
        ]

        for i, params in enumerate(generation_params):
            try:
                response = model.generate_response(input_text, **params)
                if response:
                    print(f"DEBUG - Generated response {i+1} (params: {params}):")
                    print(f"  Response: {response}")
                    grammar_score, sentiment_score = self.text_analyzer.analyze_text(response)
                    print(f"  Grammar Score: {grammar_score}")
                    print(f"  Sentiment Score: {sentiment_score}")
                    responses.append(response)
                else:
                    print(f"DEBUG - No response generated for parameters {params}")
            except Exception as e:
                print(f"Error generating response {i+1}: {str(e)}")
                print(f"Params: {params}")
                print(traceback.format_exc())
        if not responses:
            return "I'm sorry, I couldn't generate a response. Please try again with a different input."

        # Choose one response at random
        selected_response = random.choice(responses)

        print(f"DEBUG - Selected response: {selected_response}")

        try:
            model.fine_tune(input_text, selected_response)
            knowledge_base.add_knowledge(user_input, selected_response)
        except Exception as e:
            print(f"Error during fine-tuning or knowledge base update: {str(e)}")

        conversation.conversation_history += f" User: {user_input} AI: {selected_response}"
    
        return selected_response

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

# Example usage
if __name__ == "__main__":
    def model_factory(user_id):
        # Replace this with your actual model initialization
        return YourLanguageModelClass(user_id, "path/to/model")

    cli = EnhancedMultiUserQuestionAnswerCLI(model_factory)
    cli.run()
