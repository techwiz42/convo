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
import threading
import queue 
import time

'''
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)
''' 

class EnhancedMultiUserQuestionAnswerCLI:
    def __init__(self, model):
        self.model = model
        self.users = {}
        self.user_lock = threading.Lock()
        self.text_analyzer = TextAnalyzer()
        self.question_history = set()
        self.max_question_history = 100
        self.user_queues = {}
        self.stop_event = threading.Event()

    def get_or_create_user(self, user_id: str) -> Dict:
        with self.user_lock:
            if user_id not in self.users:
                self.users[user_id] = {
                    "knowledge_base": UserKnowledgeBase(user_id),
                    "conversation": Conversation(user_id)
                }
            return self.users[user_id]

    def process_user_input(self, user_id: str, user_input: str) -> str:
        user_data = self.get_or_create_user(user_id)
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        relevant_knowledge = knowledge_base.get_relevant_knowledge(user_input)
        context = " ".join(relevant_knowledge)

        input_text = f"Given the context: {context}\n\nPlease provide a response to the following: {user_input}\n\nResponse:"

        # Generate multiple responses with different parameters
        responses = []
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_new_tokens": 200}
        ]

        for params in generation_params:
            try:
                response = self.model.generate_response(input_text, **params)
                if response:
                    grammar_score, sentiment_score = self.text_analyzer.analyze_text(response)
                    total_score = abs(sentiment_score) + grammar_score
                    responses.append({"response": response, "score": total_score})
            except Exception as e:
                print(f"Error generating response: {str(e)}")

        if not responses:
            return "I'm sorry, I couldn't generate a response. Please try again with a different input."
        selected_response = max(responses, key=lambda x:x['score'])
        selected_response = selected_response.get("response")

        try:
            knowledge_base.add_knowledge(user_input, selected_response)
        except Exception as e:
            print(f"Error during knowledge base update: {str(e)}")

        # Update conversation in the database
        '''
        db = next(get_db())
        db_conversation = db.query(DBConversation).filter_by(user_id=user_id).first()
        if db_conversation:
            db_conversation.conversation_history += f" User: {user_input} AI: {selected_response}"
            db_conversation.last_question = user_input
        else:
            db_conversation = DBConversation(
                user_id=user_id,
                conversation_history=f"User: {user_input} AI: {selected_response}",
                last_question=user_input
            )
            db.add(db_conversation)
        db.commit()
        '''
        return selected_response

    def save_user_data(self, user_id: str):
        # No need to save model data, as it's shared
        pass

    def load_user_data(self, user_id: str) -> bool:
        user_data = self.get_or_create_user(user_id)
        
        # Load conversation from the database
        db = next(get_db())
        db_conversation = db.query(DBConversation).filter_by(user_id=user_id).first()
        if db_conversation:
            user_data["conversation"].conversation_history = db_conversation.conversation_history
            user_data["conversation"].last_question = db_conversation.last_question
            return True
        return False

    def run(self):
        print("Starting Enhanced Multi-User Question Answer CLI")
        print("Type 'exit' to quit")

        # Start the user input thread
        input_thread = threading.Thread(target=self._user_input_loop)
        input_thread.start()

        # Start the processing thread
        processing_thread = threading.Thread(target=self._process_user_inputs)
        processing_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_event.set()
            input_thread.join()
            processing_thread.join()
            print("CLI stopped.")

    def _user_input_loop(self):
        user_id = None
        while not self.stop_event.is_set():
            try:
                if user_id is None:
                    user_id = input("Enter your user ID: ")
                    if user_id.lower() == 'exit':
                        self.stop_event.set()
                        break
                
                user_input = input(f"[{user_id}] Enter your question: ")
                if user_input.lower() == 'exit':
                    self.stop_event.set()
                    break
                
                with self.user_lock:
                    if user_id not in self.user_queues:
                        self.user_queues[user_id] = queue.Queue()
                
                self.user_queues[user_id].put(user_input)
            except EOFError:
                self.stop_event.set()
                break

    def _process_user_inputs(self):
        while not self.stop_event.is_set():
            for user_id, user_queue in self.user_queues.items():
                try:
                    user_input = user_queue.get_nowait()
                    response = self.process_user_input(user_id, user_input)
                    print(f"[{user_id}] AI: {response}")
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error processing input for user {user_id}: {str(e)}")
                    traceback.print_exc()
            time.sleep(0.1)

# Example usage
if __name__ == "__main__":
    def model_factory(user_id):
        # Replace this with your actual model initialization
        return YourLanguageModelClass(user_id, "path/to/model")

    cli = EnhancedMultiUserQuestionAnswerCLI(model_factory)
    cli.run()
