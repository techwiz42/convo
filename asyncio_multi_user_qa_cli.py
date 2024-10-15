import asyncio
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

class AsyncEnhancedMultiUserQuestionAnswerCLI:
    def __init__(self, model):
        self.model = model
        self.users = {}
        self.text_analyzer = TextAnalyzer()
        self.question_history = set()
        self.max_question_history = 100
        self.stopped = False

    async def process_user_input(self, user_id: str, user_input: str) -> str:
        user_data = await self.get_or_create_user(user_id)  # Correctly await this coroutine
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        # Assuming get_relevant_knowledge is synchronous
        relevant_knowledge = await asyncio.to_thread(knowledge_base.get_relevant_knowledge, user_input)
        context = " ".join(relevant_knowledge)

        input_text = f"Given the context: {context}\n\nPlease provide a response to the following: {user_input}\n\nResponse:"

        # Generate multiple responses with different parameters
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_new_tokens": 200}
        ]

        async def generate_response(params):
            try:
                # Run generate_response in a separate thread
                response = await asyncio.to_thread(self.model.generate_response, input_text, **params)
                if response:
                    # Assuming analyze_text is synchronous
                    grammar_score, sentiment_score = await asyncio.to_thread(self.text_analyzer.analyze_text, response)
                    total_score = abs(sentiment_score) + grammar_score
                    return {"response": response, "score": total_score}
            except Exception as e:
                print(f"Error generating response: {str(e)}")
            return None

        tasks = [generate_response(params) for params in generation_params]
        responses = await asyncio.gather(*tasks)
        responses = [r for r in responses if r is not None]

        if not responses:
            return "I'm sorry, I couldn't generate a response. Please try again with a different input."
        
        selected_response = max(responses, key=lambda x: x['score'])
        selected_response = selected_response.get("response")

        try:
            # Assuming add_knowledge is synchronous
            await asyncio.to_thread(knowledge_base.add_knowledge, user_input, selected_response)
        except Exception as e:
            print(f"Error during knowledge base update: {str(e)}")

        return selected_response

    async def get_or_create_user(self, user_id: str):
        # Implement the async logic for getting or creating a user
        # This is just a placeholder implementation
        if user_id not in self.users:
            self.users[user_id] = {
                "knowledge_base": UserKnowledgeBase(user_id),
                "conversation": Conversation(user_id)
            }
        return self.users[user_id]

    def stop(self):
        self.stopped = True

    async def run(self):
        print("Starting Async Enhanced Multi-User Question Answer CLI")
        print("Type 'exit' to quit")

        while not self.stopped:
            user_id = await asyncio.to_thread(input, "Enter your user ID: ")
            if user_id.lower() == 'exit' or self.stopped:
                break

            while not self.stopped:
                user_input = await asyncio.to_thread(input, f"[{user_id}] Enter your question: ")
                if user_input.lower() == 'exit' or self.stopped:
                    break

                response = await self.process_user_input(user_id, user_input)
                print(f"[{user_id}] AI: {response}")

        print("CLI stopped.")

# Example usage
if __name__ == "__main__":
    async def model_factory(user_id):
        # Replace this with your actual model initialization
        return YourAsyncLanguageModelClass(user_id, "path/to/model")

    async def main():
        cli = AsyncEnhancedMultiUserQuestionAnswerCLI(await model_factory("default"))
        await cli.run()

    asyncio.run(main())
