# File: async_multi_user_qa.py

import asyncio
import logging
import os
import random
import json
import aiofiles
from typing import Dict

from text_analysis import TextAnalyzer
from user_knowledge_base import UserKnowledgeBase
from model_implementations import Conversation

logger = logging.getLogger(__name__)

class AsyncMultiUserQA:
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.users: Dict[str, Dict] = {}
        self.user_lock = asyncio.Lock()
        self.text_analyzer = TextAnalyzer()
        self.question_history = set()
        self.max_question_history = 100

    async def run(self):
        print("Welcome to the Async Multi-User Fine-Tuned Question-Answer System!")
        
        while True:
            user_id = await self._get_user_input("Please enter your name (or 'exit' to quit): ")
            
            if user_id.lower() == 'exit':
                break

            user_data = await self.get_or_create_user(user_id)
            
            if await self.load_user_data(user_id):
                print(f"\nWelcome back, {user_id}! I've loaded your personalized model and previous conversation.")
                print("\nHere's a summary of our last conversation:")
                print(user_data["conversation"].conversation_history[-500:])
            else:
                print(f"\nNice to meet you, {user_id}! I'm creating a personalized model for you.")
                user_data["conversation"].conversation_history = await self._get_user_input(
                    "Please provide some context or a topic you'd like to discuss (or press Enter to start with a blank slate): "
                )

            print("\nYou can start asking questions or making statements. Type 'exit' to end the session.")

            while True:
                user_input = await self._get_user_input(f"\n{user_id}: ")
                
                if user_input.lower() == 'exit':
                    break

                try:
                    response = await self.process_user_input(user_id, user_input)
                    print(f"AI: {response}")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    print("Please try again with a different input.")

            print(f"\nThank you for the conversation, {user_id}!")
            await self.save_user_data(user_id)

        print("\nThank you for using the Async Multi-User Fine-Tuned Question-Answer System!")

    async def _get_user_input(self, prompt: str) -> str:
        return await asyncio.get_event_loop().run_in_executor(None, input, prompt)

    async def get_or_create_user(self, user_id: str) -> Dict:
        async with self.user_lock:
            if user_id not in self.users:
                self.users[user_id] = {
                    "model": await self.model_factory.create_model_instance(user_id),
                    "knowledge_base": UserKnowledgeBase(user_id),
                    "conversation": Conversation(user_id)
                }
            return self.users[user_id]

    async def process_user_input(self, user_id: str, user_input: str) -> str:
        user_data = await self.get_or_create_user(user_id)
        model = user_data["model"]
        knowledge_base = user_data["knowledge_base"]
        conversation = user_data["conversation"]

        relevant_knowledge = await knowledge_base.get_relevant_knowledge(user_input)
        context = " ".join(relevant_knowledge)

        input_text = f"Given the context: {context}\n\nPlease provide a response to the following: {user_input}\n\nResponse:"

        responses = await self._generate_multiple_responses(model, input_text)

        if not responses:
            return "I'm sorry, I couldn't generate a response. Please try again with a different input."

        selected_response = random.choice(responses)

        try:
            await model.fine_tune(input_text, selected_response)
            await knowledge_base.add_knowledge(user_input, selected_response)
        except Exception as e:
            logger.error(f"Error during fine-tuning or knowledge base update: {str(e)}")

        conversation.conversation_history += f" User: {user_input} AI: {selected_response}"
    
        return selected_response

    async def _generate_multiple_responses(self, model, input_text):
        generation_params = [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100},
            {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 150},
            {"temperature": 0.5, "top_p": 0.8, "max_new_tokens": 200}
        ]

        async def generate_response(params):
            try:
                response = await model.generate_response(input_text, **params)
                if response:
                    grammar_score, sentiment_score = await self.text_analyzer.analyze_text(response)
                    return response, grammar_score, sentiment_score
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                logger.error(f"Params: {params}")
                logger.error(traceback.format_exc())
            return None

        tasks = [generate_response(params) for params in generation_params]
        results = await asyncio.gather(*tasks)

        return [result[0] for result in results if result]

    async def save_user_data(self, user_id: str):
        user_data = self.users[user_id]
        await user_data["model"].save(f"{user_id}_model")
        await user_data["knowledge_base"].save()
        async with aiofiles.open(f"{user_id}_conversation.json", "w") as f:
            await f.write(json.dumps({
                "conversation_history": user_data["conversation"].conversation_history,
                "previous_questions": list(user_data["conversation"].previous_questions),
                "last_question": user_data["conversation"].last_question
            }))

    async def load_user_data(self, user_id: str) -> bool:
        user_data = await self.get_or_create_user(user_id)
        model_loaded = await user_data["model"].load(f"{user_id}_model")
        kb_loaded = await user_data["knowledge_base"].load()
        if os.path.exists(f"{user_id}_conversation.json"):
            async with aiofiles.open(f"{user_id}_conversation.json", "r") as f:
                conversation_data = json.loads(await f.read())
                user_data["conversation"].conversation_history = conversation_data["conversation_history"]
                user_data["conversation"].previous_questions = set(conversation_data["previous_questions"])
                user_data["conversation"].last_question = conversation_data["last_question"]
        return model_loaded and kb_loaded
