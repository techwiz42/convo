import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

class UserKnowledgeBase:
    def __init__(self, user_id):
        self.user_id = user_id
        self.knowledge = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def add_knowledge(self, topic, content):
        self.knowledge[topic] = content
        self._update_tfidf()

    def _update_tfidf(self):
        documents = list(self.knowledge.values())
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

    def get_relevant_knowledge(self, query, top_n=3):
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        return [list(self.knowledge.values())[i] for i in related_docs_indices]

    def save(self):
        with open(f"{self.user_id}_knowledge.json", "w") as f:
            json.dump(self.knowledge, f)

    def load(self):
        if os.path.exists(f"{self.user_id}_knowledge.json"):
            with open(f"{self.user_id}_knowledge.json", "r") as f:
                self.knowledge = json.load(f)
            self._update_tfidf()
            return True
        return False

class Conversation:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = ""
        self.previous_questions = set()
        self.last_question = None

class UserSpecificModel:
    def __init__(self, user_id, base_model_name):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)

    def fine_tune(self, input_text, target_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        loss = self.model(input_ids=input_ids, labels=target_ids).loss
        loss.backward()
        optimizer.step()

    def generate_response(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        output = self.model.generate(input_ids, max_length=128, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def save(self):
        self.model.save_pretrained(f"{self.user_id}_model")
        self.tokenizer.save_pretrained(f"{self.user_id}_tokenizer")

    def load(self):
        if os.path.exists(f"{self.user_id}_model"):
            self.model = T5ForConditionalGeneration.from_pretrained(f"{self.user_id}_model").to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(f"{self.user_id}_tokenizer")
            return True
        return False

class EnhancedMultiUserQuestionAnswerCLI:
    def __init__(self, base_model_name: str = "t5-small"):
        self.base_model_name = base_model_name
        self.users = {}
        self.user_lock = threading.Lock()
        self.sia = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}

    @lru_cache(maxsize=1000)
    def tokenize(self, text: str):
        user_model = self.get_or_create_user(text.split(':')[0].strip())['model']
        return user_model.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(user_model.device)

    def get_or_create_user(self, user_id: str) -> Dict:
        with self.user_lock:
            if user_id not in self.users:
                self.users[user_id] = {
                    "model": UserSpecificModel(user_id, self.base_model_name),
                    "knowledge_base": UserKnowledgeBase(user_id),
                    "conversation": Conversation(user_id)
                }
            return self.users[user_id]

    def generate_questions(self, context, sentiment):
        prompts = [
            f"generate questions: {context}",
            f"ask about: {context}",
            f"what would you like to know about: {context}",
            f"create a list of questions based on: {context}",
            f"inquire about: {context}"
        ]
        
        all_questions = []
        user_model = self.get_or_create_user(context.split(':')[0].strip())['model']
        for prompt in prompts:
            input_ids = self.tokenize(prompt)
            
            outputs = user_model.model.generate(
                input_ids,
                max_length=128,
                num_return_sequences=3,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
            
            questions = [user_model.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            questions = [f"{q}?" if not q.endswith('?') else q for q in questions]
            all_questions.extend(self.filter_questions(questions))
        
        if not all_questions:
            entities = self.extract_entities(context)
            for entity in entities:
                all_questions.extend(self.generate_entity_questions(entity))
        
        all_questions = list(set(all_questions))[:5]
        
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
        return list(set(entities))

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

    def select_question(self, questions: list, sentiment: float, user_id: str) -> str:
        conversation = self.get_or_create_user(user_id)['conversation']
        for question in questions:
            if question not in conversation.previous_questions:
                conversation.previous_questions.add(question)
                return self.adjust_question_for_sentiment(question, sentiment)
        conversation.previous_questions.clear()
        return self.select_question(questions, sentiment, user_id)

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

    def save_user_data(self, user_id):
        user_data = self.users[user_id]
        user_data["model"].save()
        user_data["knowledge_base"].save()
        with open(f"{user_id}_conversation.json", "w") as f:
            json.dump({
                "conversation_history": user_data["conversation"].conversation_history,
                "previous_questions": list(user_data["conversation"].previous_questions),
                "last_question": user_data["conversation"].last_question
            }, f)

    def load_user_data(self, user_id):
        user_data = self.get_or_create_user(user_id)
        model_loaded = user_data["model"].load()
        kb_loaded = user_data["knowledge_base"].load()
        if os.path.exists(f"{user_id}_conversation.json"):
            with open(f"{user_id}_conversation.json", "r") as f:
                conversation_data = json.load(f)
                user_data["conversation"].conversation_history = conversation_data["conversation_history"]
                user_data["conversation"].previous_questions = set(conversation_data["previous_questions"])
                user_data["conversation"].last_question = conversation_data["last_question"]
        return model_loaded and kb_loaded

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
        return response

    def run_concurrent_sessions(self, user_inputs: Dict[str, List[str]]):
        with ThreadPoolExecutor() as executor:
            futures = []
            for user_id, inputs in user_inputs.items():
                for user_input in inputs:
                    futures.append(executor.submit(self.process_user_input, user_id, user_input))

            for future in as_completed(futures):
                response = future.result()
                print(f"AI: {response}")

        print("\nAll conversations completed.")
        print("Saving sessions...")
        for user_id in user_inputs.keys():
            self.save_user_data(user_id)
        print("Sessions saved.")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-User Fine-Tuned Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5-small", help="Name of the pre-trained model to use")
    args = parser.parse_args()

    try:
        qa_cli = EnhancedMultiUserQuestionAnswerCLI(args.model)

        print("Welcome to the Enhanced Multi-User Fine-Tuned Question-Answer System!")
        user_id = input("Please enter your name: ").strip()

        if qa_cli.load_user_data(user_id):
            print(f"\nWelcome back, {user_id}! I've loaded your personalized model and previous conversation.")
            user_data = qa_cli.get_or_create_user(user_id)
            print("\nHere's a summary of our last conversation:")
            print(user_data["conversation"].conversation_history[-500:])  # Print last 500 characters as a summary
        else:
            print(f"\nNice to meet you, {user_id}! I'm creating a personalized model for you.")
            user_data = qa_cli.get_or_create_user(user_id)
            user_data["conversation"].conversation_history = input("Please provide some context or a topic you'd like to discuss (or press Enter to start with a blank slate): ")
            if not user_data["conversation"].conversation_history:
                print("\nLet's start our conversation.")
            else:
                print("\nGreat! Let's discuss based on the context you provided.")

        print("\nYou can start asking questions or making statements. Type 'exit' to end the conversation.")

        while True:
            user_input = input(f"\n{user_id}: ").strip()
            if user_input.lower() == 'exit':
                break

            response = qa_cli.process_user_input(user_id, user_input)
            print(f"AI: {response}")

        print(f"\nThank you for the conversation, {user_id}!")
        print("Saving your personalized model and conversation data...")
        qa_cli.save_user_data(user_id)
        print("Data saved. Your AI will remember this conversation and continue to learn in future sessions!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()

