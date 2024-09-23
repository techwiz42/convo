import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, BertTokenizer
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from functools import lru_cache
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class QuestionAnswerCLI:
    def __init__(self, model_name: str = "t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.sia = SentimentIntensityAnalyzer()
        self.previous_questions = set()
        self.user_name = ""
        self.sentiment_cache = {}
        
        # Load pre-trained model for question quality assessment
        self.quality_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(self.device)
        self.quality_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @lru_cache(maxsize=1000)
    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

    def generate_question(self, context: str, sentiment: float) -> str:
        input_text = f"generate question: {context}"
        input_ids = self.tokenize(input_text)

        outputs = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=10,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        filtered_questions = self.filter_questions(questions)
        
        if filtered_questions:
            selected_question = self.select_non_repetitive_question(filtered_questions, sentiment)
            if selected_question:
                return selected_question

        return self.generate_rule_based_question(context, sentiment)

    def filter_questions(self, questions):
        filtered = []
        for question in questions:
            if self.is_valid_question(question):
                quality_score = self.assess_question_quality(question)
                perplexity = self.calculate_perplexity(question)
                if quality_score > 0.7 and perplexity < 100:  # Adjust these thresholds as needed
                    filtered.append(question)
        return filtered

    def is_valid_question(self, question):
        # Enhanced heuristic rules
        if len(question.split()) < 3:
            return False
        if not question.endswith('?'):
            return False
        if not re.match(r'^[A-Z]', question):  # Check if it starts with a capital letter
            return False
        if re.search(r"\b's\b", question):  # Check for standalone 's
            return False
        if re.search(r"\b[a-z]'s\b", question):  # Check for lowercase word followed by 's
            return False
        if "your thoughts on" in question.lower() and len(question.split()) < 6:
            return False
        return True

    def assess_question_quality(self, question):
        inputs = self.quality_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.quality_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return scores[:, 1].item()  # Assuming binary classification (bad:0, good:1)

    def calculate_perplexity(self, question):
        input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()

    def select_non_repetitive_question(self, questions: list, sentiment: float) -> str:
        for question in questions:
            if question not in self.previous_questions:
                self.previous_questions.add(question)
                return self.adjust_question_for_sentiment(question, sentiment)
        return None

    def adjust_question_for_sentiment(self, question: str, sentiment: float) -> str:
        if sentiment > 0.05:
            return f"That's interesting, {self.user_name}! {question}"
        elif sentiment < -0.05:
            return f"I understand this might be challenging, {self.user_name}. {question}"
        else:
            return f"{self.user_name}, {question}"

    def generate_rule_based_question(self, context: str, sentiment: float) -> str:
        sentences = sent_tokenize(context)
        if not sentences:
            return f"{self.user_name}, can you provide more information about this topic?"

        sentence = random.choice(sentences)
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        chunked = ne_chunk(tagged)

        for subtree in chunked:
            if type(subtree) == nltk.Tree:
                entity_type = subtree.label()
                entity = " ".join([word for word, tag in subtree.leaves()])
                question = self.generate_entity_question(entity, entity_type, sentiment)
                if question not in self.previous_questions and self.is_valid_question(question):
                    self.previous_questions.add(question)
                    return question

        general_questions = [
            f"{self.user_name}, what are your thoughts on the topic of {random.choice(words)}?",
            f"How does the concept of {random.choice(words)} relate to the overall topic, {self.user_name}?",
            f"{self.user_name}, can you elaborate on the idea of {random.choice(words)}?",
            f"What's the significance of {random.choice(words)} in this context, {self.user_name}?",
            f"{self.user_name}, how does this information connect to your personal experiences?",
            f"What potential implications do you see from this information, {self.user_name}?",
            f"{self.user_name}, how might this topic evolve in the future?",
            f"What questions does this raise for you, {self.user_name}?"
        ]

        for question in general_questions:
            if question not in self.previous_questions and self.is_valid_question(question):
                self.previous_questions.add(question)
                return self.adjust_question_for_sentiment(question, sentiment)

        return f"{self.user_name}, can you share any additional insights on this topic?"

    def generate_entity_question(self, entity: str, entity_type: str, sentiment: float) -> str:
        if sentiment > 0.05:
            questions = [
                f"{self.user_name}, what aspects of {entity} do you find most intriguing?",
                f"How has {entity} positively impacted this field, {self.user_name}?",
                f"{self.user_name}, what potential do you see for {entity} in the future?"
            ]
        elif sentiment < -0.05:
            questions = [
                f"{self.user_name}, what challenges do you think {entity} faces?",
                f"How might the issues with {entity} be addressed, {self.user_name}?",
                f"{self.user_name}, what alternatives to {entity} might be worth considering?"
            ]
        else:
            if entity_type == 'PERSON':
                questions = [f"Who is {entity}, {self.user_name}?", f"{self.user_name}, what is {entity} known for?", f"How has {entity} influenced this field, {self.user_name}?"]
            elif entity_type in ['GPE', 'LOCATION']:
                questions = [f"Where is {entity}, {self.user_name}?", f"{self.user_name}, what's significant about {entity}?", f"How does {entity} relate to the topic, {self.user_name}?"]
            elif entity_type == 'ORGANIZATION':
                questions = [f"What is {entity}, {self.user_name}?", f"{self.user_name}, what role does {entity} play in this context?", f"How has {entity} evolved over time, {self.user_name}?"]
            else:
                questions = [f"{self.user_name}, what can you tell me about {entity}?", f"How does {entity} fit into the broader picture, {self.user_name}?", f"{self.user_name}, what's your perspective on {entity}?"]

        return random.choice(questions)

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

        answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        sentiments = [abs(self.analyze_sentiment(answer)) for answer in answers]
        
        return answers[sentiments.index(max(sentiments))]

    def interactive_session(self) -> None:
        self.user_name = input("Please enter your name: ")
        print(f"\nNice to meet you, {self.user_name}!")
        
        context = input("Enter the initial context (or press Enter to start with a blank context): ")
        if not context:
            print("\nLet's start our Q&A session.")
            print("The AI will ask questions, and you can respond to each question or type 'exit' to end the session.\n")
        else:
            print("\nStarting Q&A session. The AI will ask questions based on the context you provided.")
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

            context += f" Question: {question} Answer: {user_input} AI's Response: {ai_answer}"

        print(f"\nThank you for the conversation, {self.user_name}!")
        print("\nFinal context:")
        print(context)

def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5-base", help="Name of the pre-trained model to use")
    args = parser.parse_args()

    cli = QuestionAnswerCLI(args.model)
    cli.interactive_session()

if __name__ == "__main__":
    main()
