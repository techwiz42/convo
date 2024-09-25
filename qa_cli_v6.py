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
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        self.sia = SentimentIntensityAnalyzer()
        self.previous_questions = set()
        self.user_name = ""
        self.sentiment_cache = {}
        
        print("Loading BERT model for question quality assessment...")
        self.quality_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(self.device)
        
        print("Loading BERT tokenizer...")
        self.quality_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print("All models loaded successfully!")

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
        if len(question.split()) < 5:  # Increased minimum length
            return False
        if not question.endswith('?'):
            return False
        if not re.match(r'^[A-Z]', question):  # Check if it starts with a capital letter
            return False
        if re.search(r"\b's\b", question):  # Check for standalone 's
            return False
        if re.search(r"\b[a-z]'s\b", question):  # Check for lowercase word followed by 's
            return False
        if "your thoughts on" in question.lower() and len(question.split()) < 8:  # Increased minimum length for this type
            return False
        if re.search(r'\b(a|an|the)\s+\w+\?', question):  # Check for questions ending with a single word after an article
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
            return f"That's interesting! {question}"
        elif sentiment < -0.05:
            return f"I understand this might be challenging. {question}"
        else:
            return question

    def generate_rule_based_question(self, context: str, sentiment: float) -> str:
        sentences = sent_tokenize(context)
        if not sentences:
            return "Can you provide more information about this topic?"

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

        # Filter words to use in questions
        valid_words = [word for word, tag in tagged if len(word) > 2 and tag.startswith(('NN', 'VB', 'JJ'))]
        
        if not valid_words:
            return "Can you elaborate more on this topic?"

        general_questions = [
            f"What are your thoughts on the concept of {random.choice(valid_words)}?",
            f"How does the idea of {random.choice(valid_words)} relate to {random.choice(valid_words)}?",
            f"Can you elaborate on the significance of {random.choice(valid_words)}?",
            f"What's the importance of {random.choice(valid_words)} in this context?",
            f"How does the concept of {random.choice(valid_words)} connect to your personal experiences?",
            f"What potential implications do you see from the idea of {random.choice(valid_words)}?",
            f"How might the concept of {random.choice(valid_words)} evolve in the future?",
            f"What questions does the idea of {random.choice(valid_words)} raise for you?"
        ]
        #pick a general question
        question = general_questions[random.randint(0, len(general_questions)-1)]
        if question not in self.previous_questions and self.is_valid_question(question):
            self.previous_questions.add(question)
            return self.adjust_question_for_sentiment(question, sentiment)
        #could not pick a good question
        return f"Can you share any additional insights on {random.choice(valid_words)}?"

    def generate_entity_question(self, entity: str, entity_type: str, sentiment: float) -> str:
        if sentiment > 0.05:
            questions = [
                f"What aspects of {entity} do you find most intriguing?",
                f"How has {entity} positively impacted this field?",
                f"What potential do you see for {entity} in the future?",
                f"What does {entity} mean to you?",
                f"How do you imagine that {entity} could be improved?",
                f"Tell me more about your experiences with {entity}."
            ]
        elif sentiment < -0.05:
            questions = [
                f"What challenges do you think {entity} faces?",
                f"How might the issues with {entity} be addressed?",
                f"What alternatives to {entity} might be worth considering?",
                f"Did you find {entity} to be difficult to manage?",
                f"Would you consider {entity} in the future?"
            ]
        else:
            if entity_type == 'PERSON':
                questions = [f"Who is {entity}?", 
                             f"What is {entity} known for?", 
                             f"How has {entity} influenced this field?"]
            elif entity_type in ['GPE', 'LOCATION']:
                questions = [f"Where is {entity}?", 
                             f"What's significant about {entity}?", 
                             f"How does {entity} relate to the topic?"]
            elif entity_type == 'ORGANIZATION':
                questions = [f"What is {entity}?",
                             f"What role does {entity} play in this context?",
                             f"How has {entity} evolved over time?"]
            else:
                questions = [f"What can you tell me about {entity}?",
                             f"How does {entity} fit into the broader picture?",
                             f"What's your perspective on {entity}?"]

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
