import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher

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
        self.previous_questions = []

    def generate_question(self, context: str, sentiment: float) -> str:
        input_text = f"generate question: {context}"
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=10,  # Increased from 5 to 10 for more variety
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        #print(f"Model generated questions: {questions}")

        valid_questions = [q for q in questions if len(q.split()) > 3 and q.lower() not in ["true", "false"]]
        
        if valid_questions:
            selected_question = self.select_non_repetitive_question(valid_questions, sentiment)
            if selected_question:
                return selected_question

        # If no valid non-repetitive questions, generate a rule-based question
        return self.generate_rule_based_question(context, sentiment)

    def select_non_repetitive_question(self, questions: list, sentiment: float) -> str:
        for question in questions:
            if not self.is_similar_to_previous(question):
                self.previous_questions.append(question)
                return self.adjust_question_for_sentiment(question, sentiment)
        return None

    def is_similar_to_previous(self, question: str, similarity_threshold: float = 0.7) -> bool:
        return any(SequenceMatcher(None, question.lower(), prev_q.lower()).ratio() > similarity_threshold 
                   for prev_q in self.previous_questions)

    def adjust_question_for_sentiment(self, question: str, sentiment: float) -> str:
        if sentiment > 0.05:  # Positive sentiment
            return f"That's interesting! {question}"
        elif sentiment < -0.05:  # Negative sentiment
            return f"I understand this might be challenging. {question}"
        else:  # Neutral sentiment
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
                if not self.is_similar_to_previous(question):
                    self.previous_questions.append(question)
                    return question

        # If no suitable entity question, try general questions
        general_questions = [
            f"What are your thoughts on {random.choice(words)}?",
            f"How does {random.choice(words)} relate to the overall topic?",
            f"Can you elaborate on the concept of {random.choice(words)}?",
            f"What's the significance of {random.choice(words)} in this context?",
            "How does this information connect to your personal experiences?",
            "What potential implications do you see from this information?",
            "How might this topic evolve in the future?",
            "What questions does this raise for you?"
        ]

        for question in general_questions:
            if not self.is_similar_to_previous(question):
                self.previous_questions.append(question)
                return self.adjust_question_for_sentiment(question, sentiment)

        # If all else fails, use a generic question
        return "Can you share any additional insights on this topic?"

    def generate_entity_question(self, entity: str, entity_type: str, sentiment: float) -> str:
        if sentiment > 0.05:
            questions = [
                f"What aspects of {entity} do you find most intriguing?",
                f"How has {entity} positively impacted this field?",
                f"What potential do you see for {entity} in the future?"
            ]
        elif sentiment < -0.05:
            questions = [
                f"What challenges do you think {entity} faces?",
                f"How might the issues with {entity} be addressed?",
                f"What alternatives to {entity} might be worth considering?"
            ]
        else:
            if entity_type == 'PERSON':
                questions = [f"Who is {entity}?", f"What is {entity} known for?", f"How has {entity} influenced this field?"]
            elif entity_type in ['GPE', 'LOCATION']:
                questions = [f"Where is {entity}?", f"What's significant about {entity}?", f"How does {entity} relate to the topic?"]
            elif entity_type == 'ORGANIZATION':
                questions = [f"What is {entity}?", f"What role does {entity} play in this context?", f"How has {entity} evolved over time?"]
            else:
                questions = [f"What can you tell me about {entity}?", f"How does {entity} fit into the broader picture?", f"What's your perspective on {entity}?"]

        return random.choice(questions)

    def analyze_sentiment(self, text: str) -> float:
        return self.sia.polarity_scores(text)['compound']

    def interactive_session(self) -> None:
        context = input("Enter the initial context: ")
        print("\nStarting Q&A session. The AI will ask questions based on the context.")
        print("You can respond to each question or type 'exit' to end the session.\n")

        sentiment = 0  # Start with neutral sentiment

        while True:
            question = self.generate_question(context, sentiment)
            print(f"AI: {question}")

            user_input = input("Your response: ")
            if user_input.lower() == 'exit':
                break

            sentiment = self.analyze_sentiment(user_input)
            print(f"Detected sentiment: {sentiment:.2f}")

            context += f" {question} {user_input}"

        print("\nFinal context:")
        print(context)

def main() -> None:
    parser = argparse.ArgumentParser(description="Question-Answer CLI Application with Sentiment Analysis and No Repetition")
    parser.add_argument("--model", type=str, default="t5-base", help="Name of the pre-trained model to use")
    args = parser.parse_args()

    cli = QuestionAnswerCLI(args.model)
    cli.interactive_session()

if __name__ == "__main__":
    main()
