import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

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

    def generate_question(self, context: str, sentiment: float) -> str:
        input_text = f"generate question: {context}"
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(f"Model generated questions: {questions}")

        valid_questions = [q for q in questions if len(q.split()) > 3 and q.lower() not in ["true", "false"]]

        if valid_questions:
            return self.select_question_based_on_sentiment(valid_questions, sentiment)
        else:
            return self.generate_rule_based_question(context, sentiment)

    def select_question_based_on_sentiment(self, questions: list, sentiment: float) -> str:
        if sentiment > 0.05:  # Positive sentiment
            positive_questions = [q for q in questions if "why" in q.lower() or "how" in q.lower()]
            return random.choice(positive_questions) if positive_questions else random.choice(questions)
        elif sentiment < -0.05:  # Negative sentiment
            negative_questions = [q for q in questions if "what" in q.lower() or "can you explain" in q.lower()]
            return random.choice(negative_questions) if negative_questions else random.choice(questions)
        else:  # Neutral sentiment
            return random.choice(questions)

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
                if sentiment > 0.05:
                    return f"Why do you think {entity} is important?"
                elif sentiment < -0.05:
                    return f"What concerns you about {entity}?"
                else:
                    if entity_type == 'PERSON':
                        return f"Who is {entity}?"
                    elif entity_type in ['GPE', 'LOCATION']:
                        return f"Where is {entity}?"
                    elif entity_type == 'ORGANIZATION':
                        return f"What is {entity}?"

        # If no named entity is found, try to generate a question based on POS tags
        for word, tag in tagged:
            if tag.startswith('NN'):  # Noun
                return f"What can you tell me about {word}?"
            elif tag.startswith('VB'):  # Verb
                return f"What does it mean to {word}?"

        # If all else fails, use a generic question
        return f"What is the main idea of this sentence: '{sentence}'?"

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
    parser = argparse.ArgumentParser(description="Question-Answer CLI Application with Sentiment Analysis")
    parser.add_argument("--model", type=str, default="t5-base", help="Name of the pre-trained model to use")
    args = parser.parse_args()

    cli = QuestionAnswerCLI(args.model)
    cli.interactive_session()

if __name__ == "__main__":
    main()
