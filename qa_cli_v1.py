import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import os
import json
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import random

class QuestionAnswerCLI:
    """
    A class that encapsulates the functionality for loading a pre-trained question generation model
    and conducting interactive question-answer sessions.
    """

    def __init__(self, model_path: str):
        """
        Initialize the QuestionAnswerCLI with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model
        """
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def generate_question(self, context: str, answer: str = None) -> str:
        """
        Generate a question based on the given context and optional answer.

        Args:
            context (str): The context for generating the question
            answer (str, optional): The answer to incorporate into the question generation

        Returns:
            str: The generated question
        """
        if answer:
            input_text: str = f"generate question: {context} answer: {answer}"
        else:
            input_text: str = f"generate question: {context}"
        
        print(f"Input text: {input_text}")  # Logging the input text
        
        input_ids: torch.Tensor = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        
        print(f"Input shape: {input_ids.shape}")  # Logging the input shape
        
        # Generate multiple questions and filter
        num_return_sequences = 5
        outputs: torch.Tensor = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        generated_questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(f"Generated questions: {generated_questions}")  # Logging the generated questions
        
        # Filter out short or simple questions
        valid_questions = [q for q in generated_questions if len(q.split()) > 3 and q.lower() not in ["true", "false"]]
        
        if valid_questions:
            return random.choice(valid_questions)
        else:
            # Fallback to a template-based question if no valid questions were generated
            return self.generate_fallback_question(context)

    def generate_fallback_question(self, context: str) -> str:
        """
        Generate a fallback question when the model fails to produce a valid question.

        Args:
            context (str): The context for generating the question

        Returns:
            str: A fallback question
        """
        words = context.split()
        if len(words) > 5:
            subject = " ".join(words[:3])
            return f"What can you tell me about {subject}?"
        else:
            return "Can you provide more information about this topic?"

    def interactive_session(self) -> None:
        """
        Conduct an interactive question-answer session with the user.
        The session continues until the user types 'exit'.
        """
        context: str = input("Enter the initial context: ")
        print("\nStarting Q&A session. The AI will ask questions based on the context.")
        print("You can respond to each question or type 'exit' to end the session.\n")
        
        while True:
            question: str = self.generate_question(context)
            print(f"AI: {question}")
            
            user_input: str = input("Your response: ")
            if user_input.lower() == 'exit':
                break
            
            context += f" Question: {question} Answer: {user_input}"

        print("\nFinal context:")
        print(context)

# ... [rest of the script remains unchanged] ...

def main() -> None:
    """
    Main function to parse command-line arguments and either train the model or start an interactive session.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Question-Answer CLI Application")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, help="Path to training data JSON file")
    parser.add_argument("--model", type=str, default="./qa_model", help="Path to save/load the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], help="Device to use for training")
    args: argparse.Namespace = parser.parse_args()

    if args.train:
        if not args.data:
            print("Please provide a path to the training data using --data")
            return
        train_model(args.data, args.model, args.epochs, args.device)
    else:
        if not os.path.exists(args.model):
            print(f"Model not found at {args.model}. Please train the model first or provide a valid model path.")
            return
        cli: QuestionAnswerCLI = QuestionAnswerCLI(args.model)
        cli.interactive_session()

if __name__ == "__main__":
    main()
