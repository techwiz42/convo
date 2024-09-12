import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import os
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

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
        input_ids: torch.Tensor = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs: torch.Tensor = self.model.generate(input_ids, max_length=64, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def interactive_session(self) -> None:
        """
        Conduct an interactive question-answer session with the user.
        The session continues until the user types 'exit'.
        """
        context: str = input("Enter the initial context: ")
        print("\nStarting Q&A session. Type 'exit' to end the session.\n")
        
        while True:
            question: str = self.generate_question(context)
            print(f"AI: {question}")
            
            answer: str = input("You: ")
            if answer.lower() == 'exit':
                break
            
            context += f" {question} {answer}"

        print("\nFinal context:")
        print(context)

def train_model(data_path: str, output_path: str) -> None:
    """
    Train a question generation model using the provided dataset and save it to the specified path.

    Args:
        data_path (str): Path to the JSON file containing the training data
        output_path (str): Path where the trained model will be saved
    """
    dataset: Dataset = Dataset.from_json(data_path)

    # Initialize model and tokenizer
    model_name: str = "t5-small"
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Preprocess the dataset examples for training.

        Args:
            examples (Dict[str, List[Any]]): A batch of examples from the dataset

        Returns:
            Dict[str, List[Any]]: Preprocessed batch of examples
        """
        inputs: List[str] = ["generate question: " + context + " answer: " + answer 
                  for context, answer in zip(examples["context"], examples["answer"])]
        targets: List[str] = examples["question"]
        model_inputs: Dict[str, List[Any]] = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels: Dict[str, List[Any]] = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_dataset: Dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    train_dataloader: DataLoader = DataLoader(processed_dataset, shuffle=True, batch_size=8)

    # Training setup
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer: AdamW = AdamW(model.parameters(), lr=5e-5)
    num_epochs: int = 3
    num_training_steps: int = num_epochs * len(train_dataloader)
    lr_scheduler: get_linear_schedule_with_warmup = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items()}
            outputs: Any = model(**batch)
            loss: torch.Tensor = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Save the model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

def main() -> None:
    """
    Main function to parse command-line arguments and either train the model or start an interactive session.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Question-Answer CLI Application")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, help="Path to training data JSON file")
    parser.add_argument("--model", type=str, default="./qa_model", help="Path to save/load the model")
    args: argparse.Namespace = parser.parse_args()

    if args.train:
        if not args.data:
            print("Please provide a path to the training data using --data")
            return
        train_model(args.data, args.model)
    else:
        if not os.path.exists(args.model):
            print(f"Model not found at {args.model}. Please train the model first or provide a valid model path.")
            return
        cli: QuestionAnswerCLI = QuestionAnswerCLI(args.model)
        cli.interactive_session()

if __name__ == "__main__":
    main()
