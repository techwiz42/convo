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
        
        #print(f"Input text: {input_text}")  # Logging the input text
        
        input_ids: torch.Tensor = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        
        print(f"Input shape: {input_ids.shape}")  # Logging the input shape
        
        outputs: torch.Tensor = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        generated_question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print(f"Generated question: {generated_question}")  # Logging the generated question
        
        return generated_question

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
            print(f"AI: {question}?")
            
            user_input: str = input("Your response: ")
            if user_input.lower() == 'exit':
                break
            
            context += f" Question: {question} Answer: {user_input}?"

        print("\nFinal context:")
        print(context)

def train_model(data_path: str, output_path: str, num_epochs: int, device: str = None) -> None:
    """
    Train a question generation model using the provided dataset and save it to the specified path.
    If a model already exists at the output_path, it will be loaded and training will continue from there.

    Args:
        data_path (str): Path to the JSON file containing the training data
        output_path (str): Path where the trained model will be saved
        num_epochs (int): Number of training epochs
        device (str, optional): Device to use for training ('cuda', 'cpu', or None for auto-detect)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    print(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    data_dict = {
        'context': [item['context'] for item in raw_data['data']],
        'question': [item['question'] for item in raw_data['data']],
        'answer': [item['answer'] for item in raw_data['data']]
    }
    
    print("Creating dataset")
    dataset = Dataset.from_dict(data_dict)
    print(f"Dataset created with {len(dataset)} examples")

    # Initialize model and tokenizer
    if os.path.exists(output_path):
        print(f"Loading existing model from {output_path}")
        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(output_path)
        model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(output_path)
    else:
        print("Initializing new model")
        model_name: str = "t5-small"
        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name)
        model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)

    model.to(device)

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
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    print("Preprocessing dataset")
    processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print(f"Processed dataset created with {len(processed_dataset)} examples")

    train_dataloader: DataLoader = DataLoader(processed_dataset, shuffle=True, batch_size=8)

    optimizer: AdamW = AdamW(model.parameters(), lr=5e-5)
    num_training_steps: int = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in train_dataloader:
            try:
                # Process and move batch to device
                processed_batch = {k: v.to(device) for k, v in batch.items()}
                
                with autocast():
                    outputs = model(**processed_batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                total_loss += loss.item()
                print(f"Batch processed successfully. Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch structure: {batch.keys()}")
                for k, v in batch.items():
                    print(f"{k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
                continue  # Skip this batch and continue with the next one
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    # Save the model
    print(f"Saving model to {output_path}")
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
