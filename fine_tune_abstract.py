import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from model_implementations import T5LanguageModel, BERTLanguageModel, GPT2LanguageModel, RoBERTaLanguageModel
from typing import List, Dict
import random

class QuestionAnsweringDataset(Dataset):
    def __init__(self, data: List[Dict[str, any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Randomly select one answer from the list of answers
        answer = random.choice(item['answers'])
        return item['context'], item['question'], answer

def load_data(file_path: str) -> List[Dict[str, any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def fine_tune_model(model, train_dataloader: DataLoader, num_epochs: int, learning_rate: float):
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for context, question, answer in train_dataloader:
            optimizer.zero_grad()
            
            input_text = f"{question}|{context}"
            loss = model.fine_tune(input_text, answer)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an abstract language model")
    parser.add_argument("--model_type", type=str, required=True, choices=['t5', 'bert', 'gpt2', 'roberta'], help="Type of model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    args = parser.parse_args()

    # Load data
    data = load_data(args.data_path)
    dataset = QuestionAnsweringDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model_map = {
        't5': T5LanguageModel,
        'bert': BERTLanguageModel,
        'gpt2': GPT2LanguageModel,
        'roberta': RoBERTaLanguageModel
    }
    model_class = model_map.get(args.model_type)
    if model_class is None:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model_class("fine_tuning")  # Use a generic user_id for fine-tuning

    # Fine-tune the model
    fine_tune_model(model, dataloader, args.num_epochs, args.learning_rate)

    # Save the fine-tuned model
    model.save(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
