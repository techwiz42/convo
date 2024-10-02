import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from model_implementations import T5LanguageModel, BERTLanguageModel, GPT2LanguageModel, RoBERTaLanguageModel
from typing import List, Dict
import random
from tqdm import tqdm, trange

class ContextQuestionsDataset(Dataset):
    def __init__(self, data: List[Dict[str, any]], model_type: str):
        self.examples = []
        self.model_type = model_type
        for item in data:
            context = item['context']
            for question in item['questions']:
                self.examples.append((context, question))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, question = self.examples[idx]
        if self.model_type in ['bert', 'roberta']:
            return f"{question} [SEP] {context}", question
        elif self.model_type == 't5':
            return f"question: {question} context: {context}", question
        elif self.model_type == 'gpt2':
            return f"Context: {context}\nQuestion: {question}\nAnswer:", question
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

def load_data(file_path: str) -> List[Dict[str, any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def fine_tune_model(model, train_dataloader: DataLoader, num_epochs: int):
    epoch_iterator = trange(num_epochs, desc="Epoch")
    for epoch in epoch_iterator:
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for input_text, target_text in batch_iterator:
            # Ensure input_text and target_text are strings, not tensors
            input_text = input_text[0]  # Assuming batch_size=1
            target_text = target_text[0]  # Assuming batch_size=1
            
            try:
                # Call the existing fine_tune method
                model.fine_tune(input_text, target_text)
                
                # Update the progress bar
                batch_iterator.set_postfix({"Status": "Processed"})
            except Exception as e:
                print(f"Error during fine-tuning: {str(e)}")
                print(f"Input text: {input_text}")
                print(f"Target text: {target_text}")
                continue
        
        epoch_iterator.set_postfix({"Status": "Completed"})

def main():
    parser = argparse.ArgumentParser(description="Fine-tune an abstract language model")
    parser.add_argument("--model_type", type=str, required=True, choices=['t5', 'bert', 'gpt2', 'roberta'], help="Type of model to fine-tune")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = load_data(args.data_path)
    dataset = ContextQuestionsDataset(data, args.model_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    print(f"Initializing {args.model_type} model...")
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
    print("Starting fine-tuning...")
    fine_tune_model(model, dataloader, args.num_epochs)

    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {args.output_dir}")
    model.save(args.output_dir)
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()
