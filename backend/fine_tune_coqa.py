import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os
import traceback

DECREASE_RATE = 0.75

class ContextQuestionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for item in data:
            context = item['context']
            for question in item['questions']:
                self.examples.append({
                    'question': question,
                    'context': context,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_text = f"context: {item['context']} question: {item['question']}"
        target_text = "Generate a concise answer based on the given context."

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def train(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, gradient_accumulation_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 100 == 0:            
                progress_bar.update(step)
                progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
        
        progress_bar.close()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
        
        learning_rate *= DECREASE_RATE
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        print(f"Learning rate decreased to {learning_rate}")
    
    return model

def main():
    try:
        parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 model for context and question data")
        parser.add_argument("--model_type", type=str, default="flan-t5", help="Type of model to fine-tune")
        parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Specific model name")
        parser.add_argument("--train_data", type=str, required=True, help="Path to the training data JSON file")
        parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data JSON file")
        parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
        parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass")
        parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

        args = parser.parse_args()

        if args.output_dir is None:
            args.output_dir = os.path.join("./models", args.model_type)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

        print("Preparing datasets...")
        train_data = load_json_data(args.train_data)
        val_data = load_json_data(args.val_data)

        train_dataset = ContextQuestionDataset(train_data, tokenizer, max_length=args.max_length)
        val_dataset = ContextQuestionDataset(val_data, tokenizer, max_length=args.max_length)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

        print("Starting fine-tuning...")
        fine_tuned_model = train(model, train_dataloader, val_dataloader, device, args.num_epochs, args.learning_rate, args.gradient_accumulation_steps)

        print(f"Saving fine-tuned model to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        fine_tuned_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Fine-tuning completed!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
