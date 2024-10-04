import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForQuestionAnswering, BertTokenizer,
    RobertaForQuestionAnswering, RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os

class BidirectionalQADataset(Dataset):
    def __init__(self, data, tokenizer, model_type, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

        for item in data:
            context = item['context']
            for question, answer in zip(item['questions'], item['answers']):
                self.examples.append({
                    'input': question,
                    'output': answer,
                    'context': context,
                    'type': 'question'
                })
                self.examples.append({
                    'input': answer,
                    'output': question,
                    'context': context,
                    'type': 'answer'
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        if self.model_type == 'gpt2':
            if item['type'] == 'question':
                input_text = f"Context: {item['context']} Q: {item['input']} A:"
            else:
                input_text = f"Context: {item['context']} A: {item['input']} Q:"
            encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            target_encoding = self.tokenizer(item['output'], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }
        elif self.model_type == 't5':
            if item['type'] == 'question':
                input_text = f"answer: {item['context']} {item['input']}"
            else:
                input_text = f"generate question: {item['context']} {item['input']}"
            target_text = item['output']
            input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }
        elif self.model_type in ['bert', 'roberta']:
            encoding = self.tokenizer(item['context'], item['input'], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            }

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def train(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, gradient_accumulation_steps, model_type):
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
            
            if model_type in ['gpt2', 't5']:
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            elif model_type in ['bert', 'roberta']:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                progress_bar.update(100 if step != len(train_dataloader) - 1 else len(train_dataloader) % 100)
                progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
            
            del input_ids, attention_mask, outputs
            if model_type in ['gpt2', 't5']:
                del labels
            if model_type in ['bert', 'roberta']:
                del token_type_ids
            torch.cuda.empty_cache()
        
        progress_bar.close()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        val_progress_bar = tqdm(total=len(val_dataloader), desc="Validation", unit="batch")
        
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                if model_type in ['gpt2', 't5']:
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                elif model_type in ['bert', 'roberta']:
                    token_type_ids = batch['token_type_ids'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                if (step + 1) % 100 == 0 or (step + 1) == len(val_dataloader):
                    val_progress_bar.update(100 if step != len(val_dataloader) - 1 else len(val_dataloader) % 100)
                    val_progress_bar.set_postfix({'loss': f"{total_val_loss / (step + 1):.4f}"})
                
                del input_ids, attention_mask, outputs
                if model_type in ['gpt2', 't5']:
                    del labels
                if model_type in ['bert', 'roberta']:
                    del token_type_ids
                torch.cuda.empty_cache()
        
        val_progress_bar.close()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for bidirectional question answering")
    parser.add_argument("--model_type", type=str, required=True, choices=['gpt2', 't5', 'bert', 'roberta'], help="Type of model to fine-tune")
    parser.add_argument("--model_name", type=str, default=None, help="Specific model name (e.g., 'gpt2-medium', 't5-base')")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data JSON file")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    if args.model_name is None:
        if args.model_type == 't5':
            args.model_name = 't5-base'
        elif args.model_type == 'bert':
            args.model_name = 'bert-base-uncased'
        elif args.model_type == 'roberta':
            args.model_name = 'roberta-base'
        else:
            args.model_name = args.model_type

    # Function to check if the output directory contains a valid model
    def is_valid_model_dir(dir_path):
        return os.path.exists(os.path.join(dir_path, "config.json")) and \
               os.path.exists(os.path.join(dir_path, "pytorch_model.bin"))

    if os.path.exists(args.output_dir) and is_valid_model_dir(args.output_dir):
        print(f"Loading previously fine-tuned model from {args.output_dir}")
        if args.model_type == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(args.output_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        elif args.model_type == 't5':
            model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
            tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        elif args.model_type == 'bert':
            model = BertForQuestionAnswering.from_pretrained(args.output_dir)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir)
        elif args.model_type == 'roberta':
            model = RobertaForQuestionAnswering.from_pretrained(args.output_dir)
            tokenizer = RobertaTokenizer.from_pretrained(args.output_dir)
    else:
        print(f"Loading base model {args.model_name}")
        if args.model_type == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(args.model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        elif args.model_type == 't5':
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
            tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        elif args.model_type == 'bert':
            model = BertForQuestionAnswering.from_pretrained(args.model_name)
            tokenizer = BertTokenizer.from_pretrained(args.model_name)
        elif args.model_type == 'roberta':
            model = RobertaForQuestionAnswering.from_pretrained(args.model_name)
            tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = model.to(device)

    print("Preparing datasets...")
    train_data = load_json_data(args.train_data)
    val_data = load_json_data(args.val_data)

    train_dataset = BidirectionalQADataset(train_data, tokenizer, args.model_type)
    val_dataset = BidirectionalQADataset(val_data, tokenizer, args.model_type)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("Starting fine-tuning...")
    fine_tuned_model = train(model, train_dataloader, val_dataloader, device, args.num_epochs, args.learning_rate, args.gradient_accumulation_steps, args.model_type)

    print(f"Saving fine-tuned model to {args.output_dir}")
    fine_tuned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
