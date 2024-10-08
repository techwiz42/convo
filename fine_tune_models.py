# File: fine_tune_models.py

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os
from torch.cuda.amp import GradScaler
import gc
import traceback

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
                    'question': question,
                    'answer': answer,
                    'context': context,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        if self.model_type in ['gpt2', 'gpt-j']:
            input_text = f"Context: {item['context']} Q: {item['question']} A:"
            target_text = item['answer']
        elif self.model_type in ['t5', 'flan-t5']:
            input_text = f"question: {item['question']} context: {item['context']}"
            target_text = item['answer']
        elif self.model_type in ['bert', 'roberta']:
            input_text = f"{item['question']} [SEP] {item['context']}"
            target_text = item['answer']
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        if self.model_type in ['bert', 'roberta']:
            answer_start = item['context'].find(item['answer'])
            answer_end = answer_start + len(item['answer'])
            
            tokenized_answer = self.tokenizer(item['context'], return_offsets_mapping=True, max_length=self.max_length, truncation=True, padding='max_length')
            start_positions = end_positions = 0
            
            for idx, (start, end) in enumerate(tokenized_answer.offset_mapping):
                if start <= answer_start < end:
                    start_positions = idx
                if start < answer_end <= end:
                    end_positions = idx
                    break
            
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'start_positions': torch.tensor(start_positions, dtype=torch.long),
                'end_positions': torch.tensor(end_positions, dtype=torch.long)
            }
        else:
            targets = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': targets.input_ids.squeeze()
            }

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, gradient_accumulation_steps, model_type):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if model_type in ['bert', 'roberta']:
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
            else:
                labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():
                if model_type in ['bert', 'roberta']:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                progress_bar.update(100 if step != len(train_dataloader) - 1 else len(train_dataloader) % 100)
                progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
            
            del input_ids, attention_mask
            if model_type in ['bert', 'roberta']:
                del start_positions, end_positions
            else:
                del labels
            clear_cuda_memory()
        
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
                
                if model_type in ['bert', 'roberta']:
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                else:
                    labels = batch['labels'].to(device)

                with torch.cuda.amp.autocast():
                    if model_type in ['bert', 'roberta']:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                if (step + 1) % 100 == 0 or (step + 1) == len(val_dataloader):
                    val_progress_bar.update(100 if step != len(val_dataloader) - 1 else len(val_dataloader) % 100)
                    val_progress_bar.set_postfix({'loss': f"{total_val_loss / (step + 1):.4f}"})
                
                del input_ids, attention_mask
                if model_type in ['bert', 'roberta']:
                    del start_positions, end_positions
                else:
                    del labels
                clear_cuda_memory()
        
        val_progress_bar.close()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
        
        clear_cuda_memory()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for bidirectional question answering")
    parser.add_argument("--model_type", type=str, required=True, choices=['gpt2', 't5', 'bert', 'roberta', 'flan-t5', 'gpt-j'], help="Type of model to fine-tune")
    parser.add_argument("--model_name", type=str, default=None, help="Specific model name (e.g., 'gpt2-medium', 't5-base')")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data JSON file")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="models/", help="Directory to save the fine-tuned model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clear_cuda_memory()

    print("Loading tokenizer and model...")
    if args.model_name is None:
        model_paths = {
            "gpt2": "gpt2",
            "t5": "t5-small",
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "flan-t5": "google/flan-t5-small",
            "gpt-j": "EleutherAI/gpt-j-6B"
        }
        args.model_name = model_paths[args.model_type]

    if os.path.exists(args.output_dir):
        print(f"Loading previously fine-tuned model from {args.output_dir}")
        if args.model_type in ['t5', 'flan-t5']:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
        elif args.model_type in ['bert', 'roberta']:
            model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    else:
        print(f"Loading base model {args.model_name}")
        if args.model_type in ['t5', 'flan-t5']:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        elif args.model_type in ['bert', 'roberta']:
            model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = model.to(device)

    clear_cuda_memory()

    print("Preparing datasets...")
    train_data = load_json_data(args.train_data)
    val_data = load_json_data(args.val_data)

    train_dataset = BidirectionalQADataset(train_data, tokenizer, args.model_type)
    val_dataset = BidirectionalQADataset(val_data, tokenizer, args.model_type)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    clear_cuda_memory()

    print("Starting fine-tuning...")
    fine_tuned_model = train(model, train_dataloader, val_dataloader, device, args.num_epochs, args.learning_rate, args.gradient_accumulation_steps, args.model_type)

    clear_cuda_memory()

    print(f"Saving fine-tuned model to {args.output_dir}")
    fine_tuned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    clear_cuda_memory()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        clear_cuda_memory()
