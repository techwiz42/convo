import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForQuestionAnswering, BertTokenizer,
    RobertaForQuestionAnswering, RobertaTokenizer,
    BlenderbotForConditionalGeneration, BlenderbotTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
import traceback
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os

DECREASE_RATE = 0.9

class BidirectionalQADataset(Dataset):
    def __init__(self, data, tokenizer, model_type, max_length=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        for item in data:
            context = item['context']
            for question, answer in zip(item.get('questions'), item.get('answers')):
                #print(f"context: {context}, question: {question}, answer: {answer}")
                self.examples.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        
        if self.model_type == 'gpt2':
            input_text = f"Context: {item['context']} Question: {item['question']} Answer: {item['answer']}<|endoftext|>"
            encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            }
        
        elif self.model_type in ['t5', 'flan-t5']:
            input_text = f"question: {item['question']} context: {item['context']}"
            target_text = item['answer']
            input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }
        elif self.model_type == 'blenderbot':
            input_text = f"Human: {item['question']} Context: {item['context']}"
            target_text = f"Assistant: {item['answer']}"
            input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }    
        elif self.model_type in ['bert', 'roberta']:
            question_tokens = self.tokenizer.tokenize(item['question'])
            context_tokens = self.tokenizer.tokenize(item['context'])
            answer_tokens = self.tokenizer.tokenize(item['answer'])

            # Truncate context if necessary
            max_context_length = self.max_length - len(question_tokens) - 3  # 3 for [CLS], [SEP], [SEP]
            if len(context_tokens) > max_context_length:
                context_tokens = context_tokens[:max_context_length]

            # Combine tokens
            tokens = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + context_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            # Pad sequences
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

            # Find start and end positions of the answer
            answer_start = ' '.join(context_tokens).find(' '.join(answer_tokens))
            if answer_start != -1:
                answer_end = answer_start + len(' '.join(answer_tokens))
                answer_start_token = len(question_tokens) + 2  # Adjust for [CLS] and [SEP]
                for i, token in enumerate(context_tokens):
                    if i == answer_start:
                        break
                    answer_start_token += 1
                answer_end_token = answer_start_token + len(answer_tokens) - 1
            else:
                answer_start_token = answer_end_token = 0

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'start_positions': torch.tensor(answer_start_token, dtype=torch.long),
                'end_positions': torch.tensor(answer_end_token, dtype=torch.long)
            }

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def train(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, gradient_accumulation_steps, model_type):
    torch.cuda.empty_cache()
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
            
            if model_type in ['bert', 'roberta']:
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
            else:
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # Debug print
            #print(f"Step {step}, Raw loss: {loss.item()}")

            # Check for NaN loss
            if torch.isnan(loss).any():
                #print(f"NaN loss detected at step {step}. Skipping this batch.")
                continue

            # Loss scaling
            loss = loss * 100  # Scale up the loss

            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Debug print for gradients
                #for name, param in model.named_parameters():
                    #if param.grad is not None:
                        #print(f"Gradient norm for {name}: {param.grad.norm().item()}")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                progress_bar.update(100 if step != len(train_dataloader) - 1 else len(train_dataloader) % 100)
                progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
            
            # Memory management
            del input_ids, attention_mask, outputs, loss
            if model_type in ['bert', 'roberta']:
                del start_positions, end_positions
            else:
                del labels
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
                
                if model_type in ['bert', 'roberta']:
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                else:
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                if (step + 1) % 100 == 0 or (step + 1) == len(val_dataloader):
                    val_progress_bar.update(100 if step != len(val_dataloader) - 1 else len(val_dataloader) % 100)
                    val_progress_bar.set_postfix({'loss': f"{total_val_loss / (step + 1):.4f}"})
                
                # Memory management
                del input_ids, attention_mask, outputs, loss
                if model_type in ['bert', 'roberta']:
                    del start_positions, end_positions
                else:
                    del labels
                torch.cuda.empty_cache()
        
        val_progress_bar.close()
        learning_rate *= DECREASE_RATE
        print(f"Learning rate decreased to {learning_rate}") 
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
    
    return model

def get_default_model_name(model_type):
    default_models = {
        'gpt2': 'gpt2',
        't5': 't5-base',
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'flan-t5': 'google/flan-t5-base',
        'blenderbot': 'facebook/blenderbot-90M'
    }
    return default_models.get(model_type, model_type)


def prepare_data_for_blenderbot(examples, tokenizer, max_length=128):
    inputs = tokenizer(examples["input_text"], truncation=True, max_length=max_length, padding="max_length")
    outputs = tokenizer(examples["target_text"], truncation=True, max_length=max_length, padding="max_length")
    
    inputs["labels"] = outputs["input_ids"]
    
    return inputs


def main():
    try:
        parser = argparse.ArgumentParser(description="Fine-tune a model for bidirectional question answering")
        parser.add_argument("--model_type", type=str, required=True, choices=['gpt2', 't5', 'bert', 'roberta', 'flan-t5', 'blenderbot'], help="Type of model to fine-tune")
        parser.add_argument("--model_name", type=str, default=None, help="Specific model name (e.g., 'gpt2-medium', 't5-base')")
        parser.add_argument("--train_data", type=str, required=True, help="Path to the training data JSON file")
        parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data JSON file")
        parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
        parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model (default: './models/<model_type>')")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass")
        parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")

        args = parser.parse_args()

        # Set the default output directory if not provided
        if args.output_dir is None:
            args.output_dir = os.path.join("./models", args.model_type)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        args.model_name = get_default_model_name(args.model_type)

        print("Loading tokenizer and model...")
        if args.model_name is None:
            args.model_name = args.model_type

        # Check if the output directory contains a valid model
        model = None
        if os.path.exists(args.output_dir) and os.path.isfile(os.path.join(args.output_dir, "config.json")):
            print(f"Loading previously fine-tuned model from {args.output_dir}")
            try:
                if args.model_type == 'gpt2':
                    model = GPT2LMHeadModel.from_pretrained(args.output_dir)
                    tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
                elif args.model_type == 't5':
                    model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
                    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
                elif args.model_type == 'blenderbot':
                    model = BlenderbotForConditionalGeneration.from_pretrained(args.output_dir)
                    tokenizer = BlenderbotTokenizer.from_pretrained(args.output_dir)
                elif args.model_type == 'bert':
                    model = BertForQuestionAnswering.from_pretrained(args.output_dir)
                    tokenizer = BertTokenizer.from_pretrained(args.output_dir)
                elif args.model_type == 'roberta':
                    model = RobertaForQuestionAnswering.from_pretrained(args.output_dir)
                    tokenizer = RobertaTokenizer.from_pretrained(args.output_dir)
                elif args.model_type == 'flan-t5':
                    model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
                    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            except Exception as e:
                print(f"Error loading model from {args.output_dir}: {str(e)}")
                print("Will load the base model instead.")
                model = None

        if model is None:
            print(f"Loading base model {args.model_name}")
            if args.model_type == 'gpt2':
                model = GPT2LMHeadModel.from_pretrained(args.model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
            elif args.model_type == 't5':
                model = T5ForConditionalGeneration.from_pretrained(args.model_name)
                tokenizer = T5Tokenizer.from_pretrained(args.model_name)
            elif args.model_type == 'blenderbot':
                model = BlenderbotForConditionalGeneration.from_pretrained(args.model_name)
                tokenizer = BlenderbotTokenizer.from_pretrained(args.model_name)
            elif args.model_type == 'bert':
                model = BertForQuestionAnswering.from_pretrained(args.model_name)
                tokenizer = BertTokenizer.from_pretrained(args.model_name)
            elif args.model_type == 'roberta':
                model = RobertaForQuestionAnswering.from_pretrained(args.model_name)
                tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
            elif args.model_type == 'flan-t5':
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if args.model_type in ['gpt2']:
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
        os.makedirs(args.output_dir, exist_ok=True)
        fine_tuned_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("All done!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
