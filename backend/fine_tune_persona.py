import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import argparse
import os
import re
import html

def clean_text(text):
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove control characters and non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    # Remove special tokens and repetitive padding
    text = re.sub(r'(<pad>|</s>){2,}', '', text)  # Remove repetitive <pad> and </s>
    text = re.sub(r'<pad>|</s>', '', text)  # Remove any remaining single <pad> or </s>
    
    # Remove any remaining repetitive patterns (e.g., repeated spaces or punctuation)
    text = re.sub(r'(\S)(\1{3,})', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()
def load_personachat_data(file_path):
    data = []
    current_conversation = None
    line_number = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        print("Processing input...", end = '', flush = True)
        for line in f:
            line_number += 1
            line = clean_text(line)
            
            if not line:  # Skip empty lines after cleaning
                continue
            
            if line.startswith('1 your persona:'):
                if current_conversation:
                    data.append(current_conversation)
                current_conversation = {'your_persona': [], 'partner_persona': [], 'utterances': []}
            
            parts = line.split(' ', 1)
            if len(parts) == 2:
                number, content = parts
                if content.startswith('your persona:'):
                    current_conversation['your_persona'].append(clean_text(content[13:]))
                elif content.startswith('partner\'s persona:'):
                    current_conversation['partner_persona'].append(clean_text(content[18:]))
                else:
                    # Split utterances by the bar character and clean each utterance
                    utterances = content.split('|')
                    current_conversation['utterances'].extend([clean_text(u) for u in utterances if clean_text(u)])
            
            if line_number % 1000 == 0:
                print('.', end='', flush=True)

    if current_conversation:
        data.append(current_conversation)

    print(f"Finished processing {line_number} lines.")
    print(f"Loaded {len(data)} conversations from {file_path}")
    
    if data:
        print("Sample data:")
        print(f"Your persona: {data[0]['your_persona']}")
        print(f"Partner's persona: {data[0]['partner_persona']}")
        print(f"First 2 utterances: {data[0]['utterances'][:2]}")
    else:
        print("No data loaded.")

    return data

class PersonaChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for conversation in data:
            your_persona = " ".join(conversation['your_persona'])
            partner_persona = " ".join(conversation['partner_persona'])
            context = f"Your persona: {your_persona} Partner's persona: {partner_persona}"
            
            utterances = conversation['utterances']
            for i in range(0, len(utterances) - 1, 2):
                human_utterance = utterances[i].strip()
                ai_utterance = utterances[i + 1].strip() if i + 1 < len(utterances) else ""
                
                self.examples.append({
                    'context': context,
                    'human_input': human_utterance,
                    'ai_response': ai_utterance
                })
                
                # Update context for the next turn
                context += f" Human: {human_utterance} AI: {ai_utterance}"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_text = f"Context: {item['context']} Human: {item['human_input']}"
        target_text = item['ai_response']

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                progress_bar.update(100 if step != len(train_dataloader) - 1 else len(train_dataloader) % 100)
                progress_bar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
        
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
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                if (step + 1) % 100 == 0 or (step + 1) == len(val_dataloader):
                    val_progress_bar.update(100 if step != len(val_dataloader) - 1 else len(val_dataloader) % 100)
                    val_progress_bar.set_postfix({'loss': f"{total_val_loss / (step + 1):.4f}"})
        
        val_progress_bar.close()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 model on PersonaChat data")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="FLAN-T5 model to use")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data text file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data text file")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="./models/flan-t5-personachat", help="Directory to save the fine-tuned model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    model = None
    tokenizer = None

    if os.path.exists(args.output_dir) and os.path.isfile(os.path.join(args.output_dir, "config.json")):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            print(f"Loaded model and tokenizer from {args.output_dir}")
        except Exception as e:
            print(f"Error loading model from {args.output_dir}: {str(e)}")

    if model is None:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            print(f"Loaded model and tokenizer: {args.model_name}")
        except Exception as e:
            print(f"Error loading specified model {args.model_name}: {str(e)}")
            print("Falling back to default FLAN-T5 base model")
            default_model = "google/flan-t5-base"
            model = AutoModelForSeq2SeqLM.from_pretrained(default_model)
            tokenizer = AutoTokenizer.from_pretrained(default_model)
            print(f"Loaded default model and tokenizer: {default_model}")

    model = model.to(device)   
 
    print("Preparing datasets...")
    train_data = load_personachat_data(args.train_data)
    val_data = load_personachat_data(args.val_data)

    if not train_data or not val_data:
        print("No data loaded. Exiting.")
        return

    train_dataset = PersonaChatDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = PersonaChatDataset(val_data, tokenizer, max_length=args.max_length)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    if len(train_dataset) > 0:
        print("Sample input:")
        sample = train_dataset[0]
        print(tokenizer.decode(sample['input_ids']))
        print("Sample target:")
        print(tokenizer.decode(sample['labels']))

    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("Starting fine-tuning...")
    fine_tuned_model = train(model, train_dataloader, val_dataloader, device, args.num_epochs, args.learning_rate, args.gradient_accumulation_steps)

    print(f"Saving fine-tuned model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    fine_tuned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()
