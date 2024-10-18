import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os
import traceback

class PersonaChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for conversation in data:
            your_persona = " ".join(conversation.get('your_persona', []))
            partner_persona = " ".join(conversation.get('partner_persona', []))
            context = f"Your persona: {your_persona} Partner's persona: {partner_persona}"
            dialogue = conversation.get('dialogue', [])
            for i in range(0, len(dialogue) - 1, 2):
                self.examples.append({
                    'context': context,
                    'input': dialogue[i],
                    'target': dialogue[i+1] if i+1 < len(dialogue) else ""
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_text = f"context: {item['context']} input: {item['input']}"
        target_text = item['target']

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def load_personachat_data(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_conversation = {'your_persona': [], 'partner_persona': [], 'dialogue': []}
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue
            index, content = parts
            
            try:
                index = int(index)
            except ValueError:
                # If index is not a number, skip this line
                continue
            
            if index == 1 and current_conversation['dialogue']:
                conversations.append(current_conversation)
                current_conversation = {'your_persona': [], 'partner_persona': [], 'dialogue': []}
            
            if 1 <= index <= 5:
                if ': ' in content:
                    current_conversation['your_persona'].append(content.split(': ', 1)[1])
                else:
                    current_conversation['your_persona'].append(content)
            elif 6 <= index <= 10:
                if ': ' in content:
                    current_conversation['partner_persona'].append(content.split(': ', 1)[1])
                else:
                    current_conversation['partner_persona'].append(content)
            elif index > 10:
                current_conversation['dialogue'].append(content)
        
        if current_conversation['dialogue']:
            conversations.append(current_conversation)
    
    return conversations

def train(model, train_dataloader, val_dataloader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (step + 1) % 100 == 0 or (step + 1) == len(train_dataloader):
                progress_bar.update(100 if step != len(train_dataloader) - 1 else len(train_dataloader) % 100)
                progress_bar.set_postfix({'loss': total_loss / (step + 1)})            
        
        progress_bar.close()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs} completed. Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_progress_bar = tqdm(total=len(val_dataloader), desc="Validation", unit="batch")
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                if (step + 1) % 100 == 0 or (step + 1) == len(val_dataloader):
                    val_progress_bar.update(100 if step != len(val_dataloader) - 1 else len(val_dataloader) % 100)
                    val_progress_bar.set_postfix({'loss': total_val_loss / (step + 1)})
        
        val_progress_bar.close()
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation completed. Average validation loss: {avg_val_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 model for PersonaChat data")
    parser.add_argument("--model_path", type=str, default="./models/flan-t5", help="Path to the local model directory")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data TXT file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to the validation data TXT file")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="./models/flan-t5-personachat", help="Directory to save the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True).to(device)

    print("Preparing datasets...")
    train_data = load_personachat_data(args.train_data)
    val_data = load_personachat_data(args.val_data)

    train_dataset = PersonaChatDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = PersonaChatDataset(val_data, tokenizer, max_length=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("Starting fine-tuning...")
    fine_tuned_model = train(model, train_dataloader, val_dataloader, device, args)

    print(f"Saving fine-tuned model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    fine_tuned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()
