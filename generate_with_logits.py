import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_with_logits(model_path, input_text):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    for _ in range(50):  # Generate up to 50 new tokens
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            
        next_token_logits = logits[0]
        
        # Print top 5 most likely next tokens
        top_5 = torch.topk(next_token_logits, 5)
        print("Top 5 next tokens:")
        for i in range(5):
            token = tokenizer.decode([top_5.indices[i]])
            prob = torch.softmax(next_token_logits, dim=0)[top_5.indices[i]]
            print(f"{token}: {prob.item():.4f}")
        
        # Sample next token
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
        
        # Break if we generate an EOS token
        if next_token.item() == model.config.eos_token_id:
            break
        
        # Add the next token to the input sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        print(f"Generated: {tokenizer.decode(input_ids[0])}")
        print()

    return tokenizer.decode(input_ids[0])

# Usage
result = generate_with_logits("/home/scooter/projects/convo/models/gpt2", "Hello, how are you?")
print("Final result:", result)
