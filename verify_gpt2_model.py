from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def verify_gpt2_model(model_name="gpt2"):
    try:
        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        print(f"Model {model_name} loaded successfully.")

        # Check model attributes
        print(f"Model type: {type(model)}")
        print(f"Number of parameters: {model.num_parameters()}")
        print(f"Vocabulary size: {len(tokenizer)}")

        # Check model configuration
        print(f"Model configuration: {model.config}")

        # Test with a simple input
        input_text = "Hello, how are you?"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Input: {input_text}")
        print(f"Output: {generated_text}")

        return True
    except Exception as e:
        print(f"Error loading or verifying model: {str(e)}")
        return False

# Usage
if verify_gpt2_model():
    print("Model verified successfully.")
else:
    print("Model verification failed.")
