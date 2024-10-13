from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def compare_models(fine_tuned_path, base_model="gpt2"):
    # Load models
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_path)
    base_model = GPT2LMHeadModel.from_pretrained(base_model)

    # Compare a few weights
    for name, param in fine_tuned_model.named_parameters():
        if "weight" in name:
            base_param = base_model.get_parameter(name)
            diff = torch.abs(param - base_param).mean().item()
            print(f"{name}: Average absolute difference = {diff}")
            if diff < 1e-6:
                print(f"Warning: {name} appears unchanged from base model")
            print(f"Fine-tuned: {param.flatten()[:5]}")
            print(f"Base model: {base_param.flatten()[:5]}")
            print()
        
        if "embeddings" in name:
            break  # We don't need to check all layers

    # Compare configurations
    print("Configuration differences:")
    for key in fine_tuned_model.config.__dict__:
        if fine_tuned_model.config.__dict__[key] != base_model.config.__dict__[key]:
            print(f"{key}: Fine-tuned = {fine_tuned_model.config.__dict__[key]}, Base = {base_model.config.__dict__[key]}")

# Usage
if __name__ == "__main__":
    compare_models("/home/scooter/projects/convo/models/gpt2")
