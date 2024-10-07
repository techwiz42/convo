import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from abstract_model import AbstractLanguageModel
import traceback

class GPT2LanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_gpt2"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = GPT2LMHeadModel.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.user_model_path)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
        
            # Ensure we're not exceeding the model's maximum context length
            max_length = min(self.model.config.max_position_embeddings, input_ids.shape[1] + max_new_tokens)
        
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
            # Ensure we're only returning the newly generated text
            new_text = generated_text[len(input_text):].strip()
        
            if not new_text:
                new_text = "I'm sorry, but I couldn't generate a meaningful response. Could you please rephrase your input?"
        
            return new_text
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory. Attempting to free cache and retry...")
                torch.cuda.empty_cache()
                # Retry generation with reduced parameters
                return self.generate_response(input_text, temperature, top_p, max(50, max_new_tokens // 2))
            else:
                print(f"CUDA error: {str(e)}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                return self.generate_response(input_text, temperature, top_p, max_new_tokens)
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            return "An error occurred while generating the response. Please try again."

    def fine_tune(self, input_text: str, target_text: str):
        try:
            # Only fine-tune if the target_text is not an error message
            if "An error occurred" not in target_text and len(target_text.split()) > 5:
                # Tokenize input and target
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                target_ids = self.tokenizer.encode(target_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)

                # Ensure input and target have the same sequence length
                max_length = max(input_ids.size(1), target_ids.size(1))
                input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding="max_length", max_length=max_length, return_tensors="pt")["input_ids"].to(self.device)
                target_ids = self.tokenizer.pad({"input_ids": target_ids}, padding="max_length", max_length=max_length, return_tensors="pt")["input_ids"].to(self.device)

                # Create attention masks
                input_mask = (input_ids != self.tokenizer.pad_token_id).long()
                target_mask = (target_ids != self.tokenizer.pad_token_id).long()

                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, labels=target_ids)
                loss = outputs.loss
        
                if loss is not None:
                    loss.backward()
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
                    optimizer.step()
                    print(f"Fine-tuning completed. Loss: {loss.item()}")
                else:
                    print("Warning: Loss is None. Skipping fine-tuning.")
            else:
                print("Skipping fine-tuning due to invalid target text.")
        except Exception as e:
            print(f"Error in fine_tune: {str(e)}")
            print(traceback.format_exc())

    def save(self, path: str):
        try:
            print(f"Saving model to {self.user_model_path}")
            self.model.save_pretrained(self.user_model_path)
            self.tokenizer.save_pretrained(self.user_model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(traceback.format_exc())

    def load(self, path: str) -> bool:
        try:
            if os.path.exists(self.user_model_path):
                print(f"Loading model from {self.user_model_path}")
                self.model = GPT2LMHeadModel.from_pretrained(self.user_model_path).to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.user_model_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                print("Model loaded successfully")
                return True
            else:
                print(f"No saved model found at {self.user_model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self) -> GPT2Tokenizer:
        return self.tokenizer
