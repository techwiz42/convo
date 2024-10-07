import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
import traceback

class GPTJLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_gptj"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
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
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

                outputs = self.model(input_ids=inputs, labels=targets)
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
                self.model = AutoModelForCausalLM.from_pretrained(self.user_model_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
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

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
