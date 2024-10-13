# t5_model.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from abstract_model import AbstractLanguageModel
import traceback
import os
import traceback

class T5LanguageModel(AbstractLanguageModel):
    def __init__(self, model_name, model_path: str):
        try:
            self.model_path = "/home/scooter/projects/convo/t5" #FIXME
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize model and tokenizer
            print(f"MODEL PATH: {self.model_path}")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        except Exception as e:
            print(traceback.format_exc())

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
        
            # T5 doesn't have a max_position_embeddings attribute, so we'll use a fixed maximum length
            max_length = min(1024, input_ids.shape[1] + max_new_tokens)
        
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
            if not generated_text:
                generated_text = "I'm sorry, but I couldn't generate a meaningful response. Could you please rephrase your input?"
        
            return generated_text
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
                input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
                target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

                # Ensure input and target have the same batch size
                if input_ids.size(0) != target_ids.size(0):
                    target_ids = target_ids[:input_ids.size(0), :]

                outputs = self.model(input_ids=input_ids, labels=target_ids)
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
            print(f"Saving model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(traceback.format_exc())

    def load(self, path: str) -> bool:
        try:
            print(f"Loading model from {path}")
            self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self) -> T5Tokenizer:
        return self.tokenizer
