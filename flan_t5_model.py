# flan_t5_model.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
import traceback

class FLANT5LanguageModel(AbstractLanguageModel):
    def __init__(self, model_path: str):
        self.model_path = "/home/scooter/projects/convo/flan-t5" #FIXME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            input_ids = self.tokenizer(f"Generate a response: {input_text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
                inputs = self.tokenizer(f"Generate a response: {input_text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
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
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
