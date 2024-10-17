import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
import traceback

class FLANT5LanguageModel(AbstractLanguageModel):
    def __init__(self, model_type: str, model_path: str):
        self.model_path = os.path.join(model_path, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                print("Loaded locally trained model")
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                print("Falling back to default FLAN-T5 model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        else:
            print(f"Local model not found at {self.model_path}. Loading default FLAN-T5 model.")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 200, min_new_tokens: int = 100) -> str:
        try:
            # Tokenize input text
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return text
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
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)

                # Forward pass and calculate loss
                outputs = self.model(**inputs, labels=targets.input_ids)
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
            if os.path.exists(path):
                print(f"Loading model from {path}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                print("Model loaded successfully")
                return True
            else:
                print(f"No saved model found at {path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self):
        return self.tokenizer
