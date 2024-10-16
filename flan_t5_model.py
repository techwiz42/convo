import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
import traceback
import os

class FLANT5LanguageModel(AbstractLanguageModel):
    def __init__(self, model_type: str, model_path: str):
        try:
            self.model_path = os.path.join(model_path, model_type)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize model and tokenizer
            print(f"MODEL PATH: {self.model_path}")
            if os.path.exists(self.model_path):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                print("Loaded locally trained model")
            else:
                print(f"Local model not found at {self.model_path}. Loading default FLAN-T5 model.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            
            self.model.eval()  # Set the model to evaluation mode
        except Exception as e:
            print(f"Error initializing FLAN-T5 model: {str(e)}")
            print(traceback.format_exc())

    def generate_response(self, input_text: str, temperature: float = 0.8, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            # Tokenize input text
            input_ids = self.tokenizer(f"Generate a response: {input_text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    min_length=input_ids.shape[1] + 20,  # Ensure some new tokens are generated
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                )
            
            # Decode the generated tokens, ignoring the input
            generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # If the generated text is too short or empty, try again with different parameters
            if len(generated_text.split()) < 5:
                return self.generate_response(input_text, temperature=1.0, top_p=0.95, max_new_tokens=max_new_tokens + 50)
            
            return generated_text.strip()
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
            if "An error occurred" not in target_text and len(target_text.split()) > 5:
                input_text = f"Generate a response: {input_text}"
                input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
                target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

                self.model.train()
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
            
                if loss is not None:
                    loss.backward()
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
                    optimizer.step()
                    print(f"Fine-tuning completed. Loss: {loss.item()}")
                else:
                    print("Warning: Loss is None. Skipping fine-tuning.")
                
                self.model.eval()  # Set back to evaluation mode after fine-tuning
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
            self.model.eval()  # Set the model to evaluation mode
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
