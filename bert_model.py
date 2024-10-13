# bert_model.py
import os
import torch
from transformers import BertForMaskedLM, BertTokenizer
from abstract_model import AbstractLanguageModel
import traceback

class BERTLanguageModel(AbstractLanguageModel):
    def __init__(self, model_type: str,  model_path: str):
        self.model_path = os.path.join(model_path, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MODEL_PATH {self.model_path}") 
        # Initialize model and tokenizer
        self.model = BertForMaskedLM.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        try:
            # Tokenize input text
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate response iteratively
            for _ in range(max_new_tokens):
                # Predict next token
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    predictions = outputs.logits[:, -1, :]
                
                # Apply temperature
                predictions = predictions / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(predictions, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                predictions[indices_to_remove] = float('-inf')
                
                # Sample next token
                next_token = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)
                
                # Append next token to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we generate the end of sequence token
                if next_token.item() == self.tokenizer.sep_token_id:
                    break
            
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
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
                inputs = self.tokenizer(input_text, target_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(self.device)
                inputs['labels'] = inputs.input_ids.clone()

                outputs = self.model(**inputs)
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
            self.model = BertForMaskedLM.from_pretrained(path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self) -> BertTokenizer:
        return self.tokenizer
