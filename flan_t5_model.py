import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
from text_analysis import TextAnalyzer
import traceback
import random

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

    def generate_response(self, input_text: str, temperature: float = 0.9, top_p: float = 0.95, **kwargs) -> str:
        analyzer = TextAnalyzer()
        try:
            # Handle both max_length and max_new_tokens
            max_length = kwargs.get('max_length', 512)
            max_new_tokens = kwargs.get('max_new_tokens', None)

            # Tokenize input text on CPU
            input_ids = self.tokenizer(f"Generate a detailed, informative, and extensive response: {input_text}", 
                                       return_tensors="pt", 
                                       max_length=512, 
                                       truncation=True).input_ids

            # Prepare generation parameters
            gen_params = {
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "num_return_sequences": 3,
                "num_beams": 5,
                "no_repeat_ngram_size": 4,  # Increased from 3
                "repetition_penalty": 1.2,  # Increased from 1.0
                "length_penalty": 2.5,  # Increased from 2.0
                "early_stopping": False,  # Changed from True
                "min_length": 32,  # Added min_length parameter
            }

            # Generate three responses
            with torch.no_grad():
                # Move input_ids to the correct device just before generation
                input_ids = input_ids.to(self.device)
                outputs = self.model.generate(input_ids, **gen_params)
                # Move outputs back to CPU immediately
                outputs = outputs.cpu()

            # Decode the three generated sequences
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                print(f"TEXT: {text}")     
                sentiment, grammar = analyzer.analyze_text(text)           
                generated_texts.append({"text": text, "score": abs(sentiment) + grammar})
            # Select the most response with the highest score
            new_text = max(generated_texts, key=lambda x: x["score"]).get("text")

            return new_text

        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg:
                print("CUDA out of memory. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                return self.generate_response(input_text, temperature, top_p, **kwargs)
            elif "Already borrowed" in error_msg:
                print("CUDA error: Already borrowed. Retrying on CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                return self.generate_response(input_text, temperature, top_p, **kwargs)
            else:
                print(f"Runtime error in generate_response: {error_msg}")
                return "An error occurred while generating the response. Please try again."

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())

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
        pass
        # Commented out as in the GPT-2 model
        '''
        try:
            print(f"Saving model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(traceback.format_exc())
        '''

    def load(self, path: str) -> bool:
        pass
        # Commented out as in the GPT-2 model
        '''
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
        '''

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
