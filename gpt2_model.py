import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from abstract_model import AbstractLanguageModel
from text_anysis import TextAnalyzer
import traceback

class GPT2LanguageModel(AbstractLanguageModel):
    def __init__(self, model_type: str, model_path: str):
        self.model_path = os.path.join(model_path, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_analyzer = TextAnalyzer()
        
        print(f"Loading model from: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
                print("Loaded locally trained model")
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                print("Falling back to default GPT-2 model")
                self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            print(f"Local model not found at {self.model_path}. Loading default GPT-2 model.")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #print(f"Model config: {self.model.config}")      
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_response(self, input_text: str, context: str = "", temperature: float = 1.0, top_p: float = 1.0, max_new_tokens: int = 50) -> List[str]:
        try:
            full_input = f"{context}\n{input_text}".strip()
            input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            max_length = min(self.model.config.n_positions, input_ids.shape[1] + max_new_tokens)

            responses = {}
            param_sets = [
                {"temperature": 0.7, "top_p": 0.9},
                {"temperature": 1.0, "top_p": 1.0},
                {"temperature": 1.2, "top_p": 0.95}
            ]

            for params in param_sets:
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                new_text = generated_text[len(full_input):].strip()
                sentiment, grammar = text_analyzer.analyze(new_text)
            
                if not new_text:
                    print(f"Warning: No new text generated for params: {params}")
                    new_text = "Null" 
                    score = -1
                print(f"{new_text}, {score}\n\n")
                responses.update({"response" new_text, "score": abs(sentiment) + grammar})

            return max(responses, key=lambda x: x["score"]).get("response")
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            return ["An error occurred while generating the response. Please try again."] * 3

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
        pass
        '''
        try:
            print(f"Saving model to {self.user_model_path}")
            self.model.save_pretrained(self.user_model_path)
            self.tokenizer.save_pretrained(self.user_model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(traceback.format_exc())
        '''

    def load(self, path: str) -> bool:
        pass
        '''
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
        '''

    def get_tokenizer(self) -> GPT2Tokenizer:
        return self.tokenizer
