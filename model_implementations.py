from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForQuestionAnswering, BertTokenizer,
    RobertaForQuestionAnswering, RobertaTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer
)
import torch
from typing import Dict, Any
import os
import requests
from transformers import utils
import socket
from abc import ABC, abstractmethod
import nltk
import traceback

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class Conversation:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = ""
        self.previous_questions = set()
        self.last_question = None

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        pass

    @abstractmethod
    def fine_tune(self, input_text: str, target_text: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        pass

    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizer:
        pass

class T5LanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_t5"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = T5ForConditionalGeneration.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.user_model_path)
        elif os.path.exists(self.model_path):
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        output = self.model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature, 
            top_p=top_p, 
            do_sample=True, 
            num_return_sequences=1
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune(self, input_text: str, target_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.step()
        else:
            print("Warning: Loss is None. Check input formatting.")

    def save(self, path: str):
        self.model.save_pretrained(self.user_model_path)
        self.tokenizer.save_pretrained(self.user_model_path)

    def load(self, path: str) -> bool:
        if os.path.exists(self.user_model_path):
            self.model = T5ForConditionalGeneration.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.user_model_path)
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class BERTLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_bert"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = BertForQuestionAnswering.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(self.user_model_path)
        elif os.path.exists(self.model_path):
            self.model = BertForQuestionAnswering.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        else:
            self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        return answer

    def fine_tune(self, input_text: str, target_text: str):
        encoding = self.tokenizer(input_text, target_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        start_positions = torch.tensor([1]).to(self.device)
        end_positions = torch.tensor([len(input_ids[0]) - 1]).to(self.device)

        outputs = self.model(input_ids, 
                             attention_mask=attention_mask, 
                             start_positions=start_positions, 
                             end_positions=end_positions)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.step()
        else:
            print("Warning: Loss is None. Check input formatting.")

    def save(self, path: str):
        self.model.save_pretrained(self.user_model_path)
        self.tokenizer.save_pretrained(self.user_model_path)

    def load(self, path: str) -> bool:
        if os.path.exists(self.user_model_path):
            self.model = BertForQuestionAnswering.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(self.user_model_path)
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

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
            print(traceback.format_exc())

    def fine_tune(self, input_text: str, target_text: str):
        # Only fine-tune if the target_text is not an error message
        if "An error occurred" not in target_text and len(target_text.split()) > 5:
            try:
                input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
                target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
            
                if loss is not None:
                    loss.backward()
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
                    optimizer.step()
                    print(f"Fine-tuning completed. Loss: {loss.item()}")
                else:
                    print("Warning: Loss is None. Skipping fine-tuning.")
            except Exception as e:
                print(traceback.format_exc())
        else:
            print("Skipping fine-tuning due to invalid target text.")


    def save(self, path: str):
        try:
            print(f"Saving model to {self.user_model_path}")
            self.model.save_pretrained(self.user_model_path)
            self.tokenizer.save_pretrained(self.user_model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

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
            return False

    def get_tokenizer(self) -> GPT2Tokenizer:
        return self.tokenizer

class RoBERTaLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_roberta"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = RobertaForQuestionAnswering.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.user_model_path)
        elif os.path.exists(self.model_path):
            self.model = RobertaForQuestionAnswering.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
        else:
            self.model = RobertaForQuestionAnswering.from_pretrained("roberta-base").to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        return answer

    def fine_tune(self, input_text: str, target_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.step()
        else:
            print("Warning: Loss is None. Check input formatting.")

    def save(self, path: str):
        self.model.save_pretrained(self.user_model_path)
        self.tokenizer.save_pretrained(self.user_model_path)

    def load(self, path: str) -> bool:
        if os.path.exists(self.user_model_path):
            self.model = RobertaForQuestionAnswering.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.user_model_path)
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class FLANT5LanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_flan_t5"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
        elif os.path.exists(self.model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        input_ids = self.tokenizer(f"Generate a response to the following: {input_text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature, 
            top_p=top_p, 
            do_sample=True, 
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def fine_tune(self, input_text: str, target_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.step()
        else:
            print("Warning: Loss is None. Check input formatting.")

    def save(self, path: str):
        self.model.save_pretrained(self.user_model_path)
        self.tokenizer.save_pretrained(self.user_model_path)

    def load(self, path: str) -> bool:
        if os.path.exists(self.user_model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class GPTJLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str, model_path: str):
        self.user_id = user_id
        self.model_path = model_path
        self.user_model_path = f".models/{user_id}_gptj"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        if os.path.exists(self.user_model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
        elif os.path.exists(self.model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_length: int = 100) -> str:
        input_ids = self.tokenizer(f"Human: {input_text}\nAI:", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature, 
            top_p=top_p, 
            do_sample=True, 
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def fine_tune(self, input_text: str, target_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        
        if loss is not None:
            loss.backward()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            optimizer.step()
        else:
            print("Warning: Loss is None. Check input formatting.")

    def save(self, path: str):
        self.model.save_pretrained(self.user_model_path)
        self.tokenizer.save_pretrained(self.user_model_path)

    def load(self, path: str) -> bool:
        if os.path.exists(self.user_model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.user_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.user_model_path)
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
