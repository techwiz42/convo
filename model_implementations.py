from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BertForQuestionAnswering, BertTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    RobertaForQuestionAnswering, RobertaTokenizer,
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

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_response(self, input_text: str) -> str:
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
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        # Set timeout for downloads
        utils.TIMEOUT = 500
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(500)

        # Initialize model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small", local_files_only=False).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", local_files_only=False)


    def generate_response(self, input_text: str) -> str:
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
        output = self.model.generate(input_ids, max_length=128, num_return_sequences=1)
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
        self.model.save_pretrained(f"{path}_t5")
        self.tokenizer.save_pretrained(f"{path}_t5")

    def load(self, path: str) -> bool:
        if os.path.exists(f"{path}_t5"):
            self.model = T5ForConditionalGeneration.from_pretrained(f"{path}_t5").to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(f"{path}_t5")
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class BERTLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(self.device)
       # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        # Initialize model and tokenizer
        self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def generate_response(self, input_text: str) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        return answer

    def fine_tune(self, input_text: str, target_text: str):
        # Format input for question answering
        encoding = self.tokenizer(input_text, target_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Randomly set start and end positions for this example
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
        self.model.save_pretrained(f"{path}_bert")
        self.tokenizer.save_pretrained(f"{path}_bert")

    def load(self, path: str) -> bool:
        if os.path.exists(f"{path}_bert"):
            self.model = BertForQuestionAnswering.from_pretrained(f"{path}_bert").to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(f"{path}_bert")
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class GPT2LanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set timeout for downloads
        utils.TIMEOUT = 1200
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(1200)

        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, input_text: str) -> str:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
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
        self.model.save_pretrained(f"{path}_gpt2")
        self.tokenizer.save_pretrained(f"{path}_gpt2")

    def load(self, path: str) -> bool:
        if os.path.exists(f"{path}_gpt2"):
            self.model = GPT2LMHeadModel.from_pretrained(f"{path}_gpt2").to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(f"{path}_gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

class RoBERTaLanguageModel(AbstractLanguageModel):
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set timeout for downloads
        utils.TIMEOUT = 500
        requests.adapters.DEFAULT_RETRIES = 5
        requests.DEFAULT_RETRIES = 5
        socket.setdefaulttimeout(500)

        self.model = RobertaForQuestionAnswering.from_pretrained("roberta-base").to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def generate_response(self, input_text: str) -> str:
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
        self.model.save_pretrained(f"{path}_roberta")
        self.tokenizer.save_pretrained(f"{path}_roberta")

    def load(self, path: str) -> bool:
        if os.path.exists(f"{path}_roberta"):
            self.model = RobertaForQuestionAnswering.from_pretrained(f"{path}_roberta").to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(f"{path}_roberta")
            return True
        return False

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
