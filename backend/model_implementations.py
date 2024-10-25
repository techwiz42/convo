import os
import torch
from transformers import PreTrainedTokenizer
from abc import ABC, abstractmethod

# Import individual model classes
from abstract_model import AbstractLanguageModel
from gpt2_model import GPT2LanguageModel
from t5_model import T5LanguageModel
from bert_model import BERTLanguageModel
from roberta_model import RoBERTaLanguageModel
from flan_t5_model import FLANT5LanguageModel

class Conversation:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = ""
        self.previous_questions = set()
        self.last_question = None

def get_model_class(model_type: str):
    model_classes = {
        'gpt2': GPT2LanguageModel,
        't5': T5LanguageModel,
        'bert': BERTLanguageModel,
        'roberta': RoBERTaLanguageModel,
        'flan-t5': FLANT5LanguageModel
    }
    return model_classes.get(model_type)

def create_model(model_type: str, model_path) -> AbstractLanguageModel:
    ModelClass = get_model_class(model_type)
    if ModelClass is None:
        raise ValueError(f"Unsupported model: {model_type}")
    return ModelClass(model_type, model_path)

# Example usage
if __name__ == "__main__":
    model = create_model('gpt2', './models')
    response = model.generate_response("Hello, how are you?")
    print(response)
