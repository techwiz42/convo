from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

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

    def generate_response(self, input_text: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
    # Assuming you're using a Hugging Face model
        inputs = self.tokenizer(input_text, return_tensors="pt", max_new_tokens=512, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class Conversation:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = ""
        self.previous_questions = set()
        self.last_question = None
