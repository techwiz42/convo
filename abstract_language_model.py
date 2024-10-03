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
