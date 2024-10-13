# test_base_model.py

import unittest
import torch
from transformers import PreTrainedTokenizer
from abstract_model import AbstractLanguageModel

class BaseModelTest(unittest.TestCase):
    model_class = None
    model_path = None

    def setUp(self):
        self.model = self.model_class(self.model_path)

    def test_init(self):
        self.assertIsInstance(self.model, AbstractLanguageModel)
        self.assertIsInstance(self.model.model, torch.nn.Module)
        self.assertIsInstance(self.model.tokenizer, PreTrainedTokenizer)

    def test_generate_response(self):
        input_text = "Hello, how are you?"
        response = self.model.generate_response(input_text)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_fine_tune(self):
        input_text = "What is the capital of France?"
        target_text = "The capital of France is Paris."
        self.model.fine_tune(input_text, target_text)
        # We can't easily test the result of fine-tuning, but we can ensure it doesn't raise an exception

    def test_save_and_load(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "test_model")
            self.model.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            new_model = self.model_class(self.model_path)
            load_success = new_model.load(save_path)
            self.assertTrue(load_success)

    def test_get_tokenizer(self):
        tokenizer = self.model.get_tokenizer()
        self.assertIsInstance(tokenizer, PreTrainedTokenizer)
