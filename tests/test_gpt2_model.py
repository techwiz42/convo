# test_gpt2_model.py

import unittest
from test_base_model import BaseModelTest
from gpt2_model import GPT2LanguageModel

class TestGPT2Model(BaseModelTest):
    model_class = GPT2LanguageModel
    model_path = "gpt2"  # You can change this to a different GPT-2 model if needed

if __name__ == '__main__':
    unittest.main()

