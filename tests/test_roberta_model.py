# test_roberta_model.py

import unittest
from test_base_model import BaseModelTest
from roberta_model import RoBERTaLanguageModel

class TestRoBERTaModel(BaseModelTest):
    model_class = RoBERTaLanguageModel
    model_path = "roberta-base"  # You can change this to a different RoBERTa model if needed

if __name__ == '__main__':
    unittest.main()
