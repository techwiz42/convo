# test_bert_model.py

import unittest
from test_base_model import BaseModelTest
from bert_model import BERTLanguageModel

class TestBERTModel(BaseModelTest):
    model_class = BERTLanguageModel
    model_path = "bert-base-uncased"  # You can change this to a different BERT model if needed

if __name__ == '__main__':
    unittest.main()
