# test_t5_model.py

import unittest
from test_base_model import BaseModelTest
from t5_model import T5LanguageModel

class TestT5Model(BaseModelTest):
    model_class = T5LanguageModel
    model_path = "t5-small"  # You can change this to a different T5 model if needed

if __name__ == '__main__':
    unittest.main()
