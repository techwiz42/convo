# test_flan_t5_model.py

import unittest
from test_base_model import BaseModelTest
from flan_t5_model import FLANT5LanguageModel

class TestFLANT5Model(BaseModelTest):
    model_class = FLANT5LanguageModel
    model_path = "google/flan-t5-small"  # You can change this to a different FLAN-T5 model if needed

if __name__ == '__main__':
    unittest.main()
