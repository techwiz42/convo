import pytest
from unittest.mock import Mock, patch, mock_open
import torch
import sys
import os
import importlib

@pytest.fixture(scope="session")
def qa_cli_version(request):
    return request.config.getoption("--qa-cli-version")

@pytest.fixture(scope="session")
def qa_cli_module(qa_cli_version):
    module_name = f'qa_cli_{qa_cli_version}'
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, parent_dir)
    
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pytest.fail(f"Could not import {module_name}. Make sure the file exists and is named correctly.")

@pytest.fixture
def mock_tokenizer():
    return Mock()

@pytest.fixture
def mock_model():
    model = Mock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model

@pytest.fixture
def qa_cli(qa_cli_module, mock_tokenizer, mock_model):
    with patch(f'{qa_cli_module.__name__}.T5Tokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch(f'{qa_cli_module.__name__}.T5ForConditionalGeneration.from_pretrained', return_value=mock_model):
        return qa_cli_module.QuestionAnswerCLI('mock_model_path')

def test_generate_question(qa_cli, mock_tokenizer, mock_model):
    mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
    mock_tokenizer.decode.return_value = "Generated question"

    result = qa_cli.generate_question("Test context", "Test answer")

    assert result == "Generated question"
    mock_tokenizer.assert_called_once_with("generate question: Test context answer: Test answer", return_tensors="pt", max_length=512, truncation=True)
    mock_model.generate.assert_called_once()

def test_train_model(qa_cli_module):
    with patch(f'{qa_cli_module.__name__}.torch.device') as mock_device, \
         patch(f'{qa_cli_module.__name__}.T5Tokenizer.from_pretrained') as mock_tokenizer, \
         patch(f'{qa_cli_module.__name__}.T5ForConditionalGeneration.from_pretrained') as mock_model, \
         patch(f'{qa_cli_module.__name__}.Dataset.from_dict') as mock_dataset, \
         patch(f'{qa_cli_module.__name__}.DataLoader') as mock_dataloader, \
         patch(f'{qa_cli_module.__name__}.AdamW') as mock_adamw, \
         patch(f'{qa_cli_module.__name__}.get_linear_schedule_with_warmup') as mock_scheduler:

        mock_device.return_value = 'cpu'
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_dataset.return_value = Mock()
        mock_dataloader.return_value = [{'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]]), 'labels': torch.tensor([[4, 5, 6]])}]
        mock_adamw.return_value = Mock()
        mock_scheduler.return_value = Mock()

        with patch('builtins.open', mock_open(read_data='{"data": [{"context": "test", "question": "test?", "answer": "test"}]}')):
            qa_cli_module.train_model('mock_data.json', 'mock_output', num_epochs=1)

        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        mock_dataset.assert_called_once()
        mock_dataloader.assert_called_once()
        mock_adamw.assert_called_once()
        mock_scheduler.assert_called_once()

def test_preprocess_function(qa_cli_module):
    examples = {
        "context": ["Test context 1", "Test context 2"],
        "answer": ["Test answer 1", "Test answer 2"],
        "question": ["Test question 1?", "Test question 2?"]
    }
    
    with patch(f'{qa_cli_module.__name__}.tokenizer') as mock_tokenizer:
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tokenizer.as_target_tokenizer.return_value.__enter__.return_value = mock_tokenizer

        result = qa_cli_module.preprocess_function(examples)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert len(result["input_ids"]) == 2
        assert len(result["attention_mask"]) == 2
        assert len(result["labels"]) == 2
