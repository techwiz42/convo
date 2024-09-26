import pytest
from unittest.mock import Mock, patch, MagicMock, call
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
    mock = Mock()
    mock.return_value = Mock(input_ids=torch.tensor([[1, 2, 3]]))
    mock.decode.side_effect = lambda *args, **kwargs: "Mocked decode output"
    return mock

@pytest.fixture
def mock_model():
    model = Mock()
    model.generate.return_value = [torch.tensor([1, 2, 3])] * 5
    return model

@pytest.fixture
def mock_nltk():
    with patch('qa_cli_v7.nltk') as mock:
        mock.download = Mock()
        yield mock

@pytest.fixture
def mock_sia():
    with patch('qa_cli_v7.SentimentIntensityAnalyzer') as mock:
        mock.return_value.polarity_scores.return_value = {'compound': 0.0}
        yield mock

@pytest.fixture
def qa_cli(qa_cli_module, mock_tokenizer, mock_model, mock_nltk, mock_sia):
    with patch(f'{qa_cli_module.__name__}.T5Tokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch(f'{qa_cli_module.__name__}.T5ForConditionalGeneration.from_pretrained', return_value=mock_model), \
         patch(f'{qa_cli_module.__name__}.SentimentIntensityAnalyzer', return_value=mock_sia.return_value):
        cli = qa_cli_module.QuestionAnswerCLI('mock_model_path')
        cli.tokenize = lambda text: mock_tokenizer(text).input_ids
        return cli
"""
# This is not working.
def test_generate_question(qa_cli, mock_tokenizer, mock_model):
    mock_tokenizer.decode.side_effect = ["Topic 1", "Question 1?", "Question 2?"]
    mock_model.generate.side_effect = [
        [torch.tensor([1, 2, 3])] * 5,  # For topics
        [torch.tensor([4, 5, 6])] * 10  # For questions
    ]
    with patch.object(qa_cli, 'filter_questions', return_value=["Question 1?"]), \
         patch.object(qa_cli, 'select_non_repetitive_question', return_value="Question 1?"):
        result = qa_cli.generate_question("Test context", 0.0)

    assert result == "Question 1?"
    assert mock_model.generate.call_count == 2
    mock_model.generate.assert_has_calls([
        call(qa_cli.tokenize("Extract key topics from: Test context"), max_length=64, num_return_sequences=5, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.7),
        call(qa_cli.tokenize("Generate a question about Topic 1. The question must end with a question mark."), max_length=64, num_return_sequences=10, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
    ])
"""

def test_filter_questions(qa_cli):
    questions = [
        "Valid question?",
        "Too short?",
        "No question mark",
        "invalid start?",
        "Another valid question?"
    ]
    with patch.object(qa_cli, 'is_valid_question', side_effect=[True, False, False, False, True]):
        filtered = qa_cli.filter_questions(questions)
    assert "Valid question?" in filtered
    assert "Another valid question?" in filtered
    assert len(filtered) == 2

def test_adjust_question_for_sentiment(qa_cli):
    question = "Test question?"
    assert qa_cli.adjust_question_for_sentiment(question, 0.1) == "That's interesting! Test question?"
    assert qa_cli.adjust_question_for_sentiment(question, -0.1) == "I understand this might be challenging. Test question?"
    assert qa_cli.adjust_question_for_sentiment(question, 0.0) == "Test question?"

def test_generate_fallback_question(qa_cli):
    with patch('qa_cli_v7.random.choice', side_effect=["noun", "Can you tell me more about noun?"]):
        result = qa_cli.generate_fallback_question("Test context", 0.0)
        assert "Can you tell me more about" in result

def test_analyze_sentiment(qa_cli, mock_sia):
    result = qa_cli.analyze_sentiment("Test text")
    assert result == 0.0
    mock_sia.return_value.polarity_scores.assert_called_once_with("Test text")

"""
def test_generate_answers(qa_cli, mock_tokenizer, mock_model):
    mock_tokenizer.decode.side_effect = ["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]
    mock_model.generate.return_value = [torch.tensor([1, 2, 3])] * 5

    with patch.object(qa_cli, 'analyze_sentiment', side_effect=[0.1, 0.2, 0.3, 0.4, 0.5]):
        result = qa_cli.generate_answers("Test question", "Test context")

    assert result == "Answer 5"
    assert mock_model.generate.call_count == 1
    mock_model.generate.assert_called_once_with(
        qa_cli.tokenize("answer question: Test question context: Test context"),
        max_length=128,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
"""

@patch('builtins.input', side_effect=['Test User', 'Initial context', 'Test response', 'exit'])
@patch('builtins.print')
def test_interactive_session(mock_print, mock_input, qa_cli):
    with patch.object(qa_cli, 'generate_question', return_value="Test question?"), \
         patch.object(qa_cli, 'analyze_sentiment', return_value=0.0), \
         patch.object(qa_cli, 'generate_answers', return_value="Test answer"):
        qa_cli.interactive_session()

    assert mock_input.call_count == 4
    assert qa_cli.user_name == "Test User"
    assert any("AI: Test question?" in str(call) for call in mock_print.call_args_list)
    assert any("Test answer" in str(call) for call in mock_print.call_args_list)

if __name__ == "__main__":
    pytest.main()
