import sys
import os
import pytest
from unittest.mock import patch, MagicMock, call
import logging
import io
from contextlib import redirect_stdout, redirect_stderr

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import qa_service

@pytest.fixture
def mock_argparse():
    with patch('qa_service.argparse.ArgumentParser') as mock:
        mock_args = MagicMock()
        mock_args.model = 't5'
        mock.return_value.parse_args.return_value = mock_args
        yield mock

@pytest.fixture
def mock_enhanced_multi_user_qa_cli():
    with patch('qa_service.EnhancedMultiUserQuestionAnswerCLI') as mock:
        yield mock

@pytest.fixture
def mock_create_model():
    with patch('qa_service.create_model') as mock:
        yield mock

def test_logging_configuration():
    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Call the configure_logging function
    with patch('logging.basicConfig') as mock_basicConfig:
        qa_service.configure_logging()
        mock_basicConfig.assert_called_once_with(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def test_transformers_timeout():
    with patch('qa_service.utils') as mock_utils:
        qa_service.configure_transformers()
        assert mock_utils.TIMEOUT == 1200

def test_socket_timeout():
    with patch('qa_service.socket') as mock_socket:
        qa_service.configure_socket()
        mock_socket.setdefaulttimeout.assert_called_once_with(1200)

def test_environment_variables():
    with patch.dict('os.environ', {}, clear=True):
        qa_service.configure_environment()
        assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3'
        assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'

@patch('qa_service.requests')
def test_requests_configuration(mock_requests):
    qa_service.configure_requests()
    assert mock_requests.adapters.DEFAULT_RETRIES == 5
    assert mock_requests.DEFAULT_RETRIES == 5

@patch('qa_service.socket')
def test_socket_timeout(mock_socket):
    qa_service.configure_socket()
    mock_socket.setdefaulttimeout.assert_called_once_with(1200)


def test_dummy_file():
    dummy_file = qa_service.DummyFile()
    dummy_file.write("test")
    dummy_file.flush()

def test_suppress_output():
    @qa_service.suppress_output
    def test_func():
        print("This should be suppressed")
        print("This should also be suppressed", file=sys.stderr)

    # Capture stdout and stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    with redirect_stdout(stdout), redirect_stderr(stderr):
        test_func()
    
    # Check that nothing was printed to stdout or stderr
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""

def test_logging_configuration():
    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Call the configure_logging function
    with patch('logging.basicConfig') as mock_basicConfig:
        qa_service.configure_logging()
        mock_basicConfig.assert_called_once_with(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )



