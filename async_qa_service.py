import asyncio
import argparse
import logging
from transformers import utils
import os
import requests
import socket
from model_implementations import create_model
from asyncio_multi_user_qa_cli import AsyncEnhancedMultiUserQuestionAnswerCLI
import traceback
import warnings
import sys
from contextlib import contextmanager
from text_analysis import TextAnalyzer

def configure_logging():
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_transformers():
    utils.TIMEOUT = 1200

def configure_requests():
    requests.adapters.DEFAULT_RETRIES = 5
    requests.DEFAULT_RETRIES = 5

def configure_socket():
    socket.setdefaulttimeout(1200)

def configure_environment():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ensure correct CUDA device ordering
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first GPU

# Call configuration functions at module level
configure_logging()
configure_transformers()
configure_requests()
configure_socket()
configure_environment()

logger = logging.getLogger()

# Suppress stdout and stderr
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def suppress_output(func):
    def wrapper(*args, **kwargs):
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = save_stdout
            sys.stderr = save_stderr
    return wrapper

@suppress_output
def model_factory(model_type):
    def create_model_instance():
        model_paths = {
            "t5": "t5-small",
            "bert": "bert-base-uncased",
            "gpt2": "gpt2",
            "roberta": "roberta-base",
            "flan-t5": "google/flan-t5-small",
            "gpt-j": "EleutherAI/gpt-j-6B"
        }
        
        try:
            model_path = "./models" 
            return create_model(model_type, model_path)
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to create model {model_type}. Please check the logs for more details.")
    
    return create_model_instance

async def create_shared_model(model_type):
    model_paths = {
        "t5": "t5-small",
        "bert": "bert-base-uncased",
        "gpt2": "gpt2",
        "roberta": "roberta-base",
        "flan-t5": "google/flan-t5-small",
        "gpt-j": "EleutherAI/gpt-j-6B"
    }
    
    #model_dir = os.environ.get("MODEL_DIR", "models")
    #model_path = os.path.join(model_dir, model_paths[model_type])
    model_path = "/home/scooter/projects/convo/models" #FIXME
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is downloaded and placed in the correct directory.")
    
    try:
        return create_model(model_type, model_path)
    except Exception as e:
        logger.error(f"Error creating model {model_type}: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to create model {model_type}. Please check the logs for more details.")

async def main(args):
    try:
        shared_model = create_model(args.model, args.model_path)
        text_analyzer = TextAnalyzer()
        qa_cli = AsyncEnhancedMultiUserQuestionAnswerCLI(shared_model)
        await qa_cli.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AsyncEnhancedMultiUserQuestionAnswerCLI")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    args = parser.parse_args()

    asyncio.run(main(args))
