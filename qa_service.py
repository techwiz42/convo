import argparse
import logging
from transformers import utils
import os
import requests
import socket
from model_implementations import create_model
from enhanced_multi_user_qa_cli import EnhancedMultiUserQuestionAnswerCLI
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
import sys

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress specific loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if isinstance(log_obj, logging.Logger):
        log_obj.setLevel(logging.ERROR)

# Set a longer timeout
utils.TIMEOUT = 1200

# For requests library
requests.adapters.DEFAULT_RETRIES = 5
requests.DEFAULT_RETRIES = 5

# Set socket timeout
socket.setdefaulttimeout(1200)

# Set environment variables to suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ensure correct CUDA device ordering
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first GPU

# Suppress stdout and stderr
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

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
    def create_model_instance(user_id):
        model_paths = {
            "t5": "t5-small",
            "bert": "bert-base-uncased",
            "gpt2": "gpt2",
            "roberta": "roberta-base",
            "flan-t5": "google/flan-t5-small",
            "gpt-j": "EleutherAI/gpt-j-6B"
        }
        
        try:
            return create_model(model_type, user_id, model_paths[model_type])
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to create model {model_type}. Please check the logs for more details.")
    
    return create_model_instance

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-User Fine-Tuned Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5", choices=["t5", "bert", "gpt2", "roberta", "flan-t5", "gpt-j"], help="Name of the model to use")
    args = parser.parse_args()
    print(f"args {args}")
    model_class = model_factory(args.model)

    try:
        qa_cli = EnhancedMultiUserQuestionAnswerCLI(model_class)
        qa_cli.run()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        print("An error occurred. Please check the logs and try again.")

if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception as e:
        print(traceback.format_exc())
