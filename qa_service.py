import argparse
import logging
from transformers import utils
import requests
import socket
from model_implementations import T5LanguageModel, BERTLanguageModel, GPT2LanguageModel, RoBERTaLanguageModel, FLANT5LanguageModel, GPTJLanguageModel
from enhanced_multi_user_qa_cli import EnhancedMultiUserQuestionAnswerCLI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set a longer timeout
utils.TIMEOUT = 1200

# For requests library
requests.adapters.DEFAULT_RETRIES = 5
requests.DEFAULT_RETRIES = 5

# Set socket timeout
socket.setdefaulttimeout(1200)

def model_factory(model_type):
    def create_model(user_id):
        model_paths = {
            "t5": "model_t5",
            "bert": "model_bert",
            "gpt2": "model_gpt2",
            "roberta": "model_roberta",
            "flan-t5": "model_flan_t5",
            "gpt-j": "model_gpt_j"
        }
        
        if model_type == "t5":
            return T5LanguageModel(user_id, model_paths[model_type])
        elif model_type == "bert":
            return BERTLanguageModel(user_id, model_paths[model_type])
        elif model_type == "gpt2":
            return GPT2LanguageModel(user_id, model_paths[model_type])
        elif model_type == "roberta":
            return RoBERTaLanguageModel(user_id, model_paths[model_type])
        elif model_type == "flan-t5":
            return FLANT5LanguageModel(user_id, model_paths[model_type])
        elif model_type == "gpt-j":
            return GPTJLanguageModel(user_id, model_paths[model_type])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    return create_model

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-User Fine-Tuned Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5", choices=["t5", "bert", "gpt2", "roberta", "flan-t5", "gpt-j"], help="Name of the model to use")
    args = parser.parse_args()

    model_class = model_factory(args.model)
    if model_class is None:
        raise ValueError(f"Unsupported model: {args.model}")

    try:
        qa_cli = EnhancedMultiUserQuestionAnswerCLI(model_class)
        qa_cli.run()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print("An error occurred. Please check the logs and try again.")

if __name__ == "__main__":
    main()
