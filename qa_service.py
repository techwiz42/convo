import argparse
import logging
from transformers import utils
import requests
import socket
from model_implementations import T5LanguageModel, BERTLanguageModel, GPT2LanguageModel, RoBERTaLanguageModel
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

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-User Fine-Tuned Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5", help="Name of the model to use (t5, bert, gpt2, roberta)")
    args = parser.parse_args()

    model_map = {
        "t5": T5LanguageModel,
        "bert": BERTLanguageModel,
        "gpt2": GPT2LanguageModel,
        "roberta": RoBERTaLanguageModel
    }

    model_class = model_map.get(args.model)
    if model_class is None:
        raise ValueError(f"Unsupported model: {args.model}")

    def model_factory(user_id):
        model_paths = {
            "t5": "model_t5",
            "bert": "model_bert",
            "gpt2": "model_gpt2",
            "roberta": "model_roberta"
        }
        return model_class(user_id, model_paths[args.model])

    try:
        qa_cli = EnhancedMultiUserQuestionAnswerCLI(model_factory)
        qa_cli.run()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print("An error occurred. Please check the logs and try again.")

if __name__ == "__main__":
    main()
