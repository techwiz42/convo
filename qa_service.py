# File: qa_service.py

import asyncio
import argparse
import logging
from transformers import utils
import os
import warnings
import traceback

from async_multi_user_qa import AsyncMultiUserQA
from model_implementations import create_model

# Keep the existing configurations (logging, warnings, etc.)

class AsyncModelFactory:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model_paths = {
            "t5": "t5-small",
            "bert": "bert-base-uncased",
            "gpt2": "gpt2",
            "roberta": "roberta-base",
            "flan-t5": "google/flan-t5-small",
            "gpt-j": "EleutherAI/gpt-j-6B"
        }

    async def create_model_instance(self, user_id):
        try:
            return await asyncio.to_thread(
                create_model,
                self.model_type,
                user_id,
                self.model_paths[self.model_type]
            )
        except Exception as e:
            logger.error(f"Error creating model {self.model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to create model {self.model_type}. Please check the logs for more details.")

async def main():
    parser = argparse.ArgumentParser(description="Async Multi-User Fine-Tuned Question-Answer CLI Application")
    parser.add_argument("--model", type=str, default="t5", choices=["t5", "bert", "gpt2", "roberta", "flan-t5", "gpt-j"], help="Name of the model to use")
    args = parser.parse_args()
    print(f"args {args}")
    model_factory = AsyncModelFactory(args.model)

    try:
        qa_system = AsyncMultiUserQA(model_factory)
        await qa_system.run()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        print("An error occurred. Please check the logs and try again.")

if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asyncio.run(main())
    except Exception as e:
        print(traceback.format_exc())
