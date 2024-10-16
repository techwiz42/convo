# CONVO - A Curious Conversational LLM

CONVO is an advanced, multi-user conversational AI system built on large language models (LLMs). It aims to create a more dynamic and interactive experience by not only answering questions but also generating relevant questions based on the conversation context.

## Project Overview

CONVO implements a PyTorch-based language model using the Hugging Face Transformers library. It supports multiple pre-trained models and can be fine-tuned for specific tasks like question generation and answering.

Key features:
- Multi-user support with concurrent conversations
- User-specific knowledge bases and conversation histories
- Support for multiple language models (T5, GPT-2, BERT, RoBERTa, FLAN-T5, BlenderBot)
- Question generation and answering capabilities
- Sentiment analysis and text quality assessment
- Asynchronous processing for improved performance
- Fine-tuning capabilities for personalized responses

## Project Structure

- `async_qa_service.py`: Main service implementation with asynchronous processing
- `asyncio_multi_user_qa_cli.py`: Asynchronous multi-user CLI interface
- `model_implementations.py`: Abstract and concrete model implementations
- `user_knowledge_base.py`: User-specific knowledge storage and retrieval
- `gpt2_model.py`: Implementation of GPT-2 language model
- `t5_model.py`: Implementation of T5 language model
- `bert_model.py`: Implementation of BERT language model
- `text_analysis.py`: Text analysis utilities for sentiment and grammar checking
- `fine_tune_models.py`: Script for fine-tuning various language models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/convo.git
   cd convo
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers datasets nltk scikit-learn requests tqdm
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('maxent_ne_chunker')
   nltk.download('words')
   nltk.download('vader_lexicon')
   ```

## Usage

### Running the Conversational Service

To start the asynchronous multi-user question-answer service:

```
python async_qa_service.py --model <model_type> --model_path <path_to_model>
```

Replace `<model_type>` with one of: t5, bert, gpt2, roberta, flan-t5, or blenderbot.
Replace `<path_to_model>` with the path to your pre-trained or fine-tuned model.

### Fine-tuning Models

To fine-tune a model for bidirectional question answering:

```
python fine_tune_models.py --model_type <model_type> --train_data <path_to_train_data> --val_data <path_to_val_data> [additional options]
```

Options:
- `--model_type`: Type of model to fine-tune (gpt2, t5, bert, roberta, flan-t5, blenderbot)
- `--model_name`: Specific model name (e.g., 'gpt2-medium', 't5-base')
- `--train_data`: Path to the training data JSON file
- `--val_data`: Path to the validation data JSON file
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate for optimizer (default: 5e-5)
- `--output_dir`: Directory to save the fine-tuned model
- `--gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass (default: 2)
- `--max_length`: Maximum sequence length (default: 128)

The script supports loading and continuing fine-tuning from a previously saved model.

## Features

1. **Multi-User Support**: The system can handle multiple users concurrently, maintaining separate conversation histories and knowledge bases for each user.

2. **Multiple Model Support**: CONVO supports various language models, including T5, BERT, GPT-2, RoBERTa, FLAN-T5, and BlenderBot.

3. **Asynchronous Processing**: The system uses Python's asyncio for improved performance and concurrency.

4. **User-Specific Knowledge Base**: Each user has a personalized knowledge base that grows with their interactions.

5. **Dynamic Question Generation**: The system can generate relevant questions based on the conversation context.

6. **Sentiment Analysis**: Responses are analyzed for sentiment to provide more contextually appropriate answers.

7. **Text Quality Assessment**: The system attempts to filter out nonsensical or low-quality responses using basic grammar checks and named entity recognition.

8. **Fine-tuning**: The models support on-the-fly fine-tuning based on user interactions, allowing for personalized responses over time. Additionally, a separate fine-tuning script is provided for more extensive model customization.

9. **Error Handling**: Robust error handling for CUDA out-of-memory errors, with fallback to CPU processing when necessary.

10. **Flexible Response Generation**: The models use techniques like temperature adjustment and top-p sampling for diverse and contextually relevant responses.

## Model Implementations

The project includes implementations for GPT-2, T5, and BERT models, each with their own specific features and capabilities. These implementations support local model loading, fine-tuning, and flexible response generation.

## Text Analysis

The text analysis module provides basic grammar checking using NLTK, sentiment analysis, and text quality assessment based on grammatical structure and named entity recognition.

## Fine-tuning Script

The `fine_tune_models.py` script provides a flexible way to fine-tune various language models for bidirectional question answering tasks. Key features include:

- Support for multiple model types (GPT-2, T5, BERT, RoBERTa, FLAN-T5, BlenderBot)
- Custom dataset handling for different model architectures
- Gradient accumulation for handling larger batch sizes
- Learning rate scheduling with warm-up
- Validation during training
- Ability to resume training from a saved model
- Comprehensive error handling and logging

## Testing

CONVO includes a comprehensive test suite to ensure the reliability and correctness of its components. The test suite includes both unit tests for individual components and an integration test for the overall system.

### Running Tests

To run the full test suite, use the following command:

```
python -m unittest discover tests
```

For more detailed output, you can use the `-v` flag:

```
python -m unittest discover tests -v
```

### Test Structure

The test suite is organized as follows:

1. **UserKnowledgeBase Tests**: 
   - Test adding knowledge to the knowledge base
   - Test retrieving relevant knowledge

2. **TextAnalyzer Tests**:
   - Test sentiment analysis functionality
   - Test basic grammar checking

3. **ModelImplementations Tests**:
   - Test model creation for different model types (GPT-2, T5)

4. **AsyncEnhancedMultiUserQuestionAnswerCLI Tests**:
   - Test processing user input and generating responses
   - Verify multiple model calls with different parameters

5. **Integration Test**:
   - Test the overall CLI interaction flow
   - Verify CLI start and stop functionality

### Writing New Tests

When adding new features or modifying existing ones, please ensure to update or add corresponding tests. Follow these guidelines:

- Place new test files in the `tests` directory
- Name test files with a `test_` prefix (e.g., `test_new_feature.py`)
- Use descriptive test method names that explain the expected behavior
- Use `unittest.mock.patch` for mocking external dependencies when necessary

### Testing Asynchronous Code

When testing asynchronous functions, we use `asyncio.run()` to execute the coroutines. Here's an example:

```python
import asyncio
from unittest.mock import patch, Mock

class TestAsyncFunction(unittest.TestCase):
    @patch('some_module.some_dependency')
    def test_async_function(self, mock_dependency):
        async def run_test():
            result = await your_async_function()
            self.assertEqual(result, expected_value)

        asyncio.run(run_test())
```

For testing the CLI, which runs indefinitely, we use a technique to start it in a separate task and then stop it after a short delay:

```python
class TestCLI(unittest.TestCase):
    @patch('builtins.input', side_effect=['user1', 'question', 'exit'])
    def test_cli_interaction(self, mock_input):
        async def run_cli():
            await self.cli.run()

        async def run_test():
            cli_task = asyncio.create_task(run_cli())
            await asyncio.sleep(0.1)
            self.cli.stop()
            try:
                await asyncio.wait_for(cli_task, timeout=1.0)
            except asyncio.TimeoutError:
                print("CLI did not stop within the expected time")

        asyncio.run(run_test())
```

This approach ensures that asynchronous functions are properly awaited and executed within the test environment, and that long-running processes like the CLI can be tested effectively without causing the tests to hang indefinitely.

### Debugging Tests

If you encounter issues with the tests, you can add logging statements to both the test files and the main code to help identify the problem. Use the `logging` module to add debug information:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code or tests
logger.debug("Some debug information")
```

## Development Roadmap

- Implement a web interface or multi-client network application to fully utilize the concurrent conversation capabilities.
- Optimize model fine-tuning process for better question generation and personalized responses.
- Implement model compression techniques to reduce the memory footprint of user-specific models.
- Enhance the question filtering mechanism to improve the quality of generated questions.
- Implement more sophisticated conversation flow management.
- Explore integration with other language models and fine-tuning techniques.
- Develop a more streamlined process for data preparation and model evaluation.
- Expand test coverage to include edge cases and additional scenarios
- Implement automated integration tests for different model types
- Set up continuous integration for automated testing on code changes

## Contributing

Contributions to CONVO are welcome! Please feel free to submit a Pull Request.

When contributing, please ensure that your code is well-tested. Add or update tests as necessary to maintain or improve the overall test coverage of the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
