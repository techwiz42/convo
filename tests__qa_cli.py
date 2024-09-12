import unittest
from unittest.mock import patch, MagicMock
import torch
from qa_cli import QuestionAnswerCLI, train_model, main

class TestQuestionAnswerCLI(unittest.TestCase):
    @patch('qa_cli.T5ForConditionalGeneration')
    @patch('qa_cli.T5Tokenizer')
    def setUp(self, mock_tokenizer, mock_model):
        self.mock_tokenizer = mock_tokenizer.return_value
        self.mock_model = mock_model.return_value
        self.cli = QuestionAnswerCLI("dummy_path")

    def test_init(self):
        self.assertIsInstance(self.cli.device, torch.device)
        self.assertIsNotNone(self.cli.tokenizer)
        self.assertIsNotNone(self.cli.model)

    def test_generate_question(self):
        self.mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
        self.mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
        self.mock_tokenizer.decode.return_value = "Generated question"

        question = self.cli.generate_question("Test context", "Test answer")
        
        self.assertEqual(question, "Generated question")
        self.mock_tokenizer.assert_called_with("generate question: Test context answer: Test answer", return_tensors="pt")
        self.mock_model.generate.assert_called()
        self.mock_tokenizer.decode.assert_called_with(torch.tensor([4, 5, 6]), skip_special_tokens=True)

    @patch('builtins.input', side_effect=['Initial context', 'Answer 1', 'exit'])
    @patch('builtins.print')
    def test_interactive_session(self, mock_print, mock_input):
        self.cli.generate_question = MagicMock(return_value="Generated question")
        
        self.cli.interactive_session()
        
        self.assertEqual(mock_input.call_count, 3)
        self.cli.generate_question.assert_called_once_with('Initial context')
        mock_print.assert_any_call('AI: Generated question')

class TestTrainModel(unittest.TestCase):
    @patch('qa_cli.Dataset')
    @patch('qa_cli.T5ForConditionalGeneration')
    @patch('qa_cli.T5Tokenizer')
    @patch('qa_cli.DataLoader')
    @patch('qa_cli.AdamW')
    @patch('qa_cli.get_linear_schedule_with_warmup')
    def test_train_model(self, mock_scheduler, mock_adamw, mock_dataloader, mock_tokenizer, mock_model, mock_dataset):
        mock_dataset.from_json.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_dataloader.return_value = [{'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([4, 5, 6])}]
        mock_adamw.return_value = MagicMock()
        mock_scheduler.return_value = MagicMock()

        train_model("dummy_data_path", "dummy_output_path")

        mock_dataset.from_json.assert_called_once_with("dummy_data_path")
        mock_tokenizer.assert_called_once_with("t5-small")
        mock_model.assert_called_once_with("t5-small")
        mock_model.return_value.save_pretrained.assert_called_once_with("dummy_output_path")
        mock_tokenizer.return_value.save_pretrained.assert_called_once_with("dummy_output_path")

class TestMain(unittest.TestCase):
    @patch('qa_cli.train_model')
    @patch('qa_cli.QuestionAnswerCLI')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    def test_main_train(self, mock_exists, mock_parse_args, mock_cli, mock_train_model):
        mock_parse_args.return_value = MagicMock(train=True, data="dummy_data", model="dummy_model")
        
        main()
        
        mock_train_model.assert_called_once_with("dummy_data", "dummy_model")
        mock_cli.assert_not_called()

    @patch('qa_cli.train_model')
    @patch('qa_cli.QuestionAnswerCLI')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.path.exists')
    def test_main_interactive(self, mock_exists, mock_parse_args, mock_cli, mock_train_model):
        mock_parse_args.return_value = MagicMock(train=False, model="dummy_model")
        mock_exists.return_value = True
        
        main()
        
        mock_train_model.assert_not_called()
        mock_cli.assert_called_once_with("dummy_model")
        mock_cli.return_value.interactive_session.assert_called_once()

if __name__ == '__main__':
    unittest.main()
