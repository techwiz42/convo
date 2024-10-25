import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abstract_model import AbstractLanguageModel
import traceback
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

class FLANT5LanguageModel(AbstractLanguageModel):
    def __init__(self, model_type: str, model_path: str):
        self.model_path = os.path.join(model_path, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                print("Loaded locally trained model")
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
                print("Falling back to default FLAN-T5 model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        else:
            print(f"Local model not found at {self.model_path}. Loading default FLAN-T5 model.")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Ensure required NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
    
    def is_question(self, text: str) -> bool:
        try:
            # First check for simple question mark
            if "?" in text:
                return True
            
            # Tokenize and get POS tags
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Check if sentence starts with WH-words or auxiliary verbs
            if pos_tags:
                first_tag = pos_tags[0][1]
                first_word = pos_tags[0][0].lower()
                
                # WH-words
                if first_tag.startswith('W'):
                    return True
                
                # Auxiliary verbs at the start
                aux_verbs = {'is', 'are', 'was', 'were', 'do', 'does', 'did',
                           'have', 'has', 'had', 'can', 'could', 'will', 'would',
                           'shall', 'should', 'may', 'might', 'must'}
                
                if first_word in aux_verbs:
                    # Check if subject follows the auxiliary verb
                    if len(pos_tags) > 1 and pos_tags[1][1].startswith(('NN', 'PRP')):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error in is_question: {str(e)}")
            return "?" in text  # Fallback to simple check if NLTK fails

    def generate_response(self, input_text: str, context: str = "", previous_input: str = "", temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 200, min_new_tokens: int = 100) -> str:
        try:
            # Use NLTK to determine if input is a question
            is_question = self.is_question(input_text)

            if is_question:
                # If input is a question, generate a statement
                formatted_input = f"Instruction: Convert this question into a statement: {input_text}"
            else:
                # If input is a statement, generate a question
                formatted_input = f"Instruction: Generate a relevant question about: {input_text}"

            # Add context if provided
            if context:
                formatted_input = f"Context: {context} {formatted_input}"
            if previous_input:
                formatted_input = f"Previous input: {previous_input} {formatted_input}"

            # Tokenize formatted input
            input_ids = self.tokenizer(formatted_input, return_tensors="pt").input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=False
                )
            
            # Decode the generated tokens, ignoring the input
            generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Ensure the response ends with appropriate punctuation
            if is_question:
                if not generated_text.rstrip().endswith(('.', '!', '...')):
                    generated_text = generated_text.rstrip() + '.'
            else:
                if not generated_text.rstrip().endswith('?'):
                    generated_text = generated_text.rstrip() + '?'
            
            return generated_text

        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())

    def generate_response_OLD(self, input_text: str, context: str = "", previous_input: str = "", temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 200, min_new_tokens: int = 100) -> str:
        try:
            # Determine the type of response to generate
            if context and previous_input:
                return self.generate_dialogue_continuation(context, previous_input, temperature, top_p, max_new_tokens, min_new_tokens)
            elif context:
                formatted_input = f"Context: {context} Instruction: {input_text}"
            elif previous_input:
                formatted_input = f"Previous input: {previous_input} Instruction: {input_text}"
            else:
                formatted_input = f"Instruction: {input_text}"

            # Tokenize formatted input
            input_ids = self.tokenizer(formatted_input, return_tensors="pt").input_ids.to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=False
                )
            
            # Decode the generated tokens, ignoring the input
            generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            return generated_text
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            return "An error occurred while generating the response. Please try again."

    def generate_dialogue_continuation(self, context: str, previous_input: str, temperature: float, top_p: float, max_new_tokens: int, min_new_tokens: int) -> str:
        try:
            # Prepare input for dialogue continuation
            formatted_input = f"Continue the following conversation:\nContext: {context}\nHuman: {previous_input}\nAI:"

            # Tokenize formatted input
            input_ids = self.tokenizer(formatted_input, return_tensors="pt").input_ids.to(self.device)
            
            # Generate dialogue continuation
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=False
                )
            
            # Decode the generated tokens, ignoring the input
            generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            return generated_text
        except Exception as e:
            print(f"Error in generate_dialogue_continuation: {str(e)}")
            print(traceback.format_exc())
            return "An error occurred while generating the dialogue continuation. Please try again."
    def format_input_with_context(self, input_text: str, context: str) -> str:
        if context:
            return f"Given the previous interaction: {context}\n\nRespond to: {input_text}"
        else:
            return f"Respond to: {input_text}"

    def fine_tune(self, input_text: str, target_text: str):
        try:
            if "An error occurred" not in target_text and len(target_text.split()) > 5:
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)
                targets = self.tokenizer(target_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)

                outputs = self.model(**inputs, labels=targets.input_ids)
                loss = outputs.loss
        
                if loss is not None:
                    loss.backward()
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
                    optimizer.step()
                    print(f"Fine-tuning completed. Loss: {loss.item()}")
                else:
                    print("Warning: Loss is None. Skipping fine-tuning.")
            else:
                print("Skipping fine-tuning due to invalid target text.")
        except Exception as e:
            print(f"Error in fine_tune: {str(e)}")
            print(traceback.format_exc())

    def save(self, path: str):
        try:
            print(f"Saving model to {path}")
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print(traceback.format_exc())

    def load(self, path: str) -> bool:
        try:
            if os.path.exists(path):
                print(f"Loading model from {path}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                print("Model loaded successfully")
                return True
            else:
                print(f"No saved model found at {path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_tokenizer(self):
        return self.tokenizer
