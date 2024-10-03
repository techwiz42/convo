import json
import random
from datasets import load_dataset
from tqdm import tqdm
import os

def process_squad(examples, num_samples):
    processed = []
    for example in tqdm(examples, desc="Processing SQuAD"):
        try:
            context = example['context']
            for question, answer in zip(example['question'], example['answers']['text']):
                processed.append({
                    'context': context,
                    'questions': [question],
                    'answers': [answer[0] if isinstance(answer, list) else answer]
                })
            if len(processed) >= num_samples:
                break
        except Exception as e:
            print(f"Error processing SQuAD example: {str(e)}")
    return processed[:num_samples]

def process_natural_questions(examples, num_samples):
    processed = []
    for example in tqdm(examples, desc="Processing Natural Questions"):
        try:
            if 'document' not in example or 'tokens' not in example['document']:
                continue
            context = ' '.join(example['document']['tokens']['token'])
            question = example['question']['text']
            if example['annotations'] and example['annotations']['short_answers']:
                answer_start = example['annotations']['short_answers'][0]['start_token']
                answer_end = example['annotations']['short_answers'][0]['end_token']
                answer = ' '.join(example['document']['tokens']['token'][answer_start:answer_end])
                processed.append({
                    'context': context,
                    'questions': [question],
                    'answers': [answer]
                })
            if len(processed) >= num_samples:
                break
        except Exception as e:
            print(f"Error processing Natural Questions example: {str(e)}")
    return processed[:num_samples]

def process_quac(examples, num_samples):
    processed = []
    for example in tqdm(examples, desc="Processing QuAC"):
        try:
            context = example['background']
            for question, answer in zip(example['questions'], example['answers']):
                if answer['text'] != 'CANNOTANSWER':
                    processed.append({
                        'context': context,
                        'questions': [question['text']],
                        'answers': [answer['text']]
                    })
                if len(processed) >= num_samples:
                    break
            if len(processed) >= num_samples:
                break
        except Exception as e:
            print(f"Error processing QuAC example: {str(e)}")
    return processed[:num_samples]

def process_race(examples, num_samples):
    processed = []
    for example in tqdm(examples, desc="Processing RACE"):
        try:
            context = example['article']
            for question, answer in zip(example['questions'], example['answers']):
                processed.append({
                    'context': context,
                    'questions': [question],
                    'answers': [answer]
                })
            if len(processed) >= num_samples:
                break
        except Exception as e:
            print(f"Error processing RACE example: {str(e)}")
    return processed[:num_samples]

def safe_load_dataset(dataset_name, split, **kwargs):
    try:
        return load_dataset(dataset_name, split=split, **kwargs)
    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        return None

def main():
    # Prepare training set
    train_data = []
    train_samples = 40000  # Total number of training samples

    # SQuAD
    squad = safe_load_dataset("squad", split="train")
    if squad:
        train_data.extend(process_squad(squad, train_samples))

    # Shuffle the training data
    random.shuffle(train_data)

    # Save training data
    with open('train_data.json', 'w') as f:
        json.dump(train_data, f)

    print(f"Training set created with {len(train_data)} examples")

    # Prepare validation set
    val_data = []
    val_samples_per_dataset = 1000

    # Prepare validation set
    val_data = []
    val_samples = 3000  # Total number of validation samples

    # SQuAD for validation
    squad_val = safe_load_dataset("squad", split="validation")
    if squad_val:
        val_data.extend(process_squad(squad_val, val_samples))

    # Shuffle the validation data
    random.shuffle(val_data)

    # Save validation data
    with open('val_data.json', 'w') as f:
        json.dump(val_data, f)

    print(f"Validation set created with {len(val_data)} examples")

if __name__ == "__main__":
    main()
