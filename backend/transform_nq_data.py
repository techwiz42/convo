import json
import random

def transform_nq_data(input_file, output_train_file, output_val_file, val_ratio=0.1):
    transformed_data = []

    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            context = item['document_html']  # Changed from 'document_text' to 'document_html'
            
            qa_pairs = []
            for annotation in item['annotations']:  # Changed from 'question_answer_pairs' to 'annotations'
                question = item.get('question_text', 'No question provided')  # Assuming question_text is at the top level
                
                # Use the first short answer if available, otherwise use long answer or "No answer"
                if annotation['short_answers']:
                    answer = annotation['short_answers'][0]
                    answer_text = context[answer['start_byte']:answer['end_byte']]
                    qa_pairs.append({"question": question, "answer": answer_text})
                if annotation['yes_no_answer'] != 'NONE':
                    answer = annotation['yes_no_answer']
                    qa_pairs.append({"question": question, "answer": answer})
                if annotation['long_answer']:
                    long_answer = annotation['long_answer']
                    answer_text = context[long_answer['start_byte']:long_answer['end_byte']]
                    qa_pairs.append({"question": question, "answer": answer_text})
            
            transformed_item = {
                "context": context,
                "question_answer_pairs": qa_pairs
            }
            
            transformed_data.append(transformed_item)
            print('.', end='', flush = True)
    # Shuffle the data
    random.shuffle(transformed_data)

    # Split the data
    split_index = int(len(transformed_data) * (1 - val_ratio))
    train_data = transformed_data[:split_index]
    val_data = transformed_data[split_index:]

    # Write training data
    with open(output_train_file, 'w') as f:
        json.dump(train_data, f, indent=2)

    # Write validation data
    with open(output_val_file, 'w') as f:
        json.dump(val_data, f, indent=2)

if __name__ == "__main__":
    transform_nq_data('./data/data_nq.jsonl', './data/train_nq.json', './data/val_nq.json', val_ratio=0.1)
