import json
import argparse
import sys

def process_coqa_data(input_file, output_file):
    # Read the input CoQA JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Process each data point
    processed_data = []
    for item in data['data']:
        processed_item = {
            'context': item['story'],
            'questions': [turn['input_text'] for turn in item['questions']]
        }
        processed_data.append(processed_item)
        
        # Print progress indicator
        print(f"\n{item['source']}: ", end='', flush=True)
        for _ in processed_item['questions']:
            print('.', end='', flush=True)
    
    print()  # Final newline after processing all items

    # Write the processed data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"\nProcessed data has been saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process CoQA dataset into simplified format.")
    parser.add_argument("input_file", help="Path to the input CoQA dataset JSON file")
    parser.add_argument("output_file", help="Path to save the processed JSON file")
    
    # Parse arguments
    args = parser.parse_args()

    # Process the data
    process_coqa_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
