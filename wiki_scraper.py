import wikipedia
import json
import random
import nltk
from nltk.tokenize import sent_tokenize
import argparse
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def get_random_wikipedia_article():
    """Fetch a random Wikipedia article."""
    while True:
        try:
            title = wikipedia.random(1)
            page = wikipedia.page(title)
            return page.title, page.content
        except wikipedia.exceptions.DisambiguationError as e:
            # If we get a disambiguation page, try again
            continue
        except wikipedia.exceptions.PageError:
            # If the page doesn't exist, try again
            continue

def generate_qa_pair(context):
    """Generate a simple question-answer pair from the given context."""
    sentences = sent_tokenize(context)
    if len(sentences) < 2:
        return None, None, None
    
    answer_sentence = random.choice(sentences)
    words = answer_sentence.split()
    if len(words) < 4:
        return None, None, None
    
    answer = " ".join(words[:3])  # Use the first three words as the answer
    question = f"What comes after '{' '.join(words[:2])}'?"
    
    return context, question, answer

def create_dataset(num_articles, existing_data):
    """Create a dataset with the specified number of articles."""
    dataset = existing_data
    existing_titles = set(item.get('title', '') for item in dataset)
    articles_added = 0

    while articles_added < num_articles:
        title, content = get_random_wikipedia_article()
        if title in existing_titles:
            continue  # Skip if we already have this article

        context, question, answer = generate_qa_pair(content)
        if context and question and answer:
            dataset.append({
                "title": title,
                "context": context,
                "question": question,
                "answer": answer
            })
            existing_titles.add(title)
            articles_added += 1
        print(f"Processed article: {title}")
    
    return dataset

def load_existing_data(filename):
    """Load existing data from a JSON file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('data', [])
        except json.JSONDecodeError:
            print(f"Error decoding existing file {filename}. Starting with empty dataset.")
    return []

def main():
    parser = argparse.ArgumentParser(description="Generate a QA dataset from Wikipedia articles")
    parser.add_argument("--num_articles", type=int, default=100, help="Number of articles to fetch")
    parser.add_argument("--output", type=str, default="wikipedia_dataset.json", help="Output JSON file name")
    args = parser.parse_args()

    existing_data = load_existing_data(args.output)
    print(f"Loaded {len(existing_data)} existing articles.")

    print(f"Fetching {args.num_articles} new Wikipedia articles...")
    dataset = create_dataset(args.num_articles, existing_data)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"data": dataset}, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset with {len(dataset)} total QA pairs saved to {args.output}")

if __name__ == "__main__":
    main()
