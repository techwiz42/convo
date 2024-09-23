import wikipedia
import json
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import argparse
import os
import logging
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_wikipedia_article():
    """Fetch a random Wikipedia article."""
    while True:
        try:
            title = wikipedia.random(1)
            page = wikipedia.page(title)
            return page.title, page.content
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            continue

def extract_entity(sentence):
    """Extract a named entity from the sentence."""
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    chunked = ne_chunk(tagged)
    
    for subtree in chunked:
        if type(subtree) == nltk.Tree:
            entity_type = subtree.label()
            entity = " ".join([word for word, tag in subtree.leaves()])
            return entity_type, entity
    
    return None, None

def generate_question_from_sentence(sentence):
    """Generate a question from a given sentence using various methods."""
    entity_type, entity = extract_entity(sentence)
    
    if entity:
        if entity_type == 'PERSON':
            return f"Who is {entity}?", entity
        elif entity_type in ['GPE', 'LOCATION']:
            return f"Where is {entity} located?", entity
        elif entity_type == 'ORGANIZATION':
            return f"What is {entity}?", entity
    
    # If no named entity is found, try to generate a question based on the sentence structure
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    
    # Look for a subject-verb pattern
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith('NN'):  # Noun
            for j, (verb, vtag) in enumerate(tagged[i+1:], start=i+1):
                if vtag.startswith('VB'):  # Verb
                    subject = " ".join(word for word, _ in tagged[:j])
                    predicate = " ".join(word for word, _ in tagged[j:])
                    return f"What {predicate}?", subject
    
    # If no pattern is found, use a generic question
    return f"What does the following sentence mean: '{sentence}'?", sentence

def generate_qa_pair(context):
    """Generate a question-answer pair from the given context."""
    sentences = sent_tokenize(context)
    if len(sentences) < 2:
        return None, None, None
    
    for _ in range(10):  # Try up to 10 times to generate a question
        sentence = random.choice(sentences)
        question, answer = generate_question_from_sentence(sentence)
        
        # Ensure the question is not too short and the answer is in the context
        if len(question.split()) > 3 and answer.lower() in context.lower():
            return context, question, answer
    
    return None, None, None

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
            logging.info(f"Processed article: {title}")
            logging.info(f"Generated question: {question}")
            logging.info(f"Answer: {answer}")
            logging.info("---")
    
    return dataset

def load_existing_data(filename):
    """Load existing data from a JSON file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('data', [])
        except json.JSONDecodeError:
            logging.error(f"Error decoding existing file {filename}. Starting with empty dataset.")
    return []

def main():
    parser = argparse.ArgumentParser(description="Generate a QA dataset from Wikipedia articles")
    parser.add_argument("--num_articles", type=int, default=100, help="Number of articles to fetch")
    parser.add_argument("--output", type=str, default="wikipedia_dataset.json", help="Output JSON file name")
    args = parser.parse_args()

    existing_data = load_existing_data(args.output)
    logging.info(f"Loaded {len(existing_data)} existing articles.")

    logging.info(f"Fetching {args.num_articles} new Wikipedia articles...")
    dataset = create_dataset(args.num_articles, existing_data)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"data": dataset}, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Dataset with {len(dataset)} total QA pairs saved to {args.output}")

if __name__ == "__main__":
    main()
