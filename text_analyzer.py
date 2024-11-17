import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Load spaCy model for subject extraction
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer from nltk
sia = SentimentIntensityAnalyzer()

# Function to rank subjects by meaningfulness (basic heuristic based on common psychological importance)
def rank_subjects(subjects):
    # Heuristic: rank by length and common meaningful words
    importance_order = ["mother", "father", "family", "friend", "child", "dog", "cat", "home", "love"]
    subjects.sort(key=lambda x: (-sum(1 for word in importance_order if word in x.lower()), len(x)), reverse=True)
    return subjects

# Function to rank topics by meaningfulness (basic heuristic based on psychological and global relevance)
def rank_topics(topics):
    meaningful_keywords = ["love", "war", "family", "economy", "health", "freedom", "peace", "life", "happiness"]
    ranked_topics = sorted(topics, key=lambda topic: -sum(1 for word in topic.split(", ") if word in meaningful_keywords))
    return ranked_topics

# Combined function to return sentiment, topics, and ordered subjects
def analyze_text(text, texts_for_topics=[], num_topics=3, num_words=3):
    # Sentiment analysis
    sentiment = sia.polarity_scores(text)['compound']

    # Topic extraction (if given a corpus for context)
    topics = []
    if texts_for_topics:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts_for_topics)
        nmf = NMF(n_components=num_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        words = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf.components_):
            topic_words = [words[i] for i in topic.argsort()[-num_words:]]
            topics.append(", ".join(topic_words))

        # Rank topics
        topics = rank_topics(topics)

    # Subject extraction and ranking
    doc = nlp(text)
    subjects = [chunk.text for chunk in doc.noun_chunks]
    ranked_subjects = rank_subjects(subjects)

    return (sentiment, topics, ranked_subjects)

# Example usage
texts = [
    "My mother loves me dearly, and I love her back.",
    "The war in the world is terrible and painful.",
    "The price of eggs in Siam has gone up dramatically due to shortages."
]

for text in texts:
    sentiment, topics, subjects = analyze_text(text, texts_for_topics=texts)
    print(f"Text: {text}")
    print(f"Sentiment Score: {sentiment}")
    print(f"Topics: {', '.join(topics) if topics else 'N/A'}")
    print(f"Ordered Subjects: {', '.join(subjects)}")
    print()

