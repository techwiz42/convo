import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        grammar_score = self.basic_grammar_check(text)
        sentiment_score = self.analyze_sentiment(text)
        return grammar_score, sentiment_score

    def basic_grammar_check(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        ne_tree = ne_chunk(tagged)
        
        tag_counts = Counter(tag for word, tag in tagged)
        
        has_noun = any(tag.startswith('NN') for word, tag in tagged)
        has_verb = any(tag.startswith('VB') for word, tag in tagged)
        has_determiner = 'DT' in tag_counts
        has_proper_capitalization = text[0].isupper()
        has_named_entity = any(isinstance(chunk, nltk.Tree) for chunk in ne_tree)
        
        score = 0
        if has_noun: score += 0.20
        if has_verb: score += 0.20
        if has_determiner: score += 0.20
        if has_proper_capitalization: score += 0.20
        if has_named_entity: score += 0.20
        
        return score

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)['compound']
