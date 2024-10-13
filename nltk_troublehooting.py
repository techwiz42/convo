import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Check NLTK version
print(f"NLTK version: {nltk.__version__}")

# Attempt to download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Test NLTK functionality
try:
    from nltk.corpus import words
    word_list = words.words()
    print(f"Successfully loaded {len(word_list)} words")
except Exception as e:
    print(f"Error loading NLTK corpus: {str(e)}")

# If the above fails, try manual corpus loading
try:
    from nltk.corpus.reader import WordListCorpusReader
    from nltk.data import find
    word_list = WordListCorpusReader(find('corpora/words'), ['en'])
    print(f"Successfully loaded {len(word_list.words())} words using manual method")
except Exception as e:
    print(f"Error in manual corpus loading: {str(e)}")

