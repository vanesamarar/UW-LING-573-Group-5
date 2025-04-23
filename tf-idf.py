# Authors: Nina Koh, 
# Summarizes review data via Term Frequency-Inverse Document Frequency (TF-IDF)

import os, glob, re, nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {} # maps topics to a list of reviews
    search_path = os.path.join(data_dir, "*.txt.data") # data/*.txt.data
    # For testing on a small patch of data:
    # search_path = os.path.join(data_dir, "price_amazon_kindle.txt.data")
    files = glob.glob(search_path) # files that match search pattern
    for file in files:
        filename = os.path.basename(file) # e.g. price_amazon_kindle.txt.data
        base = os.path.splittext(filename)[0] # e.g. price_amazon_kindle.txt
        topic = os.path.splittext(base)[0] # e.g. price_amazon_kindle
        with open(file, 'r') as f:
            lines = []
            for line in f:
                if line.strip(): # avoid empty strings
                    line = clean_line(line.strip()) # remove whitespace
                    lines.append(line)
            data[topic] = lines
    return data

def clean_line(line):
    """Clean the given line by removing capitalization, punctuation, & multiple whitespaces"""
    line = line.lower() # convert to lowercase
    line = re.sub(r'[^\w\s]', '', line) # remove punctuation
    line = re.sub(r'\s+', ' ', line) # collapse multiple mid-sent whitespaces into one
    return line

def tokenize_line(line):
    """Tokenizes the cleaned line into a list & removes semantically-empty words (e.g., a, the)"""
    tokens = nltk.word_tokenize(line)
    filter_tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return filter_tokens

def stem_words(tokens):
    """Stems the given tokens via an English-specific subclass (e.g., generously->generous)"""
    stemmer = SnowballStemmer("english") # instance of english-specific stemmer
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def summarize_topic(n):
    """Summarizes the top n sentences"""

if __name__ == "__main__":
    data = load_data("data/") # loads data directly from data folder
    # tokens = tokenize_line(line)