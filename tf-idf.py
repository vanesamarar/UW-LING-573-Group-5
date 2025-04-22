# Authors: Nina Koh, 
# Summarizes review data via Term Frequency-Inverse Document Frequency (TF-IDF)

import os, glob, re, nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {} # maps topics to a list of reviews
    search_path = os.path.join(data_dir, "*.txt.data") # data/*.txt.data
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
    """Tokenizes the cleaned line & removes semantically-empty words (e.g., a, the)"""
    tokens = nltk.word_tokenize(line)
    # remove stop words here
    return tokens

if __name__ == "__main__":
    data = load_data("data/") # loads data directly from data folder