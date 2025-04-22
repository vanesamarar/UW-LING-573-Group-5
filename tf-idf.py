# Summarizes review data via Term Frequency-Inverse Document Frequency (TF-IDF)

import os, glob, sys, nltk

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
                    line = clean_line(line.strip())
                    lines.append(line)
            data[topic] = lines
    return data

def clean_line(line):
    """Clean the given line by removing punctuation, etc."""
    # relevant code here
    return line

if __name__ == "__main__":
    data = load_data("data/") # loads data directly from data folder