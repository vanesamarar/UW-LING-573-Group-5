# Author: Nina Koh
# Summarizes review data via the LexRank algorithm

import os, json, glob, re
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {} # maps topics to a list of reviews
    search_path = os.path.join(data_dir, "*.txt.data") # data/*.txt.data
    # For testing on a small patch of data:
    # search_path = os.path.join(data_dir, "price_amazon_kindle.txt.data")
    files = glob.glob(search_path) # files that match search pattern
    for file in files:
        filename = os.path.basename(file) # e.g. price_amazon_kindle.txt.data
        base = os.path.splitext(filename)[0] # e.g. price_amazon_kindle.txt
        topic = os.path.splitext(base)[0] # e.g. price_amazon_kindle
        with open(file, 'r', encoding='utf-8', errors='ignore') as f: #ignore non-unicode characters
            lines = []
            for line in f:
                if line.strip(): # avoid empty strings
                    line = clean_line(line.strip()) # remove whitespace
                    lines.append(line)
            data[topic] = lines
    return data

def clean_line(line):
    """Clean the given line by removing capitalization & multiple whitespaces"""
    line = line.lower() # convert to lowercase
    line = re.sub(r'\s+', ' ', line) # collapse multiple mid-sent whitespaces into one
    return line

def summarize_reviews(reviews, lexrank):
    """Summarizes reviews using the LexRank algorithm"""
    # return lexrank.get_summary(reviews, summary_size=7, threshold=0.1)
    return lexrank.get_summary(reviews, summary_size=2, threshold=0.1)

if __name__ == "__main__":
    data = load_data("topics/")
    # Create a corpus of reviews
    corpus = []
    for reviews in data.values(): 
        for review in reviews:
            corpus.append(review)
    
    # Initialize the LexRank instance
    lexrank = LexRank([corpus], stopwords=STOPWORDS['en'])

    summaries = {}
    for topic, reviews in data.items():
        summary = summarize_reviews(reviews, lexrank)
        summaries[topic] = summary
        
    # with open('lexrank_ante-hoc.json', 'w', encoding='utf-8') as f:
    #     json.dump(summaries, f, indent=2, ensure_ascii=False)
    
    with open('lexrank_summaries.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)