# Author: Nina Koh
# Selects review data via the Maximal Marginal Relevance (MMR) algorithm

import os, json, glob, re, nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {} # maps topics to a list of reviews
    search_path = os.path.join(data_dir, "*.txt.data") # data/*.txt.data
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

def select_mmr(tokenized_sentences, summary_size=7, lambda_param=0.7):
    """Selects reviews using the Maximal Marginal Relevance (MMR) algorithm"""
    # Check if number of reviews doesn't fall short of summary size
    if len(tokenized_sentences) < summary_size:
        print(f"Not enough sentences to summarize. Found {len(tokenized_sentences)}, expected at least {summary_size}.")
        return []
    
    # Use TF-IDF to vectorize sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_sentences)
    
    # Compute cosine similarities across all reviews pairs
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Construct a (synthetic) centroid vector
    centroid = np.mean(tfidf_matrix.toarray(), axis=0)
    centroid = centroid.reshape(1, -1) # convert to 2D array
    
    # Compute relevance scores for each sentence (compared to the centroid)
    relevance = cosine_similarity(tfidf_matrix, centroid).flatten() # convert back to 1D array
    
    # Initialize tracking variables 
    selected = [] # to store indices of selected reviews
    remaining = list(range(len(tokenized_sentences))) # indices of remaining reviews (initially all)
    
    # Iteratively select reviews based on MMR
    while len(selected) < summary_size and remaining:
        # Compute MMR scores for remaining reviews
        mmr_scores = []
        for idx in remaining: # iterate through candidate reviews
            # Identify highest similarity score between candidates & selected reviews
            # to serve as redundancy penalty
            redundancy = np.max(sim_matrix[idx, selected]) if selected else 0
            # MMR = λ * relevance - (1 - λ) * redundancy
            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * redundancy
            mmr_scores.append(mmr_score)
        
        # Select the review with the highest MMR score
        max_mmr_index = np.argmax(mmr_scores)
        selected.append(remaining[max_mmr_index])
        remaining.remove(remaining[max_mmr_index])
        
    # Return the list of reviews represented by indices in selected
    return [tokenized_sentences[i] for i in selected]

if __name__ == "__main__":
    data = load_data("topics/")
    summaries = {}

    for topic, sentences in data.items():        
        # Preprocess each sentence
        cleaned_sentences = [clean_line(s) for s in sentences]
        tokenized_sentences = [' '.join(stem_words(tokenize_line(s))) for s in cleaned_sentences]

        # Run MMR on preprocessed reviews to select a summary
        # summary = select_mmr(tokenized_sentences, summary_size=7, lambda_param=0.7)
        summary = select_mmr(tokenized_sentences, summary_size=2, lambda_param=0.7)

        # Map preprocessed review back to original version (avoid aggressive stemming in output)
        indices = [tokenized_sentences.index(s) for s in summary if s in tokenized_sentences]
        summary = [sentences[i] for i in indices] 
        summaries[topic] = summary
        
    # with open('mmr_results/mmr_ante-hoc.json', 'w', encoding='utf-8') as f:
    #     json.dump(summaries, f, indent=2, ensure_ascii=False)
    
    with open('mmr_results/mmr_summaries.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
        