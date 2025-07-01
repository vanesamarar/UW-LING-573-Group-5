# Authors: Nina Koh, Chelsea Kendrick, Vanesa Marar
# Summarizes review data via Term Frequency-Inverse Document Frequency (TF-IDF)

import os, glob, re, nltk, json
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
    """Clean the given line by removing capitalization, punctuation, & multiple whitespaces"""
    line = line.lower() # convert to lowercase
    # Preserve punctuation for fluent summaries
    # line = re.sub(r'[^\w\s]', '', line) # remove punctuation
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
    """Summarizes the top n sentences for each topic using TF-IDF scoring"""
    data = load_data("topics/")
    summaries = {}

    for topic, sentences in data.items():
        print(f"\n--- Summary for topic: {topic} ---")
        
        # Preprocess each sentence
        cleaned_sentences = [clean_line(s) for s in sentences]
        tokenized_sentences = [' '.join(stem_words(tokenize_line(s))) for s in cleaned_sentences]

        # Use TF-IDF to vectorize and score sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_sentences)

        # Compute the sum of TF-IDF scores for each sentence
        sentence_scores = tfidf_matrix.sum(axis=1).flatten().tolist()[0]
        scored_sentences = list(zip(sentence_scores, sentences))

        # Get top n sentences by score
        top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:n]

        summary_sentences = [sentence for score, sentence in top_sentences]

        for sentence in summary_sentences:
            print(f"- {sentence}")

        summaries[topic] = summary_sentences  # Use dictionary format

    with open("tfidf_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    nltk.download('punkt')
    summarize_topic(2) #choose top n sentences (can be changed)
