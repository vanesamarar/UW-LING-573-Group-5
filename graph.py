import os, glob, re, nltk, json
import networkx as nx
from nltk.stem.snowball import SnowballStemmer

def load_data(data_dir):
    """Loads .txt.data files from the Opinosis dataset into a dictionary"""
    data = {}
    search_path = os.path.join(data_dir, "*.txt.data")
    files = glob.glob(search_path)
    for file in files:
        filename = os.path.basename(file)
        base = os.path.splitext(filename)[0]
        topic = os.path.splitext(base)[0]
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for line in f:
                if line.strip():
                    line = clean_line(line.strip())
                    lines.append(line)
            data[topic] = lines
    return data

def clean_line(line):
    """Clean the given line by removing capitalization, punctuation, & multiple whitespaces"""
    line = line.lower()
    line = re.sub(r'[^\w\s]', '', line)
    line = re.sub(r'\s+', ' ', line)
    return line

def tokenize_line(line):
    """Tokenizes the cleaned line into a list of tokens"""
    tokens = nltk.word_tokenize(line)
    return tokens

def build_word_graph(sentences):
    """Constructs a directed graph where nodes are words and edges represent word order"""
    G = nx.DiGraph()
    for sentence in sentences:
        tokens = tokenize_line(sentence)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
    return G

def extract_summary_paths(G, min_length=7):
    """Finds good summary paths from the word graph"""
    summaries = []
    nodes = list(G.nodes())
    
    for start_node in nodes:
        for end_node in nodes:
            if start_node != end_node:
                # Find all simple paths from start to end, up to reasonable length
                for path in nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=min_length+3):
                    if len(path) >= min_length:
                        summaries.append(path)
    return summaries

def score_path(G, path):
    """Scores a path based on cumulative edge weights (frequency of transitions)"""
    score = 0
    for i in range(len(path) - 1):
        score += G[path[i]][path[i+1]]['weight']
    return score

def summarize_topic(n):
    """Summarizes the top n paths for each topic using a graph-based approach"""
    data = load_data("data/")
    summaries = []

    for topic, sentences in data.items():
        print(f"\n--- Graph-Based Summary for topic: {topic} ---")

        G = build_word_graph(sentences)
        paths = extract_summary_paths(G)

        scored_paths = [(score_path(G, path), path) for path in paths]
        top_paths = sorted(scored_paths, key=lambda x: x[0], reverse=True)[:n]

        summary_sentences = [' '.join(path) for score, path in top_paths]

        for sentence in summary_sentences:
            print(f"- {sentence}")

        summaries.append({
            "topic": topic,
            "original_reviews": sentences,
            "summary": summary_sentences
        })

    with open("graph_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    nltk.download('punkt')
    summarize_topic(1)  # choose top 3 summaries per topic (can be changed)

