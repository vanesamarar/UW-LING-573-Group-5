# Author: Nina Koh, Vanesa Marar
# This is a script to get quantitative informtion on the gold summaries

# Comment on Deliverable 2: How long are the human-written summaries, on average?
import os

def get_paths(gold_dir):
    """Returns a list of paths to all .gold summary files"""
    paths = []
    for topic in os.listdir(gold_dir): # generate list of topics (folders)
        topic_path = os.path.join(gold_dir, topic)
        if os.path.isdir(topic_path):
            for file in os.listdir(topic_path):
                if file.endswith(".gold"):
                    full_path = os.path.join(topic_path, file)
                    paths.append(full_path)
    return paths

def analyze_summaries():
    """Prints average word count across all gold summaries"""
    paths = get_paths(gold_dir)
    total_words = 0
    total_summaries = 0

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            summary = f.read().strip()
            word_count = len(summary.split())
            total_words += word_count
            total_summaries += 1

    if total_summaries == 0:
        print("No summaries were found.")
    else:
        avg_length = total_words / total_summaries
        print(f"Total summaries: {total_summaries}")
        print(f"Average summary length: {avg_length:.2f} words")
    
    
if __name__ == "__main__":
    root_dir = "." 
    gold_dir = os.path.join(root_dir, "summaries-gold")
    analyze_summaries(gold_dir)
