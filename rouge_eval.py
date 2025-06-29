# Authors: Nina Koh, Vanesa Marar, Chelsea Kendrick
# Evaluates generated summaries against gold standard using ROUGE-1, ROUGE-2, and ROUGE-SU4
# via Hugging Face's rouge.rouge_scorer

import os
import json
# from rouge_metric import PyRouge
from rouge import rouge_scorer # Hugging Face implementation
import re
from nltk.stem import PorterStemmer

def load_gold_summaries(gold_dir):
    """Loads the gold summaries from the 'summaries-gold' directory."""
    gold_summaries = {}
    for topic_dir in os.listdir(gold_dir):
        topic_path = os.path.join(gold_dir, topic_dir)
        if os.path.isdir(topic_path):
            gold_summaries[topic_dir] = []
            for gold_file in os.listdir(topic_path):
                if gold_file.endswith('.gold'):
                    with open(os.path.join(topic_path, gold_file), 'r', encoding='utf-8') as f:
                        gold_summaries[topic_dir].append(f.read().strip())
    return gold_summaries

def load_generated_summaries(summary_file):
    """Loads the generated summaries from the TF-IDF summary JSON file."""
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# What happens if I use less aggressive preprocessing?
# stemmer = PorterStemmer()
# def preprocess(text):
#     """Preprocessing: lowercase, remove punctuation, stem words."""
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#     tokens = text.split()
#     stemmed = [stemmer.stem(t) for t in tokens]
#     return " ".join(stemmed)

def evaluate_with_rouge_metric(gold_summaries, generated_summaries):
    """Evaluate via Hugging Face's rouge_scorer."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    # Initialize nested scores dictionary
    total_scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }
    topic_count = 0
    
    for topic, gold_list in gold_summaries.items():
        if topic not in generated_summaries:
            continue
        
        generated_summary = generated_summaries[topic]
        if not generated_summary:
            continue
        
        # For now, get the first reference summary (may be modified later)
        gold_summary = gold_list[0]
        
        # Calculate ROUGE scores
        score = scorer.score(gold_summary, ' '.join(generated_summary))
        for rouge_type in total_scores.keys():
            total_scores[rouge_type]['precision'] += score[rouge_type].precision
            total_scores[rouge_type]['recall'] += score[rouge_type].recall
            total_scores[rouge_type]['fmeasure'] += score[rouge_type].fmeasure
        topic_count += 1
        
    # Average the scores   
    for rouge_type in total_scores.keys():
        total_scores[rouge_type]['precision'] /= topic_count
        total_scores[rouge_type]['recall'] /= topic_count
        total_scores[rouge_type]['fmeasure'] /= topic_count
    return total_scores
    
    # Old PyRouge implementation:
    # hypotheses = []
    # references = []
    # for topic, generated_summary in generated_summaries.items():
    #     gold_summary = gold_summaries.get(topic, [])
    #     if not gold_summary:
    #         continue
    #     hyp = preprocess(' '.join(generated_summary))
    #     refs = [preprocess(ref) for ref in gold_summary]
    #     hypotheses.append(hyp)
    #     references.append(refs)
    # rouge = PyRouge(
    #     rouge_n=(1, 2, 4),
    #     rouge_l=True,
    #     rouge_w=True,
    #     rouge_s=True,
    #     rouge_su=True,
    #     skip_gap=4
    # )
    # scores = rouge.evaluate(hypotheses, references)

def print_rouge_scores(scores):
    for rouge_type in scores.keys():
        print(f"{rouge_type.upper()} Recall:    {scores[rouge_type]['recall']:.4f}")
        print(f"{rouge_type.upper()} Precision: {scores[rouge_type]['precision']:.4f}")
        print(f"{rouge_type.upper()} F1 Score:  {scores[rouge_type]['fmeasure']:.4f}")
        print('')

if __name__ == "__main__":
    gold_dir = "summaries-gold"  
    # summary_file = "tf-idf_results/tfidf_summaries.json"   
    summary_file = "t5-small_results/t5_zero_shot_summaries.json" 
    
    gold_summaries = load_gold_summaries(gold_dir)
    generated_summaries = load_generated_summaries(summary_file)

    scores = evaluate_with_rouge_metric(gold_summaries, generated_summaries)
    print_rouge_scores(scores)
