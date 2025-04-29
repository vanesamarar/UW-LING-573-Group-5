import os
import json
from rouge_metric import PyRouge

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

def evaluate_with_rouge_metric(gold_summaries, generated_summaries):
    """Evaluate using rouge-metric including SU4."""
    hypotheses = []
    references = []

    for topic, generated_summary in generated_summaries.items():
        gold_summary = gold_summaries.get(topic, [])
        if not gold_summary:
            continue
        
        hypotheses.append(' '.join(generated_summary))
        references.append(gold_summary)

    rouge = PyRouge(
        rouge_n=(1, 2, 4),
        rouge_l=True,
        rouge_w=True,
        rouge_s=True,
        rouge_su=True,
        skip_gap=4
    )

    scores = rouge.evaluate(hypotheses, references)
    return scores

def print_rouge_scores(scores):
    for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l', 'rouge-su4']:
        print(f"{rouge_type.upper()} Recall:    {scores[rouge_type]['r']:.4f}")
        print(f"{rouge_type.upper()} Precision: {scores[rouge_type]['p']:.4f}")
        print(f"{rouge_type.upper()} F1 Score:  {scores[rouge_type]['f']:.4f}")
        print('')

if __name__ == "__main__":
    gold_dir = "summaries-gold"  
    summary_file = "summaries.json"   
    
    gold_summaries = load_gold_summaries(gold_dir)
    generated_summaries = load_generated_summaries(summary_file)

    results = evaluate_with_rouge_metric(gold_summaries, generated_summaries)
    print_rouge_scores(results)
