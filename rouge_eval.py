import os
import json
from rouge_score import rouge_scorer

# Function to load gold summaries
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

# Function to load TF-IDF summaries (assumed structure of the summaries)
def load_generated_summaries(summary_file):
    """Loads the generated summaries from the TF-IDF summary JSON file."""
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to evaluate summaries using ROUGE scores
def evaluate_summary(gold_summaries, generated_summaries):
    """Evaluate the TF-IDF summaries using ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {}

    for topic, generated_summary in generated_summaries.items():
        gold_summary = gold_summaries.get(topic, [])
        
        # If no gold summary for this topic, skip
        if not gold_summary:
            continue

        # Compare the generated summary with each gold summary
        scores = []
        for g_summary in gold_summary:
            score = scorer.score(g_summary, ' '.join(generated_summary))
            scores.append(score)

        # Aggregate scores (e.g., average F-measure)
        avg_scores = {
            'rouge1': sum([s['rouge1'].fmeasure for s in scores]) / len(scores),
            'rouge2': sum([s['rouge2'].fmeasure for s in scores]) / len(scores),
            'rougeL': sum([s['rougeL'].fmeasure for s in scores]) / len(scores),
        }
        results[topic] = avg_scores

    return results

# Function to print and summarize the evaluation results
def print_rouge_results(results):
    """Prints out the average ROUGE results."""
    avg_rouge1 = sum([result['rouge1'] for result in results.values()]) / len(results)
    avg_rouge2 = sum([result['rouge2'] for result in results.values()]) / len(results)
    avg_rougeL = sum([result['rougeL'] for result in results.values()]) / len(results)

    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")

# Main code to load data, evaluate, and print results
if __name__ == "__main__":
    # Paths to your data
    gold_dir = "summaries-gold"  # Update with the correct path to the gold summaries
    summary_file = "summaries.json"      # Update with the generated summaries (TF-IDF)
    
    # Load gold summaries and generated summaries
    gold_summaries = load_gold_summaries(gold_dir)
    generated_summaries = load_generated_summaries(summary_file)

    # Evaluate the generated summaries
    results = evaluate_summary(gold_summaries, generated_summaries)
    
    # Print the ROUGE results
    print_rouge_results(results)

