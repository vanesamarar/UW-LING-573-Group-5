import os
import json
from rouge_score import rouge_scorer

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

def evaluate_summary(gold_summaries, generated_summaries):
    """Evaluate the TF-IDF summaries using ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = {}

    for topic, generated_summary in generated_summaries.items():
        gold_summary = gold_summaries.get(topic, [])

        if not gold_summary:
            continue

        recall_scores = []
        precision_scores = []
        f1_scores = []

        for g_summary in gold_summary:
            score = scorer.score(g_summary, ' '.join(generated_summary))

            # Extract the Recall, Precision, and F1 for each ROUGE metric
            recall_scores.append({
                'rouge1': score['rouge1'].recall,
                'rouge2': score['rouge2'].recall,
                'rougeL': score['rougeL'].recall,
            })
            precision_scores.append({
                'rouge1': score['rouge1'].precision,
                'rouge2': score['rouge2'].precision,
                'rougeL': score['rougeL'].precision,
            })
            f1_scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure,
            })

        # Aggregate the scores by calculating the average Recall, Precision, and F1
        avg_recall = {
            'rouge1': sum([s['rouge1'] for s in recall_scores]) / len(recall_scores),
            'rouge2': sum([s['rouge2'] for s in recall_scores]) / len(recall_scores),
            'rougeL': sum([s['rougeL'] for s in recall_scores]) / len(recall_scores),
        }

        avg_precision = {
            'rouge1': sum([s['rouge1'] for s in precision_scores]) / len(precision_scores),
            'rouge2': sum([s['rouge2'] for s in precision_scores]) / len(precision_scores),
            'rougeL': sum([s['rougeL'] for s in precision_scores]) / len(precision_scores),
        }

        avg_f1 = {
            'rouge1': sum([s['rouge1'] for s in f1_scores]) / len(f1_scores),
            'rouge2': sum([s['rouge2'] for s in f1_scores]) / len(f1_scores),
            'rougeL': sum([s['rougeL'] for s in f1_scores]) / len(f1_scores),
        }

        # Store the average recall, precision, and f1 scores for each topic
        results[topic] = {
            'recall': avg_recall,
            'precision': avg_precision,
            'f1': avg_f1
        }

    return results

def print_rouge_results(results):
    """Prints the average ROUGE scores for each metric."""
    avg_rouge1 = sum([result['recall']['rouge1'] for result in results.values()]) / len(results)
    avg_rouge2 = sum([result['recall']['rouge2'] for result in results.values()]) / len(results)
    avg_rougeL = sum([result['recall']['rougeL'] for result in results.values()]) / len(results)

    avg_precision_rouge1 = sum([result['precision']['rouge1'] for result in results.values()]) / len(results)
    avg_precision_rouge2 = sum([result['precision']['rouge2'] for result in results.values()]) / len(results)
    avg_precision_rougeL = sum([result['precision']['rougeL'] for result in results.values()]) / len(results)

    avg_f1_rouge1 = sum([result['f1']['rouge1'] for result in results.values()]) / len(results)
    avg_f1_rouge2 = sum([result['f1']['rouge2'] for result in results.values()]) / len(results)
    avg_f1_rougeL = sum([result['f1']['rougeL'] for result in results.values()]) / len(results)

    print("Average ROUGE-1 Recall: {:.4f}".format(avg_rouge1))
    print("Average ROUGE-2 Recall: {:.4f}".format(avg_rouge2))
    print("Average ROUGE-L Recall: {:.4f}".format(avg_rougeL))
    
    print("Average ROUGE-1 Precision: {:.4f}".format(avg_precision_rouge1))
    print("Average ROUGE-2 Precision: {:.4f}".format(avg_precision_rouge2))
    print("Average ROUGE-L Precision: {:.4f}".format(avg_precision_rougeL))

    print("Average ROUGE-1 F1: {:.4f}".format(avg_f1_rouge1))
    print("Average ROUGE-2 F1: {:.4f}".format(avg_f1_rouge2))
    print("Average ROUGE-L F1: {:.4f}".format(avg_f1_rougeL))

if __name__ == "__main__":
    gold_dir = "summaries-gold"  
    summary_file = "summaries.json"   
    
    gold_summaries = load_gold_summaries(gold_dir)
    generated_summaries = load_generated_summaries(summary_file)

    # Evaluate the generated summaries
    results = evaluate_summary(gold_summaries, generated_summaries)
    
    # Print the ROUGE results
    print_rouge_results(results)

