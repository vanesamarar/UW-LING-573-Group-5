from pythonrouge.pythonrouge import Pythonrouge
import os, json

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

def evaluate_rouge_su4(gold_summaries, generated_summaries):
    """Evaluates ROUGE-SU4 using the Pythonrouge package."""
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=generated_summaries,
                        reference=gold_summaries,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=False, stemming=True, stopwords=True,
                        word_level=True, length_limit=False, length=0,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=False, samples=0, favor=True, p=0.5)

    score = rouge.calc_score()

    composite_su4 = 0.5 * score['ROUGE-1-F'] + 0.5 * score['ROUGE-SU4-F']
    print(f"COMPOSITE-SU4-F: {composite_su4:.4f}")

    return score

def print_rouge_results(results):
    for metric, score in results.items():
        print(f"{metric.upper()}: {score:.4f}")

if __name__ == "__main__":
    # Set the paths for gold summaries and generated summaries
    gold_dir = "summaries-gold"  # Path to gold summaries
    summary_file = "summaries.json"  # Path to generated summaries
    
    # Load the summaries
    gold_summaries = load_gold_summaries(gold_dir)
    generated_summaries = load_generated_summaries(summary_file)
    
    results = evaluate_rouge_su4(gold_summaries, generated_summaries)
    
    print_rouge_results(results)

