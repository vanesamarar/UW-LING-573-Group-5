import os
import json
from nltk.translate.meteor_score import single_meteor_score
import nltk

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_mmr_summaries(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)  # {topic: [list of summary sentences]}

def load_references(gold_dir):
    references = {}
    for topic_dir in os.listdir(gold_dir):
        topic_path = os.path.join(gold_dir, topic_dir)
        if os.path.isdir(topic_path):
            gold_texts = []
            for gold_file in os.listdir(topic_path):
                if gold_file.endswith(".gold"):
                    with open(os.path.join(topic_path, gold_file), "r", encoding="utf-8") as f:
                        gold_texts.append(f.read().strip())
            if gold_texts:
                references[topic_dir] = gold_texts
    return references  # {topic: [list of gold refs]}

def flatten_summary(summary_sentences):
    return " ".join(summary_sentences)

def compute_avg_meteor(mmr_summaries, references):
    all_scores = []

    for topic, gen_sentences in mmr_summaries.items():
        if topic in references:
            pred = flatten_summary(gen_sentences)
            refs = references[topic]
            try:
                score = max(
                    single_meteor_score(nltk.word_tokenize(ref), nltk.word_tokenize(pred))
                    for ref in refs
                ) if refs else 0
            except Exception as e:
                print(f"Error scoring topic {topic}: {e}")
                score = 0
            all_scores.append(score)

    return sum(all_scores) / len(all_scores) if all_scores else 0

def main():
    mmr_summary_path = "mmr_results/mmr_summaries.json"
    gold_dir = "summaries-gold"
    output_file = "mmr_results/mmr_meteor_metrics.json"

    mmr_summaries = load_mmr_summaries(mmr_summary_path)
    reference_summaries = load_references(gold_dir)

    avg_score = compute_avg_meteor(mmr_summaries, reference_summaries)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"average_meteor": avg_score}, f, indent=2, ensure_ascii=False)

    print(f"\nAverage METEOR score saved to: {output_file}")

if __name__ == "__main__":
    main()
