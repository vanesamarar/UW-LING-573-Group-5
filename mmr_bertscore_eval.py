import os
import json
from bert_score import score
import nltk
nltk.download('punkt')

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

def compute_avg_bertscore(mmr_summaries, references):
    preds = []
    refs = []

    for topic, gen_sentences in mmr_summaries.items():
        if topic in references:
            pred = flatten_summary(gen_sentences)
            ref_candidates = references[topic]
            # Compute BERTScore against each reference and take max
            best_ref = max(ref_candidates, key=lambda ref: score([pred], [ref], lang="en", verbose=False)[2].item())
            preds.append(pred)
            refs.append(best_ref)

    if not preds or not refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    P, R, F1 = score(preds, refs, lang="en", verbose=True)

    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }

def main():
    mmr_summary_path = "mmr_results/mmr_summaries.json"
    gold_dir = "summaries-gold"
    output_file = "mmr_results/mmr_bertscore_metrics.json"

    mmr_summaries = load_mmr_summaries(mmr_summary_path)
    reference_summaries = load_references(gold_dir)

    avg_metrics = compute_avg_bertscore(mmr_summaries, reference_summaries)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nAverage BERTScore metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
