import os
import json
import nltk
from bert_score import score

def load_lexrank_summaries(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)  # {topic: [list of summary sentences]}

def load_gold_summaries(gold_dir):
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
    return references

def flatten_summary(sentences):
    return " ".join(sentences)

def compute_avg_bertscore(lexrank_summaries, references):
    preds = []
    refs = []

    for topic, summary_sentences in lexrank_summaries.items():
        if topic in references:
            pred = flatten_summary(summary_sentences)
            golds = references[topic]
            # Choose the best gold reference (max score)
            max_score = 0
            best_gold = None
            for ref in golds:
                P, R, F1 = score([pred], [ref], lang="en", verbose=False)
                if F1[0] > max_score:
                    max_score = F1[0]
                    best_gold = ref
            if best_gold:
                preds.append(pred)
                refs.append(best_gold)

    if preds and refs:
        P, R, F1 = score(preds, refs, lang="en", verbose=True)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    else:
        return {"precision": 0, "recall": 0, "f1": 0}

def main():
    nltk.download('punkt')

    lexrank_summary_path = "lexrank_results/lexrank_summaries.json"
    gold_dir = "summaries-gold"
    output_file = "lexrank_results/lexrank_bertscore_metrics.json"

    lexrank_summaries = load_lexrank_summaries(lexrank_summary_path)
    reference_summaries = load_gold_summaries(gold_dir)

    avg_metrics = compute_avg_bertscore(lexrank_summaries, reference_summaries)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nBERTScore metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
