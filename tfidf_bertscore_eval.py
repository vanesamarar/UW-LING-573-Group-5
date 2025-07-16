import os
import json
from bert_score import score

def load_tfidf_summaries(summary_path):
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
            references[topic_dir] = " ".join(gold_texts)
    return references


def flatten_summary(summary_sentences):
    return " ".join(summary_sentences)

def compute_bertscore(tfidf_summaries, references):
    topics = list(tfidf_summaries.keys())
    preds = []
    refs = []
    valid_topics = []

    for topic in topics:
        if topic in references:
            pred = flatten_summary(tfidf_summaries[topic])
            ref = references[topic]
            preds.append(pred)
            refs.append(ref)
            valid_topics.append(topic)

    P, R, F1 = score(preds, refs, lang="en", verbose=True)
    
    avg_metrics = {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }

    per_topic_metrics = {
        topic: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f)
        }
        for topic, p, r, f in zip(valid_topics, P, R, F1)
    }

    return avg_metrics

def main():
    tfidf_summary_path = "tf-idf_results/tfidf_ante-hoc.json"
    data_dir = "data/"
    gold_dir = "summaries-gold"
    output_file = "tf-idf_results/tfidf_bertscore_metrics.json"

    tfidf_summaries = load_tfidf_summaries(tfidf_summary_path)
    reference_summaries = load_references(gold_dir)

    result = compute_bertscore(tfidf_summaries, reference_summaries)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nBERTScore average metrics:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

