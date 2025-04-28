import json
import numpy as np
import textstat
from rouge_score import rouge_scorer

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compute_rouge_scores(reference, summary, scorer):
    scores = scorer.score(reference, summary)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def compute_readability(summary):
    return textstat.flesch_reading_ease(summary)

def evaluate_summaries(data, output_file):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_topic_scores = []
    rouge2_topic_scores = []
    rougeL_topic_scores = []
    readability_scores = []

    with open(output_file, 'w') as f:
        for topic_entry in data:
            topic = topic_entry['topic']
            references = " ".join(topic_entry['original_reviews'])
            summaries = topic_entry['summary']

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            readability_summary_scores = []

            for summary in summaries:
                rouge1_f, rouge2_f, rougeL_f = compute_rouge_scores(references, summary, rouge_scorer_instance)
                rouge1_scores.append(rouge1_f)
                rouge2_scores.append(rouge2_f)
                rougeL_scores.append(rougeL_f)
                readability_summary_scores.append(compute_readability(summary))

            rouge1_topic_avg = np.mean(rouge1_scores)
            rouge2_topic_avg = np.mean(rouge2_scores)
            rougeL_topic_avg = np.mean(rougeL_scores)
            readability_topic_avg = np.mean(readability_summary_scores)

            rouge1_topic_scores.append(rouge1_topic_avg)
            rouge2_topic_scores.append(rouge2_topic_avg)
            rougeL_topic_scores.append(rougeL_topic_avg)
            readability_scores.append(readability_topic_avg)

            f.write(f"Topic: {topic}\n")
            f.write(f"  ROUGE-1 Average Score: {rouge1_topic_avg:.4f}\n")
            f.write(f"  ROUGE-2 Average Score: {rouge2_topic_avg:.4f}\n")
            f.write(f"  ROUGE-L Average Score: {rougeL_topic_avg:.4f}\n")
            f.write(f"  Readability (Flesch Reading Ease): {readability_topic_avg:.2f}\n\n")

        final_rouge1_score = np.mean(rouge1_topic_scores)
        final_rouge2_score = np.mean(rouge2_topic_scores)
        final_rougeL_score = np.mean(rougeL_topic_scores)
        final_readability_score = np.mean(readability_scores)

        f.write("Final Average Evaluation Scores Across Topics:\n")
        f.write(f"  ROUGE-1: {round(final_rouge1_score, 4)}\n")
        f.write(f"  ROUGE-2: {round(final_rouge2_score, 4)}\n")
        f.write(f"  ROUGE-L: {round(final_rougeL_score, 4)}\n")
        f.write(f"  Readability (Flesch Reading Ease): {round(final_readability_score, 2)}\n")

def main():
    file_path = 'summaries.json'
    output_file = 'evaluation_results.txt'
    data = load_data(file_path)
    evaluate_summaries(data, output_file)

if __name__ == "__main__":
    main()