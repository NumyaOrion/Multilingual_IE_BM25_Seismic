import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------------------------------------------
languages = {
    "English": r"mMARCO/BM25_results/English_Results/English_Results.txt",
    "German": r"mMARCO/BM25_results/German_Results/German_Results.txt",
    "Italian": r"mMARCO/BM25_results/Italian_Results/Italian_Results.txt",
    "Japanese": r"mMARCO/BM25_results/Japanese_Results/Japanese_Results.txt",
    "Russian": r"mMARCO/BM25_results/Russian_Results/Russian_Results.txt"
}
reljud_path = r"mMARCO/Dataset_RelevanceJudgements_qrels.dev.tsv"
# ---------------------------------------------------------------------------------------------------------------

# Load relevance judgments (TSV: query_id, _, doc_id, relevance)
qrels = pd.read_csv(reljud_path, sep='\t', header=None, names=['query', 'zero', 'doc_id', 'rel'])
relevance_dict = {str(row['query']): str(row['doc_id']) for _, row in qrels.iterrows()}

# Compute MRR@10, Accuracy@10, Recall@1000
def compute_metrics(result_file):
    mrr_total = 0
    accuracy_total = 0
    recall_total = 0
    num_queries = 0

    # Read BM25 results
    with open(result_file, 'r') as f:
        lines = f.readlines()

    # Organize results by query
    results = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        query_id = parts[0]
        doc_id = parts[2]
        results[query_id].append(doc_id)

    # Compute metrics
    for query_id, retrieved_docs in results.items():
        num_queries += 1
        top_10 = retrieved_docs[:10]
        top_1000 = retrieved_docs[:1000]

        relevant_doc = relevance_dict.get(query_id)

        # Accuracy@10
        if relevant_doc in top_10:
            accuracy_total += 1

        # MRR@10
        if relevant_doc in top_10:
            rank = top_10.index(relevant_doc) + 1
            mrr_total += 1 / rank

        # Recall@1000
        if relevant_doc in top_1000:
            recall_total += 1

    mrr = mrr_total / num_queries
    accuracy = accuracy_total / num_queries
    recall = recall_total / num_queries

    return mrr, accuracy, recall

# Compute and print metrics for all languages
for lang, path in languages.items():
    mrr, acc, r1k = compute_metrics(path)
    print(f"{lang}: MRR@10 = {mrr:.4f}, Accuracy@10 = {acc:.4f}, Recall@1000 = {r1k:.4f}")
