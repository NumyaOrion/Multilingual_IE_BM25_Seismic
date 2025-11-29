import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------------------------------------------
languages = {
    "English": r"mMARCO/Seismic_results/English_seismic_results.trec",
    "German": r"mMARCO/Seismic_results/German_seismic_results.trec",
    "Italian": r"mMARCO/Seismic_results/Italian_seismic_results.trec",
    "Japanese": r"mMARCO/Seismic_results/Japanese_seismic_results.trec",
    "Russian": r"mMARCO/Seismic_results/Russian_seismic_results.trec"
}
reljud_path = r"mMARCO/Dataset_RelevanceJudgements_qrels.dev.tsv"
# ---------------------------------------------------------------------------------------------------------------

# Load relevance judgments (TSV: query_id, _, doc_id, relevance)
qrels = pd.read_csv(reljud_path, sep='\t', header=None, names=['query', 'zero', 'doc_id', 'rel'])
qrels = qrels[qrels['rel'] > 0]  # Only relevant docs
qrels['query'] = qrels['query'].astype(str)
qrels['doc_id'] = qrels['doc_id'].astype(str)

# Build relevance dict: query -> set of relevant docs
relevance_dict = defaultdict(set)
for _, row in qrels.iterrows():
    relevance_dict[row['query']].add(row['doc_id'])

# Memory-efficient metrics computation
def compute_metrics(result_file):
    from collections import defaultdict

    results = defaultdict(list)

    # Read results line by line (no huge DataFrame)
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            results[query_id].append(doc_id)

    mrr_total = 0
    accuracy_total = 0
    recall_total = 0
    num_queries = len(results)

    for query_id, docs in results.items():
        top_10 = docs[:10]
        top_1000 = docs[:1000]
        relevant_docs = relevance_dict.get(query_id, set())

        # Accuracy@10
        if any(doc in relevant_docs for doc in top_10):
            accuracy_total += 1

        # MRR@10
        mrr_score = 0
        for rank, doc in enumerate(top_10, start=1):
            if doc in relevant_docs:
                mrr_score = 1 / rank
                break
        mrr_total += mrr_score

        # Recall@1000
        if any(doc in relevant_docs for doc in top_1000):
            recall_total += 1

    return mrr_total / num_queries, accuracy_total / num_queries, recall_total / num_queries

# Compute and print metrics
for lang, path in languages.items():
    mrr, acc, r1k = compute_metrics(path)
    print(f"{lang}: MRR@10 = {mrr:.4f}, Accuracy@10 = {acc:.4f}, Recall@1000 = {r1k:.4f}")
