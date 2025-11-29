import json
import time
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
model_name = "xlm-roberta-base"
BATCH_SIZE = 32
TOP_K = 64
MAX_LEN = 512

input_path = "mMARCO/Dataset_Json/Russian/mMARCO_collection_converted-russian.jsonl"
output_path = "mMARCO/Dataset_Json/Russian/mMARCO_russian_vectors_multi_unicoil_xlmr_topk.jsonl"

# -----------------------------------------------------------
# Model setup
# -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("Running on device:", device)

# -----------------------------------------------------------
# Functions
# -----------------------------------------------------------
def compute_unicoil_sparse_batch(texts):
    """Compute UniCOIL sparse vectors for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_weights = outputs.last_hidden_state.mean(dim=2)

    id_lists = inputs["input_ids"].cpu().tolist()
    token_lists = [tokenizer.convert_ids_to_tokens(ids) for ids in id_lists]

    results = []
    for tokens, weights in zip(token_lists, token_weights.cpu()):
        weights = weights.tolist()

        real_tokens = []
        real_weights = []
        for tok, w in zip(tokens, weights):
            if tok != tokenizer.pad_token:
                real_tokens.append(tok)
                real_weights.append(w)

        # Keep top-K
        if TOP_K and len(real_weights) > TOP_K:
            idxs = torch.topk(torch.tensor(real_weights), TOP_K).indices.tolist()
        else:
            idxs = list(range(len(real_weights)))

        sparse = {real_tokens[i]: float(real_weights[i]) for i in idxs}
        results.append(sparse)

    return results

def count_lines(path):
    """Count total lines for progress bar."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def read_in_batches_skip_processed(path, batch_size, processed_ids):
    """Yield batches of docs, skipping already processed IDs."""
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["id"] in processed_ids:
                continue
            batch.append(doc)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# -----------------------------------------------------------
# Resume support
# -----------------------------------------------------------
processed_ids = set()
if os.path.exists(output_path):
    print("Resuming from existing output file...")
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            processed_ids.add(doc["id"])

total_docs = count_lines(input_path)
remaining_docs = total_docs - len(processed_ids)
print(f"Total documents: {total_docs}, Already processed: {len(processed_ids)}, Remaining: {remaining_docs}")

# -----------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------
start_time = time.time()
processed = len(processed_ids)

with open(output_path, "a", encoding="utf-8") as fout:
    with tqdm(total=total_docs, desc="Processing", unit="doc") as pbar:
        pbar.update(processed)  # Skip already processed

        for batch in read_in_batches_skip_processed(input_path, BATCH_SIZE, processed_ids):
            texts = [d["contents"] for d in batch]
            sparse_vectors = compute_unicoil_sparse_batch(texts)

            for doc, vec in zip(batch, sparse_vectors):
                doc["vector"] = vec
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                processed_ids.add(doc["id"])

            # Update counters and progress
            processed += len(batch)
            elapsed = time.time() - start_time
            speed = processed / max(elapsed, 1e-9)
            pbar.set_postfix({
                "batch": BATCH_SIZE,
                "speed(doc/s)": f"{speed:.1f}"
            })
            pbar.update(len(batch))

print("Done! Output written to:", output_path)
