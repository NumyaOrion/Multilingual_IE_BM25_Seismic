import json
import time
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
model_name = "xlm-roberta-base"
BATCH_SIZE = 32
TOP_K = 64
MAX_LEN = 512

input_path = "mMARCO/Dataset_Json/German/mMARCO_collection_converted-german.jsonl"
output_path = "mMARCO/Dataset_Json/German/mMARCO_german_vectors_multi_unicoil_xlmr_topk.jsonl"
# mMARCO_lang_vectors_multi_unicoil_xlmr_topk.jsonl
# -----------------------------------------------------------
# Model setup
# -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Running on device:", device)
model.eval()

# -----------------------------------------------------------
# Batched sparse encoder
# -----------------------------------------------------------
def compute_unicoil_sparse_batch(texts):
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

        if TOP_K and len(real_weights) > TOP_K:
            idxs = torch.topk(torch.tensor(real_weights), TOP_K).indices.tolist()
        else:
            idxs = list(range(len(real_weights)))

        sparse = {real_tokens[i]: float(real_weights[i]) for i in idxs}
        results.append(sparse)

    return results

# -----------------------------------------------------------
# Count total lines for progress bar
# -----------------------------------------------------------
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

total_docs = count_lines(input_path)

# -----------------------------------------------------------
# Batch reader
# -----------------------------------------------------------
def read_in_batches(path, batch_size):
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# -----------------------------------------------------------
# Pipeline with SPEED DISPLAY
# -----------------------------------------------------------
start_time = time.time()
processed = 0

with open(output_path, "w", encoding="utf-8") as fout:

    with tqdm(total=total_docs, desc="Processing", unit="doc") as pbar:

        for batch in read_in_batches(input_path, BATCH_SIZE):
            texts = [d["contents"] for d in batch]
            sparse_vectors = compute_unicoil_sparse_batch(texts)

            for doc, vec in zip(batch, sparse_vectors):
                doc["vector"] = vec
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # Update counters
            processed += len(batch)
            elapsed = time.time() - start_time
            speed = processed / max(elapsed, 1e-9)

            # Update progress bar text
            pbar.set_postfix({
                "batch": BATCH_SIZE,
                "speed(doc/s)": f"{speed:.1f}"
            })
            pbar.update(len(batch))

print("Done! Output written to:", output_path)