import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
model_name = "xlm-roberta-base"
BATCH_SIZE = 32
TOP_K = 64
MAX_LEN = 512

input_path = "mMARCO/russian_queries.dev.tsv"
output_path = "mMARCO/russian_queries_vectors.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def compute_unicoil_sparse_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_weights = outputs.last_hidden_state.mean(dim=2)

    id_lists = inputs["input_ids"].cpu().tolist()
    token_lists = [tokenizer.convert_ids_to_tokens(ids) for ids in id_lists]

    results = []
    for tokens, weights in zip(token_lists, token_weights.cpu()):
        weights = weights.tolist()

        # remove padding tokens
        valid_tokens = []
        valid_weights = []
        for tok, w in zip(tokens, weights):
            if tok != tokenizer.pad_token:
                valid_tokens.append(tok)
                valid_weights.append(w)

        # keep top-K
        if TOP_K and len(valid_weights) > TOP_K:
            idxs = torch.topk(torch.tensor(valid_weights), TOP_K).indices.tolist()
        else:
            idxs = list(range(len(valid_weights)))

        sparse = {valid_tokens[i]: float(valid_weights[i]) for i in idxs}
        results.append(sparse)

    return results

# ---------------------

queries = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        qid, text = line.strip().split("\t", 1)
        queries.append({"id": int(qid), "content": text})

with open(output_path, "w", encoding="utf-8") as fout:
    for i in tqdm(range(0, len(queries), BATCH_SIZE)):
        batch = queries[i:i+BATCH_SIZE]
        texts = [q["content"] for q in batch]

        sparse_vectors = compute_unicoil_sparse_batch(texts)

        for q, vec in zip(batch, sparse_vectors):
            out = {
                "id": q["id"],
                "content": q["content"],
                "vector": vec
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print("Done: wrote Seismic query vectors.")
