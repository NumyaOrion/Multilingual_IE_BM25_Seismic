import numpy as np
import json
from tqdm import tqdm
from seismic import SeismicIndex

from codecarbon import EmissionsTracker
tracker = EmissionsTracker()

index_path = "mMARCO/Seismic_results/Seismic_Index_Russian.index.seismic"
index = SeismicIndex.load(index_path)

file_path = "mMARCO/russian_queries_vectors.jsonl"      # input queries
output_path = "mMARCO/Seismic_results/Russian_seismic_results.trec"  # where results will be saved

tracker.start()
try:
    #### MY CODE ####
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            queries.append(json.loads(line))

    MAX_TOKEN_LEN = 30
    string_type  = f'U{MAX_TOKEN_LEN}'

    queries_ids = np.array([q['id'] for q in queries], dtype=string_type)

    query_components = []
    query_values = []

    for query in queries:
        vector = query['vector']
        query_components.append(np.array(list(vector.keys()), dtype=string_type))
        query_values.append(np.array(list(vector.values()), dtype=np.float32))

    # search and write results into a txt-file

    with open(output_path, "w") as out:
        for i in tqdm(range(len(queries_ids)), desc="Searching"):
            qid = str(queries_ids[i])
            comps = query_components[i]
            vals = query_values[i]

            results = index.search(
                query_id=qid,
                query_components=comps,
                query_values=vals,
                k=1000, # need 1000 for comparison to BM25
                query_cut=20, # changed from 20 to 1000 = k
                heap_factor=0.7, # changed to 1.0 from formerly 0.7 woudl retrun only 700 docs
                n_knn=0,
                sorted=True,
            )

            # Write results in TREC format
            for rank, hit in enumerate(results):
                doc_id = hit[2]
                score = hit[1]
                out.write(f"{qid} Q0 {doc_id} {rank} {score} seismic\n")
    #### END MY CODE ####
        
finally:
    tracker.stop()