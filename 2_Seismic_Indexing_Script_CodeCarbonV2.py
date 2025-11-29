import json
import numpy as np
import seismic
from seismic import SeismicDataset

from codecarbon import EmissionsTracker

json_input = "mMARCO/Dataset_Json/Russian/mMARCO_russian_vectors_multi_unicoil_xlmr_topk.jsonl"
# don't forget to adjust index_path below

dataset = SeismicDataset()
string_type = seismic.get_seismic_string()

tracker = EmissionsTracker()
    
print("Running indexing...")

tracker.start()
try:
    #### MY CODE ####
    
    with open(json_input, "r") as f:
        for line in f:
            d = json.loads(line)
            ks = np.array(list(d["vector"].keys()), dtype=string_type)
            vs = np.array(list(d["vector"].values()), dtype=np.float32)
            dataset.add_document(str(d["id"]), ks, vs)
    
    print("Running indexing...")

    index = seismic.SeismicIndex.build_from_dataset(
        dataset,
        n_postings=300,
        centroid_fraction=0.01,
        min_cluster_size=4,
        summary_energy=0.2,
        max_fraction=1.2,
        doc_cut=15,
        batched_indexing=100_000,   # ← CRITICAL
        num_threads=0               # use all cores
    )

    index_path = "mMARCO/Seismic_results/Seismic_Index_Russian"
    index.save(index_path)

    print("Index saved.")
    #### END MY CODE ####
        
finally:
    tracker.stop()
