import subprocess
import os
import sys
import shutil
import pyserini
#from pyserini.index import IndexReader, IndexWriter, build_index
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------------------------------------------
lang = "ru" # for language specific analyzers en de it ja ru
input_path = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO\Dataset_Json\Russian"
index_path = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO\BM25_results\Russian_Index"
results_path = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO\BM25_results\Russian_Results\Russian_Results.txt"
topics_path = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO\Russian_queries.dev.tsv"
# ---------------------------------------------------------------------------------------------------------------

def check_java():
    java_home = os.environ.get("JAVA_HOME")
    java_exists = shutil.which("java") is not None

    if java_home:
        print(f"JAVA_HOME is set: {java_home}")
    else:
        print("JAVA_HOME is not set.")
    
    if not java_exists:
        print("Java executable not found in PATH.")
    
    if not java_home or not java_exists:
        print("Please install Java JDK and set JAVA_HOME before running this script.")
        sys.exit(1)

def run_pyserini():
    # Build index command
    cmd1_index = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_path,
        "--index", index_path,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--language", lang
    ]

    # BM25 search command
    cmd2_BM25search = [
        sys.executable, "-m", "pyserini.search.lucene",
        "--index", index_path,
        "--topics", topics_path,
        "--output", results_path,
        "--bm25",
        "--language", lang
    ]

    tracker = EmissionsTracker()
    tracker.start()
    try:
        #### MY CODE ####

        print("Running indexing...")
        #subprocess.run(cmd1_index, check=True)
        print("Indexing finished.")

        print("Running BM25 search...")
        subprocess.run(cmd2_BM25search, check=True)
        print(f"Search finished. Results saved to {results_path}")

        #### END MY CODE ####
        
    finally:
        tracker.stop()

    
if __name__ == "__main__":
    check_java()
    run_pyserini()
