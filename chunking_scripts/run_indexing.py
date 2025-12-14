# run_indexing.py
import os
import glob
from index import build_hybrid_index


INDEXING_JOBS = [
    # DONE: Already finished, comment out to skip
    # {
    #     "name": "Textbook (Tier 1A)",
    #     "input_dir": "data/processed/tier1_groupA",
    #     "namespace": "textbook"
    # },
    # DONE: Already finished
    # {
    #     "name": "Glossary (Tier 1B)",
    #     "input_dir": "data/processed/tier1_groupB",
    #     "namespace": "glossary"
    # },
    {
        "name": "Filings (Tier 2)",
        "input_dir": "data/processed/tier2_filings",
        "namespace": "filings",
        "resume_at": 2300  
    },
    {
        "name": "Transcripts (Tier 2)",
        "input_dir": "data/processed/tier2_transcripts",
        "namespace": "transcripts",
        "resume_at": 0 
    }
]

def main():
    print("--- Resuming Indexing Pipeline ---")
    for job in INDEXING_JOBS:
        name = job['name']
        folder = job['input_dir']
        ns = job['namespace']
        start_chunk = job.get('resume_at', 0)
        
        json_files = glob.glob(os.path.join(folder, "*.json"))
        if not json_files: continue
            
        build_hybrid_index(
            chunk_files=json_files,
            namespace=ns,
            resume_at=start_chunk 
        )
        print(f" Finished: {name}")

if __name__ == "__main__":
    main()