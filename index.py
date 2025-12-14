from __future__ import annotations
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
from tqdm import tqdm
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import SpladeEncoder

load_dotenv()

DENSE_MODEL_PATH = os.getenv("DENSE_MODEL_PATH", "BAAI/bge-m3")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "financebot")
BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "32"))

_pc = None
_index = None
_device = None
_dense_model = None
_splade_model = None

def _get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device

def _load_dense():
    global _dense_model
    if _dense_model is None:
        print(f"Loading dense model: {DENSE_MODEL_PATH} (to {_get_device()})")
        _dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=_get_device())
        _dense_model.eval()

def _load_splade():
    global _splade_model
    if _splade_model is None:
        print(f"Loading sparse model to {_get_device()}...")
        _splade_model = SpladeEncoder(device=_get_device())

def connect_to_index() -> None:
    global _pc, _index
    if _index: return
    if _pc is None: _pc = Pinecone(api_key=PINECONE_API_KEY)
    _index = _pc.Index(PINECONE_INDEX_NAME)

@torch.no_grad()
def _encode_dense_batch(texts: List[str]) -> List[List[float]]:
    _load_dense()
    vecs = _dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in np.array(vecs)]

def _encode_sparse_batch(texts: List[str]) -> List[Dict[str, Any]]:
    _load_splade()
    return _splade_model.encode_documents(texts)

def load_chunks_from_json(path: str) -> List[Dict[str, Any]]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

def iter_all_chunks(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        for c in load_chunks_from_json(p):
            yield c

def encode_query(query_text: str) -> Dict[str, Any]:
    """
    Encodes a single query string into dense and sparse vectors.
    """
    _load_dense()
    _load_splade()
    
    dense_vec = _encode_dense_batch([query_text])[0]
    sparse_vec = _splade_model.encode_queries([query_text])[0]
    
    return {
        "dense": dense_vec,
        "sparse": sparse_vec
    }

def query_hybrid(
    query_text: str, 
    namespace: str, 
    top_k: int = 5, 
    filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs a Hybrid Search (Dense + Sparse) on the Pinecone index.
    """
    connect_to_index()
    
    vectors = encode_query(query_text)
    
    try:
        results = _index.query(
            vector=vectors["dense"],
            sparse_vector=vectors["sparse"],
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
            filter=filter 
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"matches": []}

def _flatten_and_serialize_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares chunk data for Pinecone metadata strict rules:
    1. Rename 'chunk_text' -> 'text'
    2. Flatten 'metadata_extras' (promote keys to top level)
    3. Stringify complex lists (like 'references')
    4. Cast list items to strings for fields like 'periods'
    """
    meta = {}
    
    # 1. Standardize Text Field
    if "chunk_text" in chunk:
        meta["text"] = chunk["chunk_text"]
    elif "text" in chunk:
        meta["text"] = chunk["text"]
    else:
        meta["text"] = ""

    # 2. Merge Top-Level Fields & Flatten Extras
    
    # Helper to process a single key-value pair
    def process_field(k, v):
        if k in ["chunk_text", "text", "values", "sparse_values", "id", "metadata_extras"]: 
            return 
        if v is None:
            return

        if isinstance(v, list):
           
            if all(isinstance(x, (str, int, float)) for x in v):
                
                meta[k] = [str(x) for x in v]
            else:
                meta[k] = json.dumps(v)
        
        elif isinstance(v, (str, int, float, bool)):
            meta[k] = v
        
        elif isinstance(v, dict):
           
            meta[k] = json.dumps(v)

    for k, v in chunk.items():
        process_field(k, v)
        
    if "metadata_extras" in chunk and isinstance(chunk["metadata_extras"], dict):
        for k, v in chunk["metadata_extras"].items():
            process_field(k, v)
            
    return meta

def build_hybrid_index(chunk_files: List[str], *, namespace: str, resume_at: int = 0, batch_size: int = BATCH_SIZE) -> None:
    """
    Upserts chunks to Pinecone with correct metadata formatting.
    """
    connect_to_index()
    
    batch_texts, batch_ids, batch_meta = [], [], []

    def _flush_with_retry():
        if not batch_ids: return
        
        for attempt in range(3):
            try:
                dense_vecs = _encode_dense_batch(batch_texts)
                sparse_vecs = _encode_sparse_batch(batch_texts)

                vectors = []
                for i in range(len(batch_ids)):
                    vectors.append({
                        "id": batch_ids[i],
                        "values": dense_vecs[i],
                        "sparse_values": sparse_vecs[i],
                        "metadata": batch_meta[i],
                    })
                
                _index.upsert(vectors=vectors, namespace=namespace)
                return 
                
            except Exception as e:
                print(f"\n[!] Network glitch on attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(15)
                else:
                    raise e 

        batch_ids.clear(); batch_texts.clear(); batch_meta.clear()

    print(f"Processing {len(chunk_files)} file(s) for '{namespace}'.")
    
    total_processed = 0
    for chunk in tqdm(iter_all_chunks(chunk_files), desc=f"Indexing {namespace}"):
        total_processed += 1
        if total_processed < resume_at: continue 

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("chunk_text", "")
        
        safe_meta = _flatten_and_serialize_metadata(chunk)
        
        if chunk_id and text:
            batch_ids.append(chunk_id)
            batch_texts.append(text)
            batch_meta.append(safe_meta)

        if len(batch_ids) >= batch_size:
            _flush_with_retry()
            batch_ids.clear(); batch_texts.clear(); batch_meta.clear()
    
    _flush_with_retry()
    print(f"Done with {namespace}.")