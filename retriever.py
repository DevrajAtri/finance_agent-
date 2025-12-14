from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import concurrent.futures
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from index import encode_query 

from state import AgentState

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "financebot")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
RERANKER_MAX_LENGTH = 512

ALL_NAMESPACES = ["filings", "transcripts", "textbook", "glossary"]

_pc: Optional[Pinecone] = None
_index = None
_reranker: Optional[CrossEncoder] = None

def _ensure_clients():
    global _pc, _index
    if _pc is None:
        if not PINECONE_API_KEY: raise EnvironmentError("PINECONE_API_KEY not set")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    if _index is None:
        _index = _pc.Index(PINECONE_INDEX_NAME)

def _ensure_models():
    global _reranker
    if _reranker is None:
        print(f"Loading Reranker: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH)

@dataclass
class Candidate:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    origin_namespace: str


def _search_single_namespace(
    namespace: str, 
    dense_vec: List[float], 
    sparse_vec: Dict[str, Any], 
    top_k: int,
    filters: Optional[Dict[str, Any]] = None 
) -> List[Candidate]:
    try:
        res = _index.query(
            vector=dense_vec,
            # sparse_vector=sparse_vec, # comment because my vector database does not support hybrid search 
            top_k=top_k, 
            include_metadata=True, 
            namespace=namespace,
            filter=filters 
        )
        candidates = []
        for m in (res.matches or []):
            md = m.metadata or {}
            text = md.get("text") or md.get("chunk_text") or ""
            if text:
                candidates.append(Candidate(id=m.id, score=float(m.score), text=text, metadata=md, origin_namespace=namespace))
        return candidates
    except Exception as e:
        print(f"Warning: Failed to search namespace '{namespace}': {e}")
        return []

def _rerank_candidates(query: str, candidates: List[Candidate], top_k: int) -> List[Candidate]:
    _ensure_models()
    if not candidates: return []
    
    unique_candidates = {c.id: c for c in candidates}.values()
    candidates = list(unique_candidates)
    
    pairs = [(query, c.text[:4000]) for c in candidates]
    scores = _reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    
    for c, s in zip(candidates, scores):
        c.score = float(s)
    
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:top_k]


def retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves chunks based on the query, 'search_namespaces', and 'retrieval_filters'.
    """
    print("--- RETRIEVER: Executing search ---")
    query = state["query"]
    
    target_namespaces = state.get("search_namespaces")
    
    if not target_namespaces:
        print("Warning: No namespaces in state. Defaulting to ALL.")
        target_namespaces = ALL_NAMESPACES
    
    # Remove duplicates just in case
    target_namespaces = list(set(target_namespaces))
    
    filters = state.get("retrieval_filters")
    
    print(f"   Target Namespaces: {target_namespaces}")
    if filters:
        print(f"   Applying Filters: {filters}")

    try:
        _ensure_clients()
        qvecs = encode_query(query)
        
        # 3. Parallel Search
        all_candidates = []
        k_per_ns = 15 
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                
                executor.submit(_search_single_namespace, ns, qvecs["dense"], qvecs["sparse"], k_per_ns, filters): ns
                for ns in target_namespaces
            }
            for future in concurrent.futures.as_completed(futures):
                all_candidates.extend(future.result())

        if not all_candidates:
            print("   No candidates found.")
            return {"retrieved_chunks": []}

        # 4. Rerank
        print(f"   Reranking {len(all_candidates)} candidates...")
        final_candidates = _rerank_candidates(query, all_candidates, top_k=8)

        new_chunks = []
        for c in final_candidates:
            new_chunks.append({
                "id": c.id,
                "text": c.text,
                "score": c.score,
                "source": c.origin_namespace,
                "metadata": c.metadata
            })

        prev_chunks = state.get("retrieved_chunks", [])
        
        unique_chunks = {c["id"]: c for c in prev_chunks}
        for chunk in new_chunks:
            unique_chunks[chunk["id"]] = chunk
            
        final_combined_chunks = list(unique_chunks.values())
            
        print(f"--- RETRIEVER: Success. Found {len(new_chunks)} new. Total in state: {len(final_combined_chunks)}. ---")
        return {"retrieved_chunks": final_combined_chunks}

    except Exception as e:
        print(f"!!! RETRIEVER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"retrieved_chunks": []}