import os
import json
import requests
import shutil
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path


from chunking_scripts.chunk_filings import process_single_file as process_filings
from chunking_scripts.chunk_textbook import process_single_file as process_textbook
from index import build_hybrid_index 

from state import AgentState 

# The folder where we physically store the uploaded PDF
UPLOAD_DIR = os.path.join("data", "user_uploads")

# The folders for intermediate processing
XML_DIR = os.path.join("data", "processing", "xml")
JSON_DIR = os.path.join("data", "processing", "json")

# Grobid URL (Assumed running locally via Docker)
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# Define Input Type for the Node
class IngestInput(TypedDict):
    # This matches the user_file_info in AgentState
    path: str 
    name: str

def _ensure_directories():
    """Creates the necessary folder structure if it doesn't exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(XML_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

def _call_grobid(file_path: str) -> str:
    """Sends PDF to Grobid and returns the XML string."""
    print(f"--- Calling Grobid for: {os.path.basename(file_path)} ---")
    
    files = {
        'input': (
            os.path.basename(file_path), 
            open(file_path, 'rb'), 
            'application/pdf', 
            {'Expires': '0'}
        )
    }
    
    try:
        response = requests.post(GROBID_URL, files=files, timeout=300)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Grobid Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to Grobid at localhost:8070. Is the Docker container running?")

def _decide_chunker(xml_content: str) -> str:
    """
    Scans XML to decide if it's a Filing (10-K/Q) or General/Textbook.
    Returns: 'filings' or 'textbook'
    """
    lower_xml = xml_content.lower()
    
    # Heuristics for SEC Filings
    if "form 10-k" in lower_xml or "form 10-q" in lower_xml:
        return "filings"
    if "united states securities and exchange commission" in lower_xml:
        return "filings"
        
    return "textbook"


def ingest_user_file(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrates the full pipeline:
    1. Setup Folders
    2. Grobid (PDF -> XML)
    3. Classification -> Chunker (XML -> JSON)
    4. Indexing (JSON -> Pinecone 'user' namespace)
    5. State Update (Exclusive Mode)
    """
    print("--- NODE: Ingest User File ---")
    
    # 1. Validation
    file_info = state.get("user_file_info")
    if not file_info or "path" not in file_info:
        error_msg = "No file path provided in state['user_file_info']."
        print(f"Error: {error_msg}")
        return {
            "tool_error_count": 1,
            "chat_history": [("ai", f"System Error: {error_msg}")]
        }

    original_path = file_info["path"]
    filename = os.path.basename(original_path)
    base_name = filename.replace(".pdf", "")
    
    # 2. Setup Directories
    _ensure_directories()
    
    # 3. Move/Copy File to Persistent 'data/user_uploads'
    # We copy it there to keep a history and ensure safe access
    permanent_pdf_path = os.path.join(UPLOAD_DIR, filename)
    try:
        shutil.copy2(original_path, permanent_pdf_path)
    except Exception as e:
        print(f"Warning: Could not copy file to storage: {e}. Using original path.")
        permanent_pdf_path = original_path

    try:
        # 4. GROBID Processing
        xml_content = _call_grobid(permanent_pdf_path)
        
        # Save XML to hardcoded XML_DIR
        xml_file_path = os.path.join(XML_DIR, f"{base_name}.xml")
        with open(xml_file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        # 5. Classification & Chunking
        strategy = _decide_chunker(xml_content)
        print(f"--- Strategy Detected: {strategy.upper()} ---")
        
        # The chunkers take: (input_xml_path, output_json_dir)
        if strategy == "filings":
            process_filings(xml_file_path, JSON_DIR)
        else:
            process_textbook(xml_file_path, JSON_DIR)

        # 6. Indexing (Upsert to Pinecone)
        # Expected JSON path created by the chunker
        expected_json_path = os.path.join(JSON_DIR, f"{base_name}.json")
        
        if not os.path.exists(expected_json_path):
            raise Exception(f"Chunker completed but output file not found: {expected_json_path}")
            
        print(f"--- Upserting to Namespace: 'user' ---")
        
        # Call the corrected build_hybrid_index from index.py
        # It takes a LIST of file paths
        build_hybrid_index(
            chunk_files=[expected_json_path], 
            namespace="user"
        )
        
        print("--- Ingestion Complete ---")

        
        
        return {
            "search_namespaces": ["user"], 
            "chat_history": [
                ("system", f"User file '{filename}' has been indexed. Switch to 'user' namespace mode. Answer based ONLY on this file.")
            ],
            "next_step": "router"
        }

    except Exception as e:
        print(f"!!! INGESTION FAILED: {e} !!!")
        import traceback
        traceback.print_exc()
        
        return {
            "tool_error_count": 1,
            "search_namespaces": [], 
            "chat_history": [("ai", f"I encountered an error processing the file: {str(e)}. Please check if Grobid is running.")]
        }