import os
import json
import uuid
from datetime import datetime, timezone
import tiktoken
from bs4 import BeautifulSoup
import re


def get_tokenizer():
    """Initializes and returns the tiktoken tokenizer."""
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, tokenizer):
    """Counts tokens in a text string."""
    return len(tokenizer.encode(text))

def create_glossary_chunks(tei_xml_content):
    """
    Parses, filters, chunks, and packages glossary terms from a TEI XML file.

    Args:
        tei_xml_content (str): The XML content of the glossary document.

    Returns:
        list: A list of fully formatted and metadata-rich chunk dictionaries.
    """
    print("--- Starting Glossary Chunking Pipeline ---")
    
    soup = BeautifulSoup(tei_xml_content, 'lxml-xml')
    tokenizer = get_tokenizer()

    title_tag = soup.find('title', type='main')
    doc_meta = {
        "doc_id": str(uuid.uuid4()),
        "title": title_tag.get_text(strip=True) if title_tag else "Financial Glossary",
        "tier": 1,
        "source_type": "foundational",
        "created_at": datetime.now(timezone.utc).isoformat()
    }


    print("Step 1: Parsing XML and filtering introductory content...")
    
    headings_to_skip = ['FOREWORD', 'PREFACE', 'Acknowledgement']
    all_divs = soup.body.find_all('div', recursive=False)
    
    raw_chunks = []
    current_page = 1

    for div in all_divs:
        head_tag = div.find('head', recursive=False)
        p_tag = div.find('p', recursive=False)

        if head_tag and p_tag:
            term = head_tag.get_text(strip=True)
            definition = p_tag.get_text(strip=True)

            if term.upper() in headings_to_skip or len(term) == 1:
                print(f"  - Skipping section: '{term}'")
                continue
            
            chunk_text = f"{term}: {definition}"
            
            raw_chunks.append({
                "term": term,
                "text": chunk_text,
                "page": current_page 
            })

    print(f"Step 1 & 2 Complete. Extracted {len(raw_chunks)} raw glossary terms.")

    print("Step 3: Packaging all chunks with detailed metadata...")
    packaged_chunks = []
    
    for i, chunk_data in enumerate(raw_chunks):
        term = chunk_data["term"]
        first_letter = term[0].upper()
        
        heading_path = f"Glossary › {first_letter} › {term}"
        
        chunk_text = chunk_data["text"]
        token_count = count_tokens(chunk_text, tokenizer)

        min_tokens, max_tokens = 80, 180
        if not (min_tokens <= token_count <= max_tokens):
            print(f"  - Warning: Glossary chunk for '{term}' has {token_count} tokens (outside {min_tokens}-{max_tokens} range).")

        packaged_chunks.append({
           
            "doc_id": doc_meta["doc_id"],
            "chunk_id": f"{doc_meta['doc_id']}-{i}",
            "tier": doc_meta["tier"],
            "source_type": doc_meta["source_type"],
            "title": doc_meta["title"],
            "heading_path": heading_path,
            "page_start": chunk_data["page"],
            "page_end": chunk_data["page"],
            "created_at": doc_meta["created_at"],
            "chunk_kind": "glossary",
            "term": term,
            "aliases": [], 
            "chunk_text": chunk_text,
            "token_count": token_count
        })

    print(f"Step 3 Complete. Finalized {len(packaged_chunks)} glossary chunks.")
    return packaged_chunks

import argparse
import os

def process_single_file(input_path: str, output_dir: str):
    """
    Runs the full Glossary chunking pipeline on a single XML file.
    """
    try:
        print(f"  Processing file: {os.path.basename(input_path)}")
        with open(input_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        final_output = create_glossary_chunks(xml_content)
        
        base_name = os.path.basename(input_path)
        output_filename = os.path.join(output_dir, f"{base_name}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)

    except Exception as e:
        print(f"    ✗ FAILED to process {os.path.basename(input_path)}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glossary chunking pipeline.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .xml files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final .json chunk files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Glossary Pipeline for {args.input_dir} ---")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(args.input_dir, filename)
            process_single_file(file_path, args.output_dir)
            
    print(f"--- Glossary Pipeline complete for {args.input_dir} ---")