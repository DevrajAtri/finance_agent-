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

def split_text_for_transcript(text, tokenizer, max_tokens=220, overlap_tokens=40):
    """
    Splits a long speaker turn into smaller, overlapping windows.
    (180-220 token windows with 40-token overlap)
    """
    target_window_size = 200 
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
        
    windows = []
    step = target_window_size - overlap_tokens
    for i in range(0, len(tokens), step):
        window_tokens = tokens[i: i + target_window_size]
        windows.append(tokenizer.decode(window_tokens))
        if i + target_window_size >= len(tokens):
            break
            
    return windows


def extract_speaker_turns(soup):
    """
    Step 1: Extracts all speaker turns from both <profileDesc> and <body>.
    A speaker turn is defined as a <div> with a <head> (speaker) and <p> (speech).
    """
    print("--- Step 1: Extracting Speaker Turns ---")
    raw_turns = []
    speaker_divs = soup.find_all('div', recursive=True)
    
    for div in speaker_divs:
        head = div.find('head', recursive=False)
        paragraphs = div.find_all('p', recursive=False)
        
        if head and paragraphs:
            speaker_text = head.get_text(strip=True)
            parts = [p.strip() for p in speaker_text.split('--')]
            name = parts[0]
            role = parts[1] if len(parts) > 1 else "Participant"
            
            speech = "\n".join([p.get_text(strip=True) for p in paragraphs])
            
            if speech: 
                raw_turns.append({"name": name, "role": role, "speech": speech})
                
    print(f"Extracted {len(raw_turns)} raw speaker turns.")
    return raw_turns

def chunk_and_segment_turns(raw_turns, tokenizer):
    """
    Steps 2 & 3: Identifies call segments (Prepared vs. Q&A) and applies
    token-based chunking rules to each turn.
    """
    print("--- Steps 2 & 3: Segmenting and Chunking Turns ---")
    segmented_chunks = []
    segment = "Prepared Remarks" 
    
    for turn in raw_turns:
        
        if "analyst" in turn["role"].lower() or "questions" in turn["name"].lower():
            segment = "Q&A"
        
        speech_text = turn["speech"]
        token_count = count_tokens(speech_text, tokenizer)
        
        if token_count <= 220:
            
            segmented_chunks.append({
                "name": turn["name"],
                "role": turn["role"],
                "segment": segment,
                "text": speech_text
            })
        else:
           
            windows = split_text_for_transcript(speech_text, tokenizer)
            for window_text in windows:
                segmented_chunks.append({
                    "name": turn["name"],
                    "role": turn["role"],
                    "segment": segment,
                    "text": window_text
                })
                
    print(f"Processed into {len(segmented_chunks)} segmented chunks.")
    return segmented_chunks

def package_transcript_chunks(chunks, doc_meta):
    """
    Step 4: Formats the final chunks with the required heading and full metadata.
    """
    print("--- Step 4: Formatting and Packaging with Metadata ---")
    packaged_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_meta['doc_id']}-{i}"
        
        heading_path = (f"{doc_meta['fiscal_quarter']} {doc_meta['fiscal_year']} › "
                        f"{chunk['segment']} › {chunk['name']}, {chunk['role']}")
        
        packaged_chunks.append({
            "doc_id": doc_meta["doc_id"],
            "chunk_id": chunk_id,
            "tier": doc_meta["tier"],
            "source_type": doc_meta["source_type"],
            "title": doc_meta["title"],
            "created_at": doc_meta["created_at"],
            "company": doc_meta["company"],
            "ticker": doc_meta["ticker"],
            "fiscal_year": doc_meta["fiscal_year"],
            "fiscal_quarter": doc_meta["fiscal_quarter"],
            "chunk_kind": "transcript_turn",
            "segment": chunk["segment"],
            "speaker_name": chunk["name"],
            "speaker_role": chunk["role"],
            "heading_path": heading_path,
            "chunk_text": chunk["text"],
        })
        
    print(f"Packaged {len(packaged_chunks)} chunks with full metadata.")
    return packaged_chunks
 
import argparse
import os

def process_single_file(input_path: str, output_dir: str):
    """
    Runs the full Transcript chunking pipeline on a single XML file.
    """
    try:
        print(f"  Processing file: {os.path.basename(input_path)}")
        with open(input_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        tokenizer = get_tokenizer()

        base_name = os.path.basename(input_path)
        doc_meta = {
            "doc_id": str(uuid.uuid4()),
            "title": base_name.replace(".pdf.tei.xml", ""),
            "company": "Unknown", "ticker": "XXXX",
            "fiscal_year": 2024, "fiscal_quarter": "Q4",
            "tier": 2, "source_type": "transcript",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        raw_speaker_turns = extract_speaker_turns(soup)
        segmented_and_chunked = chunk_and_segment_turns(raw_speaker_turns, tokenizer)
        final_output = package_transcript_chunks(segmented_and_chunked, doc_meta)
        
        output_filename = os.path.join(output_dir, f"{base_name}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)

    except Exception as e:
        print(f"    ✗ FAILED to process {os.path.basename(input_path)}: {e}")

# MAIN ORCHESTRATION SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transcript chunking pipeline.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .xml files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final .json chunk files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Transcript Pipeline for {args.input_dir} ---")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(args.input_dir, filename)
            process_single_file(file_path, args.output_dir)
            
    print(f"--- Transcript Pipeline complete for {args.input_dir} ---")