import os
import csv
import json
import uuid
from datetime import datetime, timezone
import tiktoken
from bs4 import BeautifulSoup
import argparse
from typing import List, Dict, Any

def get_tokenizer():
    """Initializes and returns the tiktoken tokenizer."""
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, tokenizer):
    """Counts tokens in a text string."""
    return len(tokenizer.encode(text))

def preprocess_and_cache_entities(soup, tokenizer, output_dir="structured_tables"):
    """
    Finds all entity definitions (tables, figures, footnotes), processes them
    into chunk-ready dictionaries, and stores them in a cache.
    (This version is robust against all known errors)
    """
    print("--- Starting Step A: Caching All Referenced Entities ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    entity_cache = {}

    # 1. Process Tables AND Figures in one go.
    # We find ALL <figure> tags and then decide what to do.
    
    all_figures = soup.find_all('figure')
    
    for fig in all_figures:
        
        # First, check the 'type' attribute to see if it's a table.
        fig_type = fig.get('type')
        
        if fig_type == 'table':
            #  It's a TABLE, run table logic 
            try:
                table_id = fig.get('xml:id')
                if not table_id:
                    continue

                head_tag = fig.find('head')
                title = head_tag.get_text(strip=True) if head_tag else f"Table ({table_id})"
                
                desc_tag = fig.find('figDesc')
                desc = desc_tag.get_text(strip=True) if desc_tag else ""
                
                table_tag = fig.find('table')
                if not table_tag:
                    continue
                    
                rows = []
                for row in table_tag.find_all('row'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all('cell')]
                    if cells:
                        rows.append(cells)
                
                if not rows:
                    print(f"    - Skipping table {table_id} (no rows found).")
                    continue
                    
                csv_path = os.path.join(output_dir, f"{table_id}.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(rows)

                header = rows[0]
                data_rows = rows[1:]
                
                linearized_lines = []
                if data_rows:
                    for r in data_rows:
                        if not r: continue
                        row_key = r[0] if r else ""
                        row_data = [f"{header[i]}: {r[i]}" for i in range(1, min(len(r), len(header))) if i < len(r)]
                        linearized_lines.append(f"{row_key} -> " + ", ".join(row_data))
                
                linearized_text = "\n".join(linearized_lines)
                chunk_text = f"Title: {title}\n\nDescription: {desc}\n\nData:\n{linearized_text}"
                
                entity_cache[table_id] = {
                    "chunk_kind": "table_text",
                    "heading_path": title,
                    "chunk_text": chunk_text,
                    "metadata_extras": {
                        "table_title": title,
                        "table_struct_path": csv_path,
                        "units": "See document",
                        "periods": [h for h in header if h and ("FY" in h or any(c.isdigit() for c in h))]
                    }
                }
            except Exception as e:
                print(f"    - WARNING: Skipping a table due to unexpected error: {e}")
                continue
        
        else:
            #  It's a FIGURE (or has no type), run figure logic 
            try:
                fig_id = fig.get('xml:id')
                if not fig_id:
                    continue
                
                caption_tag = fig.find('head')
                caption = caption_tag.get_text(strip=True) if caption_tag else f"Figure ({fig_id})"
                
                desc_tag = fig.find('figDesc')
                desc = desc_tag.get_text(strip=True) if desc_tag else ""
                
                chunk_text = f"Caption: {caption}\n\nDescription: {desc}".strip()
                if chunk_text:
                    entity_cache[fig_id] = {
                        "chunk_kind": "caption",
                        "heading_path": caption,
                        "chunk_text": chunk_text
                    }
            except Exception as e:
                print(f"    - WARNING: Skipping a figure due to unexpected error: {e}")
                continue

    # 3. Process Footnotes (<note place="foot">)
    for note in soup.find_all('note', {'place': 'foot'}):
        try:
            note_id = note.get('xml:id')
            if not note_id:
                continue
                
            chunk_text = note.get_text(strip=True)
            if chunk_text:
                heading = f"Footnote {note_id.replace('foot_', '')}"
                entity_cache[note_id] = {
                    "chunk_kind": "footnote",
                    "heading_path": heading,
                    "chunk_text": chunk_text
                }
        except Exception as e:
            print(f"    - WARNING: Skipping a footnote due to unexpected error: {e}")
            continue
            
    print(f"Step A Complete. Cached {len(entity_cache)} entities.")
    return entity_cache


def process_narrative_and_create_links(soup, tokenizer):
    """
    Processes the main text, creating narrative chunks and recording outbound
    references to entities in their metadata. Also tracks page numbers.
    """
    print("--- Starting Step B: Processing Narrative and Creating Links ---")
    
    narrative_chunks = []
    current_page = 1
    
    body = soup.body
    if not body:
        print("    - No <body> tag found. Skipping narrative processing.")
        return []
        
    content_elements = body.find_all(['head', 'p', 'pb'])
    if not content_elements:
        print("    - No <head>, <p>, or <pb> tags found in <body>. Skipping narrative processing.")
        return []
        
    sections = {}
    current_headings = []
    
    for element in content_elements:
        try:
            if element.name == 'pb':
                current_page = int(element.get('n', current_page))
                continue
                
            start_page = current_page
            
            if element.name == 'head':
                current_headings.append(element.get_text(strip=True))
            elif element.name == 'p':
                path = " > ".join(current_headings) if current_headings else "Introduction"
                if path not in sections:
                    sections[path] = []
                
                references = []
                for ref in element.find_all('ref'):
                    references.append({
                        "text_in_chunk": ref.get_text(strip=True),
                        "type": ref.get("type"),
                        "target_id": ref.get("target", "").replace("#", "")
                    })
                
                #  SAFE PAGE FIND 
                end_page_tag = element.find_next('pb')
                end_page = int(end_page_tag.get('n', current_page)) if end_page_tag else current_page
                
                text_content = element.get_text(strip=True)
                if text_content: 
                    sections[path].append({
                        "text": text_content,
                        "references": references,
                        "page_start": start_page,
                        "page_end": end_page
                    })
        except Exception as e:
            print(f"    - WARNING: Skipping a narrative element due to unexpected error: {e}")
            continue
            
   
    for heading, paragraphs_data in sections.items():
        current_texts, current_refs, current_pages = [], [], []
        current_tokens = 0
        
        for p_data in paragraphs_data:
            text = p_data['text']
            tokens = count_tokens(text, tokenizer)
            
            if current_tokens > 0 and current_tokens + tokens > 700:
                chunk_text = "\n\n".join(current_texts)
                narrative_chunks.append({
                    "chunk_kind": "narrative",
                    "heading_path": heading,
                    "chunk_text": chunk_text,
                    "metadata_extras": {
                        "references": current_refs,
                        "page_start": min(current_pages) if current_pages else 1,
                        "page_end": max(current_pages) if current_pages else 1
                    }
                })
                overlap_text = tokenizer.decode(tokenizer.encode(chunk_text)[-60:])
                current_texts = [overlap_text, text]
                current_refs = p_data['references']
                current_pages = [p_data['page_start'], p_data['page_end']]
                current_tokens = count_tokens("\n\n".join(current_texts), tokenizer)
            else:
                current_texts.append(text)
                current_refs.extend(p_data['references'])
                current_pages.extend([p_data['page_start'], p_data['page_end']])
                current_tokens += tokens
                
        if current_texts:
            narrative_chunks.append({
                "chunk_kind": "narrative",
                "heading_path": heading,
                "chunk_text": "\n\n".join(current_texts),
                "metadata_extras": {
                    "references": current_refs,
                    "page_start": min(current_pages) if current_pages else 1,
                    "page_end": max(current_pages) if current_pages else 1
                }
            })
            
    print(f"Step B Complete. Created {len(narrative_chunks)} narrative chunks.")
    return narrative_chunks

def finalize_chunks_with_metadata(narrative_chunks: List[Dict], entity_cache: Dict, doc_meta: Dict):
    """
    Assigns final metadata to all chunks and creates the two-way reference links.
    """
    print("--- Starting Step C: Finalizing Metadata and Cross-Linking ---")
    
    for i, chunk in enumerate(narrative_chunks):
        chunk['chunk_id'] = f"{doc_meta['doc_id']}-{i}"
        
    for entity_id, entity_chunk in entity_cache.items():
        if "metadata_extras" not in entity_chunk:
            entity_chunk["metadata_extras"] = {}
        entity_chunk["metadata_extras"]["referenced_in"] = []

    for narrative_chunk in narrative_chunks:
        #  SAFE GET 
        if "references" in narrative_chunk.get("metadata_extras", {}):
            for ref in narrative_chunk["metadata_extras"]["references"]:
                target_id = ref.get("target_id")
                if target_id in entity_cache:
                   
                    if "referenced_in" not in entity_cache[target_id].get("metadata_extras", {}):
                         entity_cache[target_id]["metadata_extras"]["referenced_in"] = []
                    entity_cache[target_id]["metadata_extras"]["referenced_in"].append(narrative_chunk['chunk_id'])

    # Combine all chunks
    all_chunks = narrative_chunks + list(entity_cache.values())
    final_packaged_chunks = []
    
    for i, chunk in enumerate(all_chunks):
        chunk_id = chunk.get('chunk_id', f"{doc_meta['doc_id']}-{i}")
        
        packaged_chunk = {
            "doc_id": doc_meta.get("doc_id"),
            "chunk_id": chunk_id,
            "tier": doc_meta.get("tier"),
            "source_type": doc_meta.get("source_type"),
            "title": doc_meta.get("title"),
            "filing_type": doc_meta.get("filing_type"),
            "filing_date": doc_meta.get("filing_date"),
            "company": doc_meta.get("company"),
            "ticker": doc_meta.get("ticker"),
            "created_at": doc_meta.get("created_at"),
            "chunk_kind": chunk.get("chunk_kind"),
            "heading_path": chunk.get("heading_path"),
            "chunk_text": chunk.get("chunk_text"),
            "page_start": 1, 
            "page_end": 1
        }
        
        if "metadata_extras" in chunk:
            packaged_chunk.update(chunk["metadata_extras"])
            
        final_packaged_chunks.append(packaged_chunk)
        
    print(f"Step C Complete. Finalized {len(final_packaged_chunks)} total chunks.")
    return final_packaged_chunks


def extract_document_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extracts high-level metadata from the document's TEI header."""
    
    title_tag = None
    title = "Unknown Document"
    
    try:
       
        title_stmt = soup.find('titleStmt')
        if title_stmt:
            title_tag = title_stmt.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
    except Exception as e:
        print(f"    - WARNING: Could not parse title. Using default. Error: {e}")

    return {
        "doc_id": str(uuid.uuid4()),
        "title": title,
        "company": "Unknown Company", 
        "ticker": "XXXX", 
        "cik": "0000000", 
        "filing_type": "Unknown Filing",
        "filing_date": datetime.now(timezone.utc).isoformat(),
        "tier": 2, 
        "source_type": "filing",
        "created_at": datetime.now(timezone.utc).isoformat()
    }


def process_single_file(input_path: str, output_dir: str):
    """
    Runs the full V3 chunking pipeline on a single XML file.
    """
    try:
        print(f"  Processing file: {os.path.basename(input_path)}")
        with open(input_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        tokenizer = get_tokenizer()
        
        doc_meta = extract_document_metadata(soup)
        entity_cache = preprocess_and_cache_entities(soup, tokenizer)
        narrative_chunks = process_narrative_and_create_links(soup, tokenizer)
        final_output = finalize_chunks_with_metadata(narrative_chunks, entity_cache, doc_meta)
        
        base_name = os.path.basename(input_path)
        output_filename = os.path.join(output_dir, f"{base_name}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)

    except Exception as e:
        
        print(f"    ✗ FAILED to process {os.path.basename(input_path)}: {e}")
        import traceback
        traceback.print_exc() # This will print the exact line where the error occurred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V3 Filings chunking pipeline.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .xml files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final .json chunk files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Filings Pipeline for {args.input_dir} ---")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(args.input_dir, filename)
            process_single_file(file_path, args.output_dir)
            
    print(f"--- Filings Pipeline complete for {args.input_dir} ---")