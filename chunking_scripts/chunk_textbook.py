import os
import json
import uuid
from datetime import datetime, timezone
import tiktoken
from bs4 import BeautifulSoup
from collections import defaultdict
import re


def get_tokenizer():
    """Initializes and returns the tiktoken tokenizer."""
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, tokenizer):
    """Counts tokens in a text string."""
    return len(tokenizer.encode(text))

def classify_content_from_xml(tei_xml_content):
    """
    Parses, classifies, and filters content from the textbook XML.
    This is the core of the V3 strategy for textbooks.
    """
    print("--- Starting Step 1: Parsing, Classifying, and Filtering Content ---")
    soup = BeautifulSoup(tei_xml_content, 'lxml-xml')
    body = soup.find('body')
    if not body: return []

    headings_to_skip = [
        "Chapter Outline", "Community Hubs", "Technology Partners", "Summary",
        "Key Terms", "Learning Outcomes", "Multiple Choice", "Review Questions",
        "Problems", "Video Activity", "CFA Institute"
    ]
    concept_box_headings = [
        "CONCEPTS IN PRACTICE", "LINK TO LEARNING", "THINK IT THROUGH", "Why It Matters"
    ]

    classified_elements = []
    current_headings = []
    current_page = 1
    skip_current_section = False
    
    for element in body.find_all(['head', 'p', 'formula', 'table', 'div', 'pb']):

        if element.name == 'pb':
            current_page = int(element.get('n', current_page))
            continue

        parent_div = element.find_parent('div')
        is_in_special_div = parent_div and parent_div.find('head') and any(
            box_title in parent_div.find('head').get_text(strip=True) for box_title in concept_box_headings + ["Key Terms"]
        )
        if is_in_special_div and element != parent_div:
            continue

        if element.name == 'head':
            heading_text = element.get_text(strip=True)
            
            level = len(element.get('n', '1.1.1.1').split('.'))
            current_headings = current_headings[:level-1]
            current_headings.append(heading_text)

            if any(skip_title in h for h in current_headings for skip_title in headings_to_skip):
                skip_current_section = True
            else:
                skip_current_section = False
            continue

        if skip_current_section:
            continue

        element_type = 'unknown'
        content_text = element.get_text(strip=True)
        heading_path = " › ".join(current_headings)

        if element.name == 'p' and content_text: element_type = 'paragraph'
        elif element.name == 'formula': element_type = 'formula_box'
        elif element.name == 'table': element_type = 'table'
        elif element.name == 'div':
            div_head_tag = element.find('head', recursive=False)
            if div_head_tag:
                div_head_text = div_head_tag.get_text(strip=True)
                if any(box_title in div_head_text for box_title in concept_box_headings):
                    element_type, heading_path = 'concept_box', f"{heading_path} › {div_head_text}"
                    content_text = element.get_text(strip=True)

        if element_type != 'unknown' and content_text:
            classified_elements.append({
                "type": element_type,
                "heading_path": heading_path,
                "text": content_text,
                "page": current_page 
            })

    print(f"Step 1 Complete. Classified {len(classified_elements)} valid content elements.")
    return classified_elements

def chunk_boxed_content(classified_elements, tokenizer):
    print("--- Starting Step 2: Chunking 'Boxed' Content ---")
    boxed_chunks = []
    for element in classified_elements:
        if element['type'] in ['formula_box', 'concept_box']:
            boxed_chunks.append({
                "chunk_kind": "formula" if element['type'] == 'formula_box' else "concept_box",
                "heading_path": element['heading_path'],
                "chunk_text": element['text'],
                "page_start": element['page'],
                "page_end": element['page']
            })
    print(f"Step 2 Complete. Created {len(boxed_chunks)} formula/concept box chunks.")
    return boxed_chunks

def split_long_paragraph(text, tokenizer, max_tokens=350, overlap_tokens=50):
    tokens = tokenizer.encode(text)
    if len(tokens) <= 400: return [text]
    windows, step_size = [], max_tokens - overlap_tokens
    for i in range(0, len(tokens), step_size):
        window_tokens = tokens[i: i + max_tokens]
        windows.append(tokenizer.decode(window_tokens))
        if i + max_tokens >= len(tokens): break
    return windows

def chunk_narrative_content(classified_elements, tokenizer):
    print("--- Starting Step 3: Chunking Main Narrative Text ---")
    sections = defaultdict(list)
    for element in classified_elements:
        if element['type'] == 'paragraph':
            sections[element['heading_path']].append(element)

    all_narrative_chunks = []
    for heading, paragraphs_data in sections.items():
        text_blocks = []
        for p_data in paragraphs_data:
            for window in split_long_paragraph(p_data['text'], tokenizer):
                text_blocks.append({"text": window, "page": p_data['page']})
        
        current_texts, current_pages = [], []
        for block in text_blocks:
            current_texts.append(block['text'])
            current_pages.append(block['page'])
            potential_chunk = "\n\n".join(current_texts)
            token_count = count_tokens(potential_chunk, tokenizer)
            
            if token_count >= 400:
                all_narrative_chunks.append({
                    "chunk_kind": "concept", "heading_path": heading, "chunk_text": potential_chunk,
                    "page_start": min(current_pages), "page_end": max(current_pages)
                })
                overlap_text = tokenizer.decode(tokenizer.encode(potential_chunk)[-50:])
                current_texts, current_pages = [overlap_text], [max(current_pages)]

        if len(current_texts) > 1 or (current_texts and not current_texts[0].startswith("...")):
            final_text = "\n\n".join(current_texts)
            if final_text.strip():
                all_narrative_chunks.append({
                    "chunk_kind": "concept", "heading_path": heading, "chunk_text": final_text,
                    "page_start": min(current_pages), "page_end": max(current_pages)
                })

    print(f"Step 3 Complete. Created {len(all_narrative_chunks)} narrative chunks.")
    return all_narrative_chunks

def finalize_chunks_with_metadata(all_raw_chunks, doc_meta):
    print("--- Starting Step 4: Finalizing Metadata for All Chunks ---")
    packaged_chunks = []
    for i, chunk in enumerate(all_raw_chunks):
        packaged_chunk = {
            "doc_id": doc_meta["doc_id"],
            "chunk_id": f"{doc_meta['doc_id']}-{i}",
            "tier": doc_meta["tier"],
            "source_type": doc_meta["source_type"],
            "title": doc_meta["title"],
            "created_at": doc_meta["created_at"],
            "chunk_kind": chunk["chunk_kind"],
            "heading_path": chunk["heading_path"],
            "chunk_text": chunk["chunk_text"],
            "page_start": chunk.get("page_start", 1),
            "page_end": chunk.get("page_end", 1)
        }
        packaged_chunks.append(packaged_chunk)
    
    print(f"Step 4 Complete. Packaged {len(packaged_chunks)} total chunks.")
    return packaged_chunks

import argparse
import os

def process_single_file(input_path: str, output_dir: str):
    """
    Runs the full Textbook chunking pipeline on a single XML file.
    """
    try:
        print(f"  Processing file: {os.path.basename(input_path)}")
        with open(input_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
            
        tokenizer = get_tokenizer()
        base_name = os.path.basename(input_path)
        doc_meta = {
            "doc_id": str(uuid.uuid4()),
            "title": base_name.replace(".pdf.tei.xml", ""),
            "tier": 1, "source_type": "foundational",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        classified_elements = classify_content_from_xml(xml_content)
        boxed_chunks = chunk_boxed_content(classified_elements, tokenizer)
        narrative_chunks = chunk_narrative_content(classified_elements, tokenizer)
        all_generated_chunks = boxed_chunks + narrative_chunks
        final_output = finalize_chunks_with_metadata(all_generated_chunks, doc_meta)
        
        output_filename = os.path.join(output_dir, f"{base_name}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)

    except Exception as e:
        print(f"  FAILED to process {os.path.basename(input_path)}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Textbook chunking pipeline.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw .xml files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final .json chunk files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Textbook Pipeline for {args.input_dir} ---")
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".xml"):
            file_path = os.path.join(args.input_dir, filename)
            process_single_file(file_path, args.output_dir)
            
    print(f"--- Textbook Pipeline complete for {args.input_dir} ---")