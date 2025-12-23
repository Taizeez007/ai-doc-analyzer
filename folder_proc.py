from read_text_from_file import read_raw_text_from_file
import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext
from doc_proc import build_index
from doc_proc import extract_fields_from_document

def process_local_folder(input_dir: str, llm, output_file: str = 'extracted.json') -> Tuple[List[Dict], pd.DataFrame]:
    """
    Processes a local folder of documents (.pdf, .docx, .txt), extracts structured fields
    using the provided LLM, writes JSON output, and returns a DataFrame.
    """
    rows = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(('.pdf', '.docx', '.txt')):
                continue

            full_path = os.path.join(root, fname)
            text = read_raw_text_from_file(full_path)

            if not text.strip():
                print(f"Skipping empty file: {fname}")
                continue

            extracted = extract_fields_from_document(llm, text)
            if isinstance(extracted, dict):
                extracted=[extracted] #wrap dict in list
            elif isinstance(extracted, list):
                extracted = [item for item in extracted if isinstance(item, dict)]
            else:
                print(f"unexpected extraction type: {type(extracted)} for file {fname}")
                continue


            for item in extracted:
                item['source_file'] = fname
                rows.append(item)
           

    # Write JSON output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    # Return DataFrame
    df = pd.DataFrame(rows)
    return rows, df