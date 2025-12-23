print(">>>>> APP START")

import os
import argparse

from doc_proc import build_index
from folder_proc import process_local_folder


DEFAULT_INPUT_DIR = os.environ.get("INPUT_DIR", "./doc_folder")
DEFAULT_OUTPUT_FILE = "extracted.json"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract structured information from local documents using LlamaIndex + HuggingFace models.'
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT_DIR,
        help='Local folder containing documents'
    )
    parser.add_argument(
        '--out', '-o',
        default=DEFAULT_OUTPUT_FILE,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    print(">>>> APP START — Using HuggingFace models")

    # --- Build vector store + load HuggingFace LLM ---
    index, llm = build_index(args.input, model="google/gemma-3-1b-it")

    # --- Perform extraction ---
    rows, df = process_local_folder(
        input_dir=args.input,
        llm=llm,
        output_file=args.out
    )

    print(f"\n✅ Successfully processed {len(rows)} documents.")
    print(f"📄 Output saved to: {args.out}")

    print("\n--- DataFrame Head ---")
    if df is not None and not df.empty:
        print(df.head().to_string())
    else:
        print("⚠️ No data extracted.")
