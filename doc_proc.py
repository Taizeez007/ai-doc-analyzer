import os
import json
from pathlib import Path
from typing import Optional, Dict, Union, Any
from enum import Enum

import torch
from pydantic import BaseModel, Field

from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    import docx
except ImportError:
    docx = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from llama_index.llms.ollama import Ollama

#OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama2")
DEFAULT_INPUT_DIR = os.environ.get("INPUT_DIR", "./doc_folder")

from pydantic import BaseModel, Field
from enum import Enum
import torch
from llama_index.embeddings.ollama import OllamaEmbedding


DEVICE='cuda' if torch.cuda.is_available() else "cpu"
# Define the allowed values for approval_status
class ApprovalStatus(str, Enum):
    YES = "yes"
    NO = "no"
    MAYBE = "maybe"
"""
q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.float16,
    )
    """


# Define the desired output structure (Pydantic Model)
class ApplicantInfo(BaseModel):
    """Structured information about an applicant."""
    name: str = Field(description="The full name of the applicant.")
    dti: Optional[float] = Field(description="The Debt-to-Income (DTI) ratio, if available.")
    credit_score: Optional[int] = Field(description="The applicant's credit score, if available.")
    reason: str = Field(description="The core reason for the approval status.")
    approval_status: ApprovalStatus = Field(description="The final approval status, one of 'yes', 'no', or 'maybe'.")

# Build index
def build_index(input_dir: str, model: Union[str, Any]):
    print(">>>>BUILD INDEX")

    llm = HuggingFaceLLM(
        model_name=model,     # <--- change if needed
        tokenizer_name="google/gemma-3-1b-it",
        device_map="auto",
        context_window=2048,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
    )
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Read documents ----
    reader = SimpleDirectoryReader(input_dir, recursive=True)
    documents = reader.load_data()

    # ---- Build vector index ----
    index = GPTVectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )
    print(">>>> INDEX BUILD COMPLETE")

    return index, llm

import re

def extract_fields_from_document(llm, doc_text: str) -> Union[Any,Dict[str,str]]:
     
    prompt = f"""You are an information extraction engine.
Follow these rules strictly:
1. Extract only information present in the text.
2. Never guess or fabricate values.
3. Output one JSON object per document.
4. Use null for missing values.
5. Do not create multiple entries.
6. Once a document is processed, do not process it again
7. Ensure each output is distinct and corresponds to a single document.
8. Do not regenerate or modify previous outputs
9. Check a document name if processed before. if not processed before, extract info in it.
10. Ensure the number of output is same as number of document
10. Follow strictly this schema for output:
    {{
       "name": "...",
       "dti": ...,
       "credit_score": ...,
       "reason": "...",
       "approval_status": "yes" | "no" | "maybe" |"null"
       }}

    Document:
    \"\"\"{doc_text}\"\"\"

    Extract the fields.
    Output ONLY the JSON object.
    """
    raw_output=None
    try:
        raw_output = llm.complete(prompt).text.strip()

        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        # --- EXTRACT ALL JSON OBJECTS ---
        json_objects = re.findall(r"\{.*?\}", raw_output, flags=re.DOTALL)
        
        results=[]
        # --- CLEANING STEP: remove markdown or code fences ---
        for obj_str in json_objects:
            try:
                data = json.loads(obj_str)
                validated = ApplicantInfo(**data)
                results.append(validated.model_dump())
            except (json.JSONDecodeError, ValueError) as e:
                results.append({"error": str(e), "raw_extraction": obj_str})

        return results
        

    except Exception as e:
        print(f"Extraction failed: {e}")
        return {"error": str(e), "raw_extraction": doc_text if 'raw_output' in locals() else None}
    
