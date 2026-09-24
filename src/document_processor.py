import io
import re
from typing import List, Dict, Any
import pdfplumber
from pypdf import PdfReader


def extract_pages_from_pdf(pdf_file, doc_name: str = "financial_report.pdf") -> List[Dict[str, Any]]:
    """Extracts raw text and table structures from a PDF stream or path."""
    pages_data = []

    if isinstance(pdf_file, bytes):
        pdf_stream = io.BytesIO(pdf_file)
    elif hasattr(pdf_file, "read"):
        pdf_stream = io.BytesIO(pdf_file.read())
    else:
        pdf_stream = pdf_file

    with pdfplumber.open(pdf_stream) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            text = page.extract_text() or ""
            
            # Extract raw tables
            raw_tables = page.extract_tables() or []
            structured_tables = []
            
            for t in raw_tables:
                cleaned_rows = []
                for row in t:
                    if any(cell is not None and str(cell).strip() != "" for cell in row):
                        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                        cleaned_rows.append(cleaned_row)
                if len(cleaned_rows) > 1:
                    structured_tables.append(cleaned_rows)
            
            pages_data.append({
                "page": page_num,
                "text": text,
                "tables": structured_tables,
                "document": doc_name
            })
            
    return pages_data


def create_naive_chunks(
    pages_data: List[Dict[str, Any]], 
    chunk_size: int = 800, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """Standard sliding-window text chunking across pages without table awareness."""
    naive_chunks = []
    
    for p in pages_data:
        text = p["text"]
        page_num = p["page"]
        doc = p["document"]
        
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            continue
            
        start = 0
        total_len = len(cleaned_text)
        
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk_content = cleaned_text[start:end]
            
            naive_chunks.append({
                "document": doc,
                "page": page_num,
                "chunk_type": "naive_text",
                "text": chunk_content
            })
            
            if end == total_len:
                break
            start += (chunk_size - chunk_overlap)
            
    return naive_chunks


def format_table_as_text(table_rows: List[List[str]]) -> str:
    """Formats 2D table grid into a structured markdown-style table string."""
    if not table_rows:
        return ""
    headers = table_rows[0]
    separator = ["---"] * len(headers)
    
    lines = [" | ".join(headers), " | ".join(separator)]
    for row in table_rows[1:]:
        padded_row = row + [""] * (len(headers) - len(row))
        lines.append(" | ".join(padded_row[:len(headers)]))
    return "\n".join(lines)


def create_table_aware_chunks(
    pages_data: List[Dict[str, Any]],
    narrative_chunk_size: int = 900
) -> List[Dict[str, Any]]:
    """Extracts tables intact as complete single units and chunks surrounding narrative text separately."""
    table_aware_chunks = []
    
    for p in pages_data:
        page_num = p["page"]
        doc = p["document"]
        tables = p.get("tables", [])
        
        # 1. Process table structures
        for t in tables:
            formatted_table = format_table_as_text(t)
            if formatted_table:
                table_aware_chunks.append({
                    "document": doc,
                    "page": page_num,
                    "chunk_type": "financial_table",
                    "text": f"[FINANCIAL TABLE - Page {page_num}]\n{formatted_table}"
                })
        
        # 2. Process narrative text
        text = p.get("text", "")
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            continue
            
        start = 0
        total_len = len(cleaned_text)
        while start < total_len:
            end = min(start + narrative_chunk_size, total_len)
            narrative = cleaned_text[start:end]
            
            table_aware_chunks.append({
                "document": doc,
                "page": page_num,
                "chunk_type": "narrative_text",
                "text": narrative
            })
            if end == total_len:
                break
            start += narrative_chunk_size - 100
            
    return table_aware_chunks
