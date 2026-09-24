import re
from typing import List, Set


def normalize_financial_numbers(text: str) -> Set[float]:
    """Extracts numerical figures from text, standardizing millions/billions and signs."""
    if not text:
        return set()
        
    pattern = r'(?:\(?[\$€£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(million|billion|b|m)?\)?|\((\d+(?:\.\d+)?)\))'
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    
    numbers = set()
    for m in matches:
        full_match = m.group(0).strip()
        num_part = m.group(1) or m.group(3)
        unit = (m.group(2) or "").lower()
        
        if not num_part:
            continue
            
        val = float(num_part.replace(",", ""))
        
        if full_match.startswith("(") and full_match.endswith(")"):
            val = -val
            
        if unit in ["billion", "b"]:
            val *= 1000.0
            
        numbers.add(round(val, 2))
        
    return numbers


def calculate_numerical_accuracy(generated_answer: str, reference_answer: str) -> float:
    """Evaluates whether key financial figures from the reference answer exist in the generated text."""
    ref_nums = normalize_financial_numbers(reference_answer)
    if not ref_nums:
        return 1.0 if reference_answer.lower() in generated_answer.lower() else 0.0
        
    gen_nums = normalize_financial_numbers(generated_answer)
    matches = ref_nums.intersection(gen_nums)
    
    return 1.0 if len(matches) == len(ref_nums) else (len(matches) / len(ref_nums))


def calculate_source_page_recall(retrieved_pages: List[int], expected_pages: List[int]) -> float:
    """Calculates Recall@5 for unique source pages."""
    if not expected_pages:
        return 1.0
    expected_set = set(expected_pages)
    retrieved_set = set(retrieved_pages[:5])
    
    hits = expected_set.intersection(retrieved_set)
    return float(len(hits) / len(expected_set))


def calculate_mrr(retrieved_pages: List[int], expected_pages: List[int]) -> float:
    """Calculates Reciprocal Rank of the first retrieved chunk containing an expected page."""
    if not expected_pages:
        return 1.0
    expected_set = set(expected_pages)
    
    for rank_idx, page in enumerate(retrieved_pages):
        if page in expected_set:
            return 1.0 / (rank_idx + 1)
            
    return 0.0
