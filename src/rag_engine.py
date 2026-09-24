import os
import time
from typing import List, Dict, Any, Tuple
from google import genai
from google.genai.errors import APIError

# Flash-Lite provides much higher availability and lower latency on the free tier
PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = PRIMARY_MODEL


def get_gemini_client(api_key: str = None) -> genai.Client:
    """Initializes Google GenAI client from explicit key or environment variable."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing GOOGLE_API_KEY. Provide it via Streamlit Secrets or Environment.")
    return genai.Client(api_key=key)


def build_context_prompt(question: str, retrieved_chunks: List[Dict[str, Any]], approach_name: str) -> str:
    """Builds token-safe grounded context prompt containing explicit page citations."""
    context_blocks = []
    
    for i, c in enumerate(retrieved_chunks):
        doc = c.get("document", "Doc")
        page = c.get("page", "Unknown")
        c_type = c.get("chunk_type", "Text")
        score = c.get("retrieval_score", 0.0)
        text = c.get("text", "").strip()
        
        truncated_text = text[:1500]
        context_blocks.append(
            f"--- SOURCE {i+1} [Doc: {doc} | Page: {page} | Type: {c_type} | Score: {score:.4f}] ---\n{truncated_text}"
        )
        
    context_text = "\n\n".join(context_blocks)
    
    prompt = f"""You are a strict financial auditor evaluating: {approach_name}.
Answer the question below using ONLY the provided context excerpts.

AUDIT RULES:
1. Do NOT use outside general knowledge.
2. If the context does not contain the answer, say: 'Context is insufficient to answer this question.'
3. Numerical Precision: State numbers and fiscal years exactly. DO NOT round or drop units (e.g., preserve '$', 'million', 'billion', '%').
4. Always list the supporting page numbers under 'Supporting Pages:'.

CONTEXT:
{context_text}

QUESTION:
{question}

Provide your answer in this format:
Answer: <concise, accurate financial statement with exact values and years>
Supporting Pages: Page <X>, Page <Y>
"""
    return prompt


def generate_answer(
    client: genai.Client,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    approach_name: str = "Standard RAG",
    model_name: str = GEMINI_MODEL,
    max_retries: int = 3
) -> Tuple[str, float, bool]:
    """
    Executes answer generation with model fallback and exponential backoff on 503 errors.
    Returns: (answer_text, generation_latency, is_success)
    """
    if not retrieved_chunks:
        return "No relevant context found to generate an answer.", 0.0, False
        
    prompt = build_context_prompt(question, retrieved_chunks, approach_name)
    
    start_time = time.perf_counter()
    models_to_try = [model_name, FALLBACK_MODEL] if model_name != FALLBACK_MODEL else [model_name]
    
    for current_model in models_to_try:
        delay = 1.5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt
                )
                gen_latency = time.perf_counter() - start_time
                
                if response.text and response.text.strip():
                    return response.text.strip(), gen_latency, True
                else:
                    return "Model returned an empty response. Verify context sufficiency.", gen_latency, False
                    
            except APIError:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break  # try next model in fallback list
            except Exception as e:
                gen_latency = time.perf_counter() - start_time
                return f"Generation request encountered an error: {str(e)}", gen_latency, False
                
    gen_latency = time.perf_counter() - start_time
    error_msg = (
        "Gemini temporarily unavailable (High Demand/503). "
        "The document retrieval completed successfully, but answer generation "
        "could not be completed. Please try again."
    )
    return error_msg, gen_latency, False
