import os
import time
from typing import List, Dict, Any, Tuple
from groq import Groq

# Universally accessible model on all Groq tiers (560+ tokens/sec)
GROQ_MODEL = "llama-3.1-8b-instant"


def get_llm_client(api_key: str = None) -> Groq:
    """Initializes Groq client safely from key or environment variable."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing GROQ_API_KEY. Provide it via Streamlit Secrets or Environment.")
    return Groq(api_key=key)


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
    client: Groq,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    approach_name: str = "Standard RAG",
    model_name: str = GROQ_MODEL,
    max_retries: int = 3
) -> Tuple[str, float, bool]:
    """
    Executes answer generation using Groq LLaMA with retry handling.
    Returns: (answer_text, generation_latency, is_success)
    """
    if not retrieved_chunks:
        return "No relevant context found to generate an answer.", 0.0, False
        
    prompt = build_context_prompt(question, retrieved_chunks, approach_name)
    
    start_time = time.perf_counter()
    delay = 1.0
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst that answers with strict factual and numerical precision."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_name,
                temperature=0.0
            )
            gen_latency = time.perf_counter() - start_time
            content = chat_completion.choices[0].message.content
            
            if content and content.strip():
                return content.strip(), gen_latency, True
            else:
                return "Model returned empty text.", gen_latency, False
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            gen_latency = time.perf_counter() - start_time
            return f"Groq Generation error: {str(e)}", gen_latency, False
            
    gen_latency = time.perf_counter() - start_time
    return "Generation timed out after retries.", gen_latency, False
