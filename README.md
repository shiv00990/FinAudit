# FinAudit RAGBench

Automated Benchmarking Platform for Financial Retrieval-Augmented Generation (RAG) Systems.

## Overview
FinAudit RAGBench benchmarks standard **Naive text chunking** against a specialized **Table-Aware retrieval pipeline**. It evaluates numerical precision, fiscal period grounding, source-page recall@5, MRR, and end-to-end latency using official financial filings (such as the Apple FY25 10-K).

## Features
- **Naive vs. Table-Aware RAG:** Side-by-side comparison on identical financial questions.
- **Table Structure Retention:** Preserves multi-year columns and header-row relationships intact.
- **Resilient Gemini Generation:** Exponential backoff retry handler built specifically for Gemini 503 load errors.
- **Audited Metrics:** Numerical Accuracy, Source-Page Recall@5, MRR, and separate retrieval vs. generation latency tracking.
- **Five Dashboard Views:**
  1. Overview & Problem Definition
  2. Benchmark Metrics & Comparative Plotly Charts
  3. Interactive Financial Q&A with live PDF upload & evidence inspection
  4. Granular Query-Level Analysis
  5. Audited Failure Analysis

## Project Structure
```text
FinAudit-RAGBench/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── benchmark_questions.json
├── results/
│   ├── final_metrics.csv
│   ├── final_benchmark_results.csv
│   └── failure_analysis.csv
└── src/
    ├── __init__.py
    ├── document_processor.py
    ├── retrieval.py
    ├── rag_engine.py
    └── benchmark.py
