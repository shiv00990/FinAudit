from pathlib import Path
from typing import List, Dict, Any, Union
import json
import numpy as np
import faiss


class FAISSRetriever:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.chunks: List[Dict[str, Any]] = []
        self.index: Union[faiss.IndexFlatIP, None] = None
        self.dimension: int = 384

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Encodes chunks, normalizes vectors, and stores them in FAISS IndexFlatIP."""
        if not chunks:
            raise ValueError("Cannot build index with empty chunks list.")
            
        self.chunks = chunks
        texts = [c.get("text", "") for c in chunks]
        
        raw_embeddings = self.embedding_model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        embeddings = raw_embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Encodes query, normalizes it, and retrieves top-k chunks with cosine scores."""
        if self.index is None or not self.chunks:
            return []
            
        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []
            
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["retrieval_score"] = float(scores[0][rank])
            chunk_copy["rank"] = rank + 1
            results.append(chunk_copy)
            
        return results

    def save_index(self, index_path: Union[str, Path], chunks_path: Union[str, Path]) -> None:
        """Persists the FAISS index and chunk metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path: Union[str, Path], chunks_path: Union[str, Path]) -> None:
        """Loads index and chunks metadata from disk."""
        self.index = faiss.read_index(str(index_path))
        self.chunks = self.load_chunks(chunks_path)
        self.dimension = self.index.d

    @staticmethod
    def load_chunks(chunks_path: Union[str, Path]) -> List[Dict[str, Any]]:
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
