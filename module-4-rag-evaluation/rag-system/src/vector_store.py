"""Vector Store implementation using ChromaDB with an in-memory fallback."""
from typing import List, Dict, Any, Optional
import json
import os
import re
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    embedding_functions = None
    CHROMA_AVAILABLE = False


class CodebaseVectorStore:
    """Vector store for code embeddings using ChromaDB or a local fallback."""

    def __init__(
        self,
        collection_name: str = "codebase",
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._fallback_entries: List[Dict[str, Any]] = []
        self._fallback_path = Path(persist_directory) / f"{collection_name}_fallback.json"
        self.backend = "fallback"

        if CHROMA_AVAILABLE:
            try:
                # Initialize ChromaDB with persistence
                self.client = chromadb.PersistentClient(path=persist_directory)

                # Use OpenAI embeddings (can swap for sentence-transformers)
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name="text-embedding-3-small"
                    )
                else:
                    # Fallback to sentence-transformers (free, local)
                    self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )

                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                self.backend = "chroma"
                return
            except Exception as exc:
                print(f"ChromaDB unavailable, using in-memory fallback: {exc}")

        self._load_fallback()

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """Add documents to the vector store."""
        if self.backend == "chroma":
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return

        # Upsert behavior for fallback backend.
        by_id = {entry["id"]: entry for entry in self._fallback_entries}
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            by_id[doc_id] = {
                "id": doc_id,
                "content": doc,
                "metadata": meta,
            }
        self._fallback_entries = list(by_id.values())
        self._persist_fallback()

    def query(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Query the vector store."""
        if n_results <= 0:
            return []

        if self.backend == "chroma":
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )

            formatted = []
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "id": results["ids"][0][i]
                })
            return formatted

        candidates = self._fallback_entries
        if where:
            candidates = [
                entry for entry in candidates
                if all(entry["metadata"].get(k) == v for k, v in where.items())
            ]

        query_tokens = self._tokenize(query)
        scored = []
        for entry in candidates:
            content_tokens = self._tokenize(entry["content"])
            meta_tokens = self._tokenize(entry["metadata"].get("filename", ""))
            similarity = self._jaccard(query_tokens, content_tokens)
            similarity = max(similarity, self._jaccard(query_tokens, meta_tokens))
            scored.append((similarity, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:n_results]
        return [
            {
                "content": entry["content"],
                "metadata": entry["metadata"],
                "distance": round(1 - similarity, 6),
                "id": entry["id"],
            }
            for similarity, entry in top
        ]

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        if self.backend == "chroma":
            return {
                "count": self.collection.count(),
                "name": self.collection.name,
                "backend": "chroma"
            }

        return {
            "count": len(self._fallback_entries),
            "name": self.collection_name,
            "backend": "fallback"
        }

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self.backend == "chroma":
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            return

        self._fallback_entries = []
        self._persist_fallback()

    def _load_fallback(self) -> None:
        """Load fallback entries from disk if present."""
        if not self._fallback_path.exists():
            return

        try:
            self._fallback_entries = json.loads(self._fallback_path.read_text(encoding="utf-8"))
        except Exception:
            self._fallback_entries = []

    def _persist_fallback(self) -> None:
        """Persist fallback entries to disk for repeatable runs."""
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path.write_text(
            json.dumps(self._fallback_entries, ensure_ascii=True),
            encoding="utf-8"
        )

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize text into lowercase alphanumeric terms."""
        return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Compute Jaccard similarity for two token sets."""
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)
