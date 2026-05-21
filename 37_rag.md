---
title: "Chapter 37 — RAG & Retrieval Patterns"
---

[← Back to Table of Contents](./README.md)

# Chapter 37 — RAG & Retrieval Patterns

> *"A language model that can look things up is more trustworthy than one that must remember everything." — paraphrasing Retrieval-Augmented Generation (Lewis et al., 2020)*

<span class="badge agentic">Agentic</span> <span class="badge mlops">MLOps</span>

---

## Overview

**Retrieval-Augmented Generation (RAG)** grounds LLM outputs in external knowledge by retrieving relevant documents at inference time and injecting them into the prompt. This reduces hallucination, enables up-to-date knowledge without retraining, and provides verifiable citations.

---

## 37.1 RAG Motivation

<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-icon">🧠</div>
    <div class="card-title">Parametric Knowledge</div>
    <div class="card-desc">Knowledge baked into model weights. Stale after training cutoff. No citations. Hard to update.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📚</div>
    <div class="card-title">RAG Knowledge</div>
    <div class="card-desc">Retrieved at query time. Always current. Citable sources. Easily updated by changing the index.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">⚙️</div>
    <div class="card-title">Fine-tuning</div>
    <div class="card-desc">Adapts model style/format. Still needs RAG for factual grounding. Expensive to update frequently.</div>
  </div>
</div>

---

## 37.2 Basic RAG Pipeline

<div class="diagram">
  <div class="diagram-title">Basic RAG Pipeline</div>
  <div class="flow">
    <div class="flow-node accent">User Query</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue">Embed Query<br/><small>text-embedding-3-small / BGE / E5</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node purple">Retrieve Top-K Chunks<br/><small>FAISS / ChromaDB / Weaviate / Pinecone</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node teal">Augment Prompt<br/><small>Insert chunks into context</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node green">Generate Answer<br/><small>LLM conditioned on query + context</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node accent">Response + Citations</div>
  </div>
</div>

---

## 37.3 Dense Retrieval

Dense retrieval encodes both query and documents as embedding vectors. Retrieval is a nearest-neighbour search in embedding space.

```python
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class DenseRetriever:
    """
    Dense vector retriever backed by an in-memory FAISS index.

    Args:
        embed_fn:  callable that converts a list of strings to a 2-D numpy array
        top_k:     number of results to return per query
    """

    embed_fn: Callable[[list[str]], np.ndarray]
    top_k: int = 5
    _docs: list[Document] = field(default_factory=list, repr=False)
    _index: object = field(default=None, repr=False)

    def build_index(self, documents: list[Document]) -> None:
        """Embed all documents and build a FAISS flat-L2 index."""
        import faiss  # pip install faiss-cpu

        self._docs = documents
        texts = [d.text for d in documents]
        embeddings = self.embed_fn(texts).astype(np.float32)
        # Normalise for cosine similarity via inner-product search
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k documents most similar to the query."""
        import faiss

        q_emb = self.embed_fn([query]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self._index.search(q_emb, self.top_k)
        return [
            {"doc": self._docs[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]


def openai_embed_fn(client, model: str = "text-embedding-3-small") -> Callable:
    """Return an embed_fn compatible with DenseRetriever."""

    def embed(texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(input=texts, model=model)
        return np.array([e.embedding for e in response.data])

    return embed


# ---- ChromaDB variant -------------------------------------------------------

def build_chroma_retriever(docs: list[Document], collection_name: str = "rag"):
    """Build a ChromaDB-backed retriever (no FAISS needed)."""
    import chromadb

    client = chromadb.Client()
    col = client.get_or_create_collection(collection_name)
    col.add(
        ids=[d.id for d in docs],
        documents=[d.text for d in docs],
        metadatas=[d.metadata for d in docs],
    )

    def retrieve(query: str, top_k: int = 5) -> list[dict]:
        results = col.query(query_texts=[query], n_results=top_k)
        return [
            {"id": results["ids"][0][i], "text": results["documents"][0][i],
             "score": 1 - results["distances"][0][i]}
            for i in range(len(results["ids"][0]))
        ]

    return retrieve
```

---

## 37.4 Sparse Retrieval — BM25

BM25 is the classic TF-IDF-based keyword retrieval algorithm. It remains competitive for queries with rare or specific terms.

```python
from dataclasses import dataclass
import math
from collections import Counter

@dataclass
class BM25Retriever:
    """
    Pure-Python BM25 retriever (no external dependency).
    For production, use rank_bm25 or Elasticsearch.
    """

    k1: float = 1.5
    b: float = 0.75
    _docs: list[Document] = field(default_factory=list)
    _idf: dict = field(default_factory=dict)
    _tf: list[dict] = field(default_factory=list)
    _avg_dl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def fit(self, documents: list[Document]) -> None:
        self._docs = documents
        tokenized = [self._tokenize(d.text) for d in documents]
        N = len(documents)
        self._avg_dl = sum(len(t) for t in tokenized) / N if N else 1
        # Term frequencies per document
        self._tf = [Counter(t) for t in tokenized]
        # Inverse document frequencies
        df: dict[str, int] = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = self._tokenize(query)
        tf = self._tf[doc_idx]
        dl = sum(tf.values())
        score = 0.0
        for term in query_terms:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            f = tf.get(term, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numerator / denominator
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scores = [(i, self.score(query, i)) for i in range(len(self._docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"doc": self._docs[i], "score": s}
            for i, s in scores[:top_k]
        ]
```

---

## 37.5 Hybrid Retrieval — RRF Fusion

Reciprocal Rank Fusion (RRF) combines dense and sparse rankings without needing calibrated scores.

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.

    Args:
        rankings: list of doc-id lists, each ordered best-first
        k:        RRF constant (60 is standard)
        top_n:    how many to return
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


@dataclass
class HybridRetriever:
    dense: DenseRetriever
    sparse: BM25Retriever
    top_k: int = 10
    rrf_k: int = 60

    def retrieve(self, query: str) -> list[dict]:
        dense_results = self.dense.retrieve(query)
        sparse_results = self.sparse.retrieve(query)

        dense_ids = [r["doc"].id for r in dense_results]
        sparse_ids = [r["doc"].id for r in sparse_results]

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k, top_n=self.top_k)

        id_to_doc = {r["doc"].id: r["doc"] for r in dense_results + sparse_results}
        return [
            {"doc": id_to_doc[doc_id], "rrf_score": score}
            for doc_id, score in fused
            if doc_id in id_to_doc
        ]
```

---

## 37.6 Reranking with a Cross-Encoder

A cross-encoder jointly encodes query + passage, producing a much better relevance score than bi-encoder similarity — but at higher cost, so it's applied only to top-K candidates.

```python
@dataclass
class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.
    Uses sentence-transformers cross-encoder.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    _model: object = field(default=None, repr=False)

    def __post_init__(self):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        candidates: list of dicts with key 'doc' (Document).
        Returns top_n re-ranked results.
        """
        pairs = [(query, c["doc"].text) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            {**cand, "rerank_score": float(score)}
            for cand, score in ranked[: self.top_n]
        ]
```

---

## 37.7 Context Compression

Long retrieved passages waste context tokens. Compression summarises each chunk to its query-relevant essence.

```python
@dataclass
class ContextCompressor:
    """
    Compress retrieved chunks by extracting only query-relevant sentences.
    Uses an LLM to summarise / filter each chunk.
    """

    llm: Callable[[list[dict]], str]
    max_tokens_per_chunk: int = 200

    def compress(self, query: str, chunks: list[str]) -> list[str]:
        compressed = []
        for chunk in chunks:
            prompt = (
                f"Query: {query}\n\n"
                f"Document chunk:\n{chunk}\n\n"
                f"Extract and return only the sentences from the chunk that are "
                f"directly relevant to answering the query. "
                f"Return nothing if the chunk is not relevant."
            )
            result = self.llm([{"role": "user", "content": prompt}])
            if result.strip():
                compressed.append(result.strip())
        return compressed

    def build_context(self, query: str, chunks: list[str]) -> str:
        compressed = self.compress(query, chunks)
        return "\n\n---\n\n".join(compressed)
```

---

## 37.8 Self-RAG

Self-RAG (Asai et al., 2023) trains the model to decide *whether* to retrieve and to reflect on the quality of retrieved content using special tokens (`[Retrieve]`, `[ISREL]`, `[ISSUP]`, `[ISUSE]`).

<div class="diagram">
  <div class="diagram-title">Self-RAG Decision Flow</div>
  <div class="flow">
    <div class="flow-node accent">Input</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node blue wide">Retrieve needed?<br/><small>[Retrieve] token prediction</small></div>
    <div class="flow-h">
      <div class="flow-node green">No → Generate directly</div>
      <div class="flow-node purple">Yes → Retrieve passages</div>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node teal wide">For each passage: ISREL? ISSUP?<br/><small>Reflection tokens score passage utility</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange">Select best passage + Generate segment</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node accent">ISUSE? → Final output</div>
  </div>
</div>

---

## 37.9 HyDE — Hypothetical Document Embeddings

HyDE generates a *hypothetical* answer with the LLM and embeds that instead of the raw query. The hypothesis is often closer to relevant documents than the query itself.

```python
@dataclass
class HyDERetriever:
    """
    Hypothetical Document Embeddings retriever.

    Steps:
    1. Generate a hypothetical answer to the query with an LLM.
    2. Embed the hypothetical answer.
    3. Retrieve documents nearest to the hypothesis embedding.
    """

    llm: Callable[[str], str]
    dense_retriever: DenseRetriever

    def retrieve(self, query: str) -> list[dict]:
        # Step 1: generate hypothesis
        hypothesis_prompt = (
            f"Write a concise factual paragraph that would answer the following question. "
            f"It is OK if it is not perfectly accurate — focus on the style and vocabulary "
            f"of a relevant document.\n\nQuestion: {query}"
        )
        hypothesis = self.llm(hypothesis_prompt)
        # Step 2 & 3: embed hypothesis and retrieve
        return self.dense_retriever.retrieve(hypothesis)
```

---

## 37.10 Chunking Strategies

<table class="compare-table">
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Method</th>
      <th>Chunk Size</th>
      <th>Pros</th>
      <th>Cons</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Fixed-size</strong></td>
      <td>Split every N tokens</td>
      <td>256–512 tokens</td>
      <td>Simple, predictable</td>
      <td>Cuts mid-sentence</td>
      <td>Quick prototypes</td>
    </tr>
    <tr>
      <td><strong>Sentence</strong></td>
      <td>NLTK / spaCy sentence boundaries</td>
      <td>1-3 sentences</td>
      <td>Preserves meaning units</td>
      <td>Uneven lengths</td>
      <td>Q&amp;A, factoid retrieval</td>
    </tr>
    <tr>
      <td><strong>Semantic</strong></td>
      <td>Split on embedding-space discontinuities</td>
      <td>Variable</td>
      <td>Topic-coherent chunks</td>
      <td>Slow, requires embeddings</td>
      <td>Long-form documents</td>
    </tr>
    <tr>
      <td><strong>Hierarchical</strong></td>
      <td>Parent=section, child=paragraph</td>
      <td>Multi-level</td>
      <td>Context at multiple granularities</td>
      <td>Complex indexing</td>
      <td>Technical docs, books</td>
    </tr>
    <tr>
      <td><strong>Proposition</strong></td>
      <td>LLM extracts atomic factual claims</td>
      <td>1 fact per chunk</td>
      <td>Maximal precision</td>
      <td>Expensive preprocessing</td>
      <td>High-precision factoid QA</td>
    </tr>
  </tbody>
</table>

---

## 37.11 Multi-Vector / Parent-Child Retrieval

Index *small* child chunks for precise retrieval, but return the *parent* chunk for richer context.

```python
from dataclasses import dataclass, field

@dataclass
class ParentChildRetriever:
    """
    Index child chunks for search, but return parent chunk to the LLM.
    This improves both retrieval precision and generation context quality.
    """

    embed_fn: Callable[[list[str]], np.ndarray]
    child_chunk_size: int = 128   # tokens
    parent_chunk_size: int = 512  # tokens
    top_k: int = 5
    _child_retriever: DenseRetriever = field(default=None, repr=False)
    _parent_map: dict[str, str] = field(default_factory=dict, repr=False)

    def build(self, documents: list[Document]) -> None:
        """Chunk each document into parents then children."""
        child_docs = []
        for doc in documents:
            words = doc.text.split()
            parent_chunks = [
                " ".join(words[i:i + self.parent_chunk_size])
                for i in range(0, len(words), self.parent_chunk_size)
            ]
            for p_idx, parent in enumerate(parent_chunks):
                parent_id = f"{doc.id}_p{p_idx}"
                child_words = parent.split()
                for c_idx in range(0, len(child_words), self.child_chunk_size):
                    child_text = " ".join(child_words[c_idx:c_idx + self.child_chunk_size])
                    child_id = f"{parent_id}_c{c_idx}"
                    child_docs.append(Document(id=child_id, text=child_text))
                    self._parent_map[child_id] = parent  # map child -> parent text

        self._child_retriever = DenseRetriever(embed_fn=self.embed_fn, top_k=self.top_k)
        self._child_retriever.build_index(child_docs)

    def retrieve(self, query: str) -> list[dict]:
        children = self._child_retriever.retrieve(query)
        seen_parents = set()
        results = []
        for hit in children:
            parent_text = self._parent_map.get(hit["doc"].id, hit["doc"].text)
            if parent_text not in seen_parents:
                seen_parents.add(parent_text)
                results.append({"parent_text": parent_text, "child_score": hit["score"]})
        return results
```

---

## 37.12 RAG Evaluation Metrics

<table class="compare-table">
  <thead>
    <tr>
      <th>Metric</th>
      <th>What It Measures</th>
      <th>Formula</th>
      <th>Typical Threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Recall@K</strong></td>
      <td>Fraction of relevant docs in top-K</td>
      <td>|relevant ∩ retrieved| / |relevant|</td>
      <td>&gt; 0.8 for RAG</td>
    </tr>
    <tr>
      <td><strong>Precision@K</strong></td>
      <td>Fraction of retrieved docs that are relevant</td>
      <td>|relevant ∩ retrieved| / K</td>
      <td>&gt; 0.6</td>
    </tr>
    <tr>
      <td><strong>MRR</strong></td>
      <td>Rank of first relevant doc</td>
      <td>1/rank<sub>first relevant</sub></td>
      <td>&gt; 0.7</td>
    </tr>
    <tr>
      <td><strong>NDCG@K</strong></td>
      <td>Graded relevance; penalises lower ranks</td>
      <td>DCG / IDCG</td>
      <td>&gt; 0.7</td>
    </tr>
    <tr>
      <td><strong>Answer Faithfulness</strong></td>
      <td>Does answer only use retrieved context?</td>
      <td>LLM-as-judge or NLI</td>
      <td>&gt; 0.9</td>
    </tr>
    <tr>
      <td><strong>Answer Relevance</strong></td>
      <td>Does answer address the query?</td>
      <td>LLM-as-judge or ROUGE-L</td>
      <td>&gt; 0.8</td>
    </tr>
  </tbody>
</table>

---

## 37.13 Full Hybrid RAG Pipeline Flow

<div class="diagram">
  <div class="diagram-title">Production RAG Pipeline</div>
  <div class="flow">
    <div class="flow-node accent wide">User Query</div>
    <div class="flow-arrow">↓</div>
    <div class="flow-h">
      <div class="flow-node blue">Dense Retriever<br/><small>FAISS / Chroma</small></div>
      <div class="flow-arrow green">⇔ parallel ⇔</div>
      <div class="flow-node teal">Sparse BM25<br/><small>keyword matching</small></div>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node purple wide">RRF Fusion<br/><small>merge ranked lists</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node orange wide">Cross-Encoder Reranker<br/><small>top-20 → top-5</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node pink">Context Compressor<br/><small>optional: summarise chunks</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node green wide">LLM Generation<br/><small>query + compressed context → answer</small></div>
    <div class="flow-arrow">↓</div>
    <div class="flow-node accent">Answer + Source Citations</div>
  </div>
</div>

<div class="callout warn">
  <div class="callout-icon">⚠️</div>
  <div class="callout-body">
    <strong>Lost-in-the-Middle problem:</strong> LLMs attend poorly to context in the middle of long prompts. Place the most relevant chunks at the <em>beginning</em> or <em>end</em> of the context window, not in the middle.
  </div>
</div>

---

*Last updated: May 2026*
