# Tiktok-TechJam

# Dual-Track Hybrid Agent for Multi-Turn Conversational Commerce

An end-to-end conversational e-commerce search engine that dynamically tracks multi-turn user intent, adapts to mid-dialogue preference updates, and leverages hybrid sparse-dense retrieval backed by semantic reranking to optimize product discovery.


## Executive Summary & Problem Statement

Traditional e-commerce search engines treat every query as an isolated event. They fail in multi-turn dialogues because they cannot retain historical context, struggle to handle user mind-changes (e.g., "Actually, I want ASICS instead of Nike"), and often flood users with broad candidate pools.

Our system solves this by introducing a Dual-Track Intent Routing Engine with Dynamic Context Programming. It balances open-ended exploratory browsing and constraint-driven purchasing while optimizing for high catalog recall, precision rank placement, and minimal dialogue turns.

## Method & Algorithmic Architecture

Our architecture operates across Four Core Technical Pillars:

### Pillar I: Dual-Track Intent Classification & Hybrid Retrieval

* Intent Tracks: Automatically classifies user utterances into Browsing Track (high exploration, semantic focus) or Buying Track (high constraint specificity).
* Sparse Keyword Index (BM25): Index built using rank_bm25 (BM25Okapi) over product titles, categories, features, and text descriptions for precise keyword and model matching.
* Dense Vector Index (FAISS): Implements cosine similarity search over IndexFlatIP using dense embeddings.
* Reciprocal Rank Fusion (RRF): Fuses candidate streams from BM25 and FAISS dynamically using RRF scoring

Track-adaptive weighting shifts toward BM25 (1.5 : 0.5) during Buying Track to enforce hard keyword constraints, and balances them (1.0 : 1.0) during Browsing Track.

### Pillar II & III: Dynamic Context Programming & State Tracking

* Incremental Slot Accumulation: Dynamically updates dialogue state slots across turns (gender, brand, category, max_price, color).
* Slot Rewriting / Intent Override: Detects preference shifts (keywords: "instead", "actually", "change to") and flushes conflicting slots seamlessly.
* Proactive Guidance Cutoff: Monitors candidate pool volume. If candidate count exceeds the over-generality threshold during exploratory phases, the agent generates structured clarification prompts to lower MTTC.
* Query Deduplication: Cleans and normalizes accumulated slot history prior to sparse retrieval to eliminate token redundancy and preserve BM25 scoring fidelity.

### Pillar IV: Constraint-Aware LLM Reranking

Reranks the top 50 fused candidates through zero-shot semantic matching and hard-constraint validation. Applies multiplicative penalty multipliers for brand, gender, or budget mismatches to elevate exact product matches to Rank 1.


## Model Choice & Technical Rationale

* Dense Text Encoder: `sentence-transformers/all-MiniLM-L6-v2`
Rationale: Maps catalog items into a 384-dimensional dense vector space. Chosen for its fast inference speed and high performance on semantic similarity benchmarks under low latency constraints.
* Dense Vector Index: `faiss-cpu` (`IndexFlatIP`)
Rationale: Provides high-throughput, inner-product similarity search for dense vector recall.
* Sparse Retriever: `rank_bm25` (`BM25Okapi`)
Rationale: In-memory lexical search engine providing keyword matching for product titles, brands, and SKUs.
* State Tracking & Reranker: Custom Rule-Informed LLM Reranker
Rationale: Combines structured constraint validation with semantic keyword alignment to score and sort candidates without model API latency bottlenecks.

---

## Evaluation & Benchmark Metrics

The system is evaluated against three core operational metrics:

1. Coverage (Hit Rate@K): Measures catalog recall and boundary capability during retrieval stage.
2. Precision (MRR / Top-K Hit Rate): Measures the system's accuracy in placing the target purchased item at Rank 1.
3. Efficiency (MTTC - Mean Turns to Conversion): Tracks average interaction turns required to complete product selection, heavily rewarding proactive guidance that minimizes user cognitive load.

### Benchmark Output Summary
==========================================================
 FINAL OFFICIAL SYSTEM METRICS SUMMARY
==========================================================
 1. Coverage (Hit Rate@10): 100.00%  (2/2 Sessions)
 2. Precision (MRR@10): 1.0000
 3. Efficiency (MTTC): 2.00 Turns to Conversion
==========================================================

## Tech Stack & Dependencies
* Language: Python 3.10+
* Frameworks & Libraries: sentence-transformers, faiss-cpu, rank_bm25, numpy, torch
* Data Format: Amazon Reviews 2023 Dataset (Clothing, Shoes & Jewelry catalog JSONL schema)

## Installation & Running the Pipeline

# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install rank-bm25 faiss-cpu sentence-transformers numpy torch

# 3. Execute the interactive pipeline and evaluation suite
python agent.py

## Limitations & Future Work
* Fine-Tuned Cross-Encoders: Integrating a micro-tuned cross-encoder model to capture nuanced product specifications (e.g., shoe arch support, material composition).
* Multi-Modal Vector Search: Incorporating vision-language models (e.g., CLIP) to execute joint visual and textual candidate retrieval.
* GPU Index Acceleration: Scaling FAISS vector search to GPU clusters (faiss-gpu) for millisecond retrieval across multi-million product catalogs.
