# Tiktok-TechJam

An intelligent conversational search engine built for e-commerce that dynamically routes user intents between open-ended browsing and targeted buying, tracks evolving constraints across turns, and re-ranks dense vector and sparse keyword search results to maximize retrieval accuracy (MRR & Hit Rate@10) while minimizing interaction turns (MTTC).

Standard e-commerce search engines fail in multi-turn dialogues because they treat every query independently, ignoring past context or failing to adapt when users change their minds (e.g., "Actually, I want ASICS instead of Nike"). Furthermore, open-ended user queries often return thousands of irrelevant items, leading to high Mean Turns to Conversion (MTTC).
Our goal was to build a low-latency, dynamic search agent that:
1. Distinguishes between exploratory browsing vs. intent-driven buying.
2. Preserves and updates dialogue slots dynamically without token bloat.
3. Delivers high-precision retrieval using hybrid BM25 + FAISS search backed by lightweight semantic reranking.

Our system is engineered around Four Core Algorithmic Pillars:
🧠 Pillar I: Dual-Track Intent Classification & Hybrid Retrieval
- Dynamic Intent Track: Automatically classifies user state into Browsing Track (high semantic exploration) or Buying Track (high constraint specificity).
- Sparse + Dense Fusion: Combines BM25 (sparse keyword match on title, features, and categories) with FAISS (dense vector similarity using all-MiniLM-L6-v2 embeddings).
- Reciprocal Rank Fusion (RRF): Dynamically shifts fusion weights based on intent track (e.g., giving higher BM25 weight to precise buying terms).

🔄 Pillar II & III: Dynamic Context Programming & State Tracking
- Incremental Slot Accumulation: Tracks context (brand, gender, price, category) across multi-turn interactions.
- Slot Rewriting / Intent Override: Detects user mind-changes (e.g., handling keywords like "instead", "actually") and flushes conflicting slots seamlessly.
- Proactive Guidance Cut-off: Detects candidate pool overload (Over-Generality) on broad queries and proactively generates structured clarification questions to reduce MTTC.
- Context Deduplication: Cleans and deduplicates accumulated query slots before execution, maintaining optimal BM25 scoring.

🎯 Pillar IV: Semantic LLM Reranking
- Filters and re-ranks the candidate pool using strict constraint matching (penalizing brand or gender mismatches) and semantic scoring to ensure target items hit Rank 1, maximizing MRR and Hit Rate@10.

Programming Language: Python 3.10+
Environment: Google Colab
Vector Indexing & Retrieval: FAISS (faiss-cpu)
Keyword Indexing: rank_bm25 (BM25Okapi)
Embeddings Model: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
Data Processing & Handling: NumPy, JSONL / Amazon Reviews 2023 Dataset (Clothing, Shoes & Jewelry)

How to Run?
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
# 2. Install required packages
pip install rank-bm25 faiss-cpu sentence-transformers numpy
# 3. Run the notebook or evaluation script
python agent.py

Limitations & Future Work
- Cross-Encoder Reranking: In future iterations, we plan to fine-tune a local cross-encoder model to capture micro-level attribute semantics (e.g., sole thickness, specific shoe arch support).
- Multi-Modal Retrieval: Incorporating product image embeddings (CLIP) alongside text metadata to handle visual search queries.
- GPU Acceleration: Scaling vector indexing from faiss-cpu to GPU clusters for real-time sub-millisecond retrieval across million-scale product catalogs.
