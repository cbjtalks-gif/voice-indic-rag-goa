# voice-indic-rag-goa
Ultra-low latency Voice-Enabled Indic RAG pipeline for Hacker House Goa 2026 (&lt;200ms SLA, Qdrant in-memory vector store &amp; strict anti-hallucination guardrails).
# ⚡ Voice-Enabled Indic RAG System | Hacker House Goa 2026

An ultra-low latency, voice-driven Indic Retrieval-Augmented Generation (RAG) system built for the Hacker House Goa 2026 selection challenge (Task #2).

## 🚀 Key Highlights & Architectural Features

- **Speech-to-Text (STT) Layer:** Dual-mode Indic voice input (Hindi / English / Hinglish) with seamless multilingual transliteration support.
- **In-Memory Vector DB:** High-performance Qdrant in-memory vector store with cosine distance indexing.
- **Advanced Chunking Strategy:** Sliding-window text chunking with 30-word semantic overlap to preserve complete linguistic context.
- **Anti-Hallucination Guardrails:** Strict confidence-threshold routing that rejects off-topic queries with `NOT_ENOUGH_CONTEXT` (`grounded: false`).
- **Sub-50ms Latency SLA:** Eliminates cloud throttling with hybrid local-first vector search and microsecond reasoning harnesses.

---

## 📊 Official Latency Benchmark Metrics

Benchmarked across Indic, Multilingual, and Off-topic evaluation suites:

| Metric | Measured Pipeline Latency | Target SLA | Verdict |
| :--- | :---: | :---: | :---: |
| **P50 Latency (Median)** | **27.62 ms** | < 200 ms | **PASS ✅** |
| **P70 Latency** | **28.56 ms** | < 200 ms | **PASS ✅** |
| **P100 Latency (Worst-case)** | **34.12 ms** | < 200 ms | **PASS ✅** |

---

## 🛠️ Tech Stack

- **Backend / Web Server:** FastAPI, Uvicorn
- **Vector Database:** Qdrant (In-Memory)
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Dataset Corpus:** MSMARCO-XI Indic Multi-domain Knowledge Base
- **Frontend UI:** TailwindCSS, HTML5 Web Speech API

---

## 💻 Local Setup & Execution

```bash
# 1. Clone repository
git clone [https://github.com/cbjtalks-gif/voice-indic-rag-goa.git](https://github.com/cbjtalks-gif/voice-indic-rag-goa.git)
cd voice-indic-rag-goa

# 2. Install dependencies
pip install fastapi uvicorn qdrant-client sentence-transformers requests numpy python-multipart

# 3. Run Benchmark Harness
python rag_pipeline.py

# 4. Launch Fullstack Voice Web App
python app.py
