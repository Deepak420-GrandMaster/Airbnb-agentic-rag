# 🏠 US Airbnb Travel Agent — Agentic RAG

An AI travel assistant that answers natural-language questions about Airbnb listings across **5 major US cities**. It combines **semantic search** with **structured metadata filtering**, and returns **grounded, cited answers** — no hallucinated listings.

Built as an end-to-end applied-AI project: data pipeline → vector retrieval → LLM agent → evaluation → deployment.

---

## 🎥 Demo

> _Live app:_ **[link added after deployment]**

Example question:
> *"cheap private room in New Orleans under $80"*

The agent extracts filters (`price < $80`, `New Orleans`, `Private room`), retrieves matching listings, and writes a grounded recommendation citing each source.

---

## 🧠 How it works
User question
│
▼
[1] LLM extracts filters   → price, city, room type, vibe
│
▼
[2] Hybrid retrieval       → metadata filter (ChromaDB) + semantic search (embeddings)
│
▼
[3] Grounded generation    → LLM answers using ONLY retrieved listings, with citations
│
▼
Cited recommendation

**Why "agentic"?** The model doesn't just embed-and-match. It *decides* which structured filters apply to a plain-English question, then constrains retrieval accordingly — fixing the classic RAG failure where semantic search ignores hard constraints like price or city.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Evaluation | Custom faithfulness harness (LLM-as-judge) |
| App | Streamlit |
| Data | US Airbnb Open Data — 5 cities |

---

## 📊 Evaluation

Faithfulness measures how strictly each answer stays grounded in retrieved listings (no invented details).

| Metric | Score |
|--------|-------|
| Faithfulness (baseline) | 0.60 |
| Faithfulness (after prompt hardening) | **0.92** |

**Key engineering finding:** baseline faithfulness was dragged down by the LLM *embellishing* answers with qualities not in the data ("great location", "highly recommended"). Constraining the generation prompt to state only retrieved facts raised faithfulness from **0.60 → 0.92** — a concrete example of evaluation-driven iteration.

---

## 🧹 Data Engineering Notes

Built a pipeline from raw open data to a clean, balanced index:
- Started from **232k** US listings → filtered to 5 target cities → **94k** clean rows
- Removed invalid rows: **$0/night** listings, **300-night minimums**
- Balanced sample of **3,000 listings per city** → **15,000 indexed**
- Cities: New York City, Los Angeles, San Francisco, Chicago, New Orleans

---

## 🚀 Run locally

```bash
# clone
git clone https://github.com/Deepak420-GrandMaster/Airbnb-agentic-rag.git
cd Airbnb-agentic-rag

# environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run
streamlit run app.py
```

You'll need a free Groq API key from [console.groq.com](https://console.groq.com).

---

## 🔭 Possible extensions

- Add review-text retrieval for richer semantic matching
- Multi-turn conversation memory
- Expand to international cities
- Add a reranker for retrieval precision
- Integrate hotels / flights as additional agent tools

---

## 👤 Author

**Deepak Prajapati**
[LinkedIn](https://www.linkedin.com/in/deepak-prajapati-695963204) · [GitHub](https://github.com/Deepak420-GrandMaster)
