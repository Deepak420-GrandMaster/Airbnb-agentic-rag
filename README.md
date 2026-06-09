# 🏠 NYC Airbnb Travel Agent — Agentic RAG

An AI travel assistant that answers natural-language questions about NYC Airbnb listings. It combines **semantic search** with **structured metadata filtering**, and returns **grounded, cited answers** — no hallucinated listings.

Built as an end-to-end applied-AI project: data cleaning → vector retrieval → LLM agent → evaluation → deployment.

---

## 🎥 Demo

> _Live app:_ **[link added after deployment]**



Example question:
> *"cheap cozy private room in Brooklyn under $80"*

The agent extracts filters (`price < $80`, `Brooklyn`, `Private room`), retrieves matching listings, and writes a grounded recommendation citing each source.

---

## 🧠 How it works
User question
│
▼
[1] LLM extracts filters   → price, neighbourhood, room type, vibe
│
▼
[2] Hybrid retrieval       → metadata filter (ChromaDB) + semantic search (embeddings)
│
▼
[3] Grounded generation    → LLM answers using ONLY retrieved listings, with citations
│
▼
Cited recommendation

**Why "agentic"?** The model doesn't just embed-and-match. It *decides* which structured filters apply to a plain-English question, then constrains retrieval accordingly — fixing the classic RAG failure where semantic search ignores hard constraints like price.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| LLM | Llama 3.3 70B (via Groq) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Evaluation | RAGAS (faithfulness, answer relevancy) |
| App | Streamlit |
| Data | NYC Airbnb Open Data (~49k listings, cleaned to ~4.9k) |

---

## 📊 Evaluation

Measured with RAGAS on a held-out question set:

| Metric | Score |
|--------|-------|
| Answer Relevancy | 0.88 |
| Faithfulness (baseline) | 0.60 |
| Faithfulness (after prompt hardening) | _updating_ |

**Key engineering finding:** baseline faithfulness was dragged down by the LLM *embellishing* answers with qualities not in the data ("great location", "highly recommended"). Constraining the generation prompt to state only retrieved facts measurably improved faithfulness — a concrete example of evaluation-driven iteration.

---

## 🧹 Data Engineering Notes

Raw Airbnb data contained quality issues handled before serving:
- Listings priced at **$0/night** (invalid) → removed
- Listings with **300-night minimums** (not realistic short stays) → removed
- Missing review fields → filled with 0
- Final clean set: ~4,900 listings

---

## 🚀 Run locally

```bash
# clone
git clone https://github.com/Deepak420-GrandMaster/airbnb-agentic-rag.git
cd airbnb-agentic-rag

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

- Add review-text retrieval (richer semantic matching)
- Multi-turn conversation memory
- Expand beyond NYC to multi-city data
- Swap in a reranker for retrieval precision

---

## 👤 Author

**Deepak Prajapati** 
[LinkedIn](https://www.linkedin.com/in/deepak-prajapati-695963204) · [GitHub](https://github.com/Deepak420-GrandMaster)