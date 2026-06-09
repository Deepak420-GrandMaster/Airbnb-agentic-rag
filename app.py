import streamlit as st
import pandas as pd
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Airbnb Travel Agent", page_icon="🏠", layout="wide")

# ---------- LOAD (cached so it runs once) ----------
@st.cache_resource
def load_everything():
    df = pd.read_csv("airbnb_clean.csv")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("airbnb")
    except:
        pass
    collection = chroma_client.create_collection(name="airbnb")

    documents = df["document"].tolist()
    ids = [str(i) for i in df.index]
    metadatas = df[[
        "name","neighbourhood_group","neighbourhood",
        "room_type","price","minimum_nights","number_of_reviews"
    ]].to_dict("records")

    embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return df, embed_model, collection

# ---------- AGENT FUNCTIONS ----------
def extract_filters(client, user_question):
    system_prompt = """You extract search filters from travel questions about NYC Airbnb listings.
Return ONLY valid JSON with these keys (use null if not mentioned):
- max_price: number or null
- neighbourhood_group: one of ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"] or null
- room_type: one of ["Entire home/apt","Private room","Shared room"] or null
- search_text: a short phrase capturing the vibe/description
Return nothing but the JSON object."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_question}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except:
        return {"max_price":None,"neighbourhood_group":None,"room_type":None,"search_text":user_question}

def retrieve_filtered(embed_model, collection, query, k=5, max_price=None, neighbourhood_group=None, room_type=None):
    conditions = []
    if max_price is not None:
        conditions.append({"price":{"$lte":max_price}})
    if neighbourhood_group is not None:
        conditions.append({"neighbourhood_group":{"$eq":neighbourhood_group}})
    if room_type is not None:
        conditions.append({"room_type":{"$eq":room_type}})
    where = None
    if len(conditions)==1:
        where = conditions[0]
    elif len(conditions)>1:
        where = {"$and":conditions}
    q_emb = embed_model.encode([query]).tolist()
    return collection.query(query_embeddings=q_emb, n_results=k, where=where)

def airbnb_agent(client, embed_model, collection, user_question, k=5):
    filters = extract_filters(client, user_question)
    results = retrieve_filtered(
        embed_model, collection,
        query=filters.get("search_text") or user_question,
        k=k,
        max_price=filters.get("max_price"),
        neighbourhood_group=filters.get("neighbourhood_group"),
        room_type=filters.get("room_type")
    )
    docs = results["documents"][0]
    if not docs:
        return {"answer":"No listings matched those filters. Try relaxing price or location.",
                "filters":filters, "sources":[]}
    context = "\n".join([f"[{i+1}] {d}" for i,d in enumerate(docs)])
answer_prompt = f"""You are a NYC travel assistant. Recommend listings using ONLY the facts shown.

STRICT RULES:
- Only state facts explicitly present: name, room type, neighbourhood, price, minimum nights, number of reviews.
- Do NOT add adjectives or qualities not in the data unless literally in the listing name.
- Do NOT use backticks, code formatting, or special symbols. Plain text only.
- Cite each listing with [number].

FORMAT your answer EXACTLY like this:
Here are your top options:

- **[1] Listing name** — Room type in Neighbourhood. $X/night, minimum N nights, R reviews.
- **[2] Listing name** — Room type in Neighbourhood. $X/night, minimum N nights, R reviews.
- **[3] Listing name** — Room type in Neighbourhood. $X/night, minimum N nights, R reviews.

End with one short sentence of honest guidance (e.g. if something doesn't fully match).

LISTINGS:
{context}

QUESTION: {user_question}"""

Answer in 2-3 sentences. State only facts from the listings above."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":answer_prompt}],
        temperature=0.3
    )
    return {"answer":resp.choices[0].message.content, "filters":filters, "sources":docs}

# ---------- UI ----------
st.title("🏠 NYC Airbnb Travel Agent")
st.caption("Agentic RAG over 5,000 NYC listings — semantic search + metadata filtering + grounded answers")

api_key = st.text_input("Enter your Groq API key", type="password", help="Get a free key at console.groq.com")

if api_key:
    client = Groq(api_key=api_key)
    with st.spinner("Loading listings + building vector store (first run only)..."):
        df, embed_model, collection = load_everything()
    st.success(f"Ready — {collection.count()} listings loaded")

    question = st.text_input("Ask about a place to stay:",
                             placeholder="e.g. cheap cozy private room in Brooklyn under $80")

    if question:
        try:
            with st.spinner("Thinking..."):
                out = airbnb_agent(client, embed_model, collection, question)
            st.markdown("### Recommendation")
            st.markdown(out["answer"])
            with st.expander("🔍 Filters the agent extracted"):
                st.json(out["filters"])
            with st.expander("📋 Source listings used"):
                for s in out["sources"]:
                    st.write("•", s)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                st.warning("😅 Whoa, slow down traveler! We've used up Groq's free tokens for now. Grab a coffee ☕ and try again in a few minutes (or tomorrow).")
            else:
                st.error(f"Something went wrong: {e}")
else:
    st.info("Enter your Groq API key above to start. Get one free at console.groq.com")