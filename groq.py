# vmitra_chatbot_streamlit.py

import streamlit as st
import json, os, requests, random
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ────────────────────────────────
# 0) Load .env
# ────────────────────────────────
load_dotenv()  # loads variables from .env into environment

# ────────────────────────────────
# 1) Streamlit page setup
# ────────────────────────────────
st.set_page_config(
    page_title="V-Mitra Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("🤝 V-Mitra Domain Chatbot")

# ────────────────────────────────
# 2) Groq API config (env var)
# ────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Please set the GROQ_API_KEY environment variable in your .env file.")
    st.stop()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL    = "llama3-8b-8192"

# ────────────────────────────────
# 3) Load V-Mitra KB
# ────────────────────────────────
kb_path = "vmitra_knowledge_base.json"
try:
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)["v_mitra_knowledge_base"]
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()

# ────────────────────────────────
# 4) Extract sample QA for suggestions
# ────────────────────────────────
example_qas = []
for sec in kb["sections"]:
    if sec.get("id") == "sample_intents_entities":
        for intent in sec["intents"]:
            for ex in intent["examples"]:
                example_qas.append((ex["question"], ex["answer"]))

questions, answers = zip(*example_qas)

# ────────────────────────────────
# 5) Embed questions for retrieval
# ────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_embedder():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs  = np.array(model.encode(questions, convert_to_tensor=False))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return model, embs

embedder, embs = init_embedder()

# ────────────────────────────────
# 6) System‐level prompt
# ────────────────────────────────
system_prompt = """
You are V-MitraBot, the official domain expert for the V-Mitra citizen audit app.
Your purpose is to help users with V-Mitra: registration, searching consumers,
reporting irregularities, tracking status, appeals, rewards, privacy, and all user-guide topics.

RULES:
1. DOMAIN ONLY: Only answer V-Mitra questions. If asked about anything else, respond:
   “I'm here to help with V-Mitra app questions—could you please reframe your query?”
2. CONTEXT USAGE: Always use the provided Context Q/A. First restate the context in 1–2 lines,
   then answer.
3. ALWAYS ANSWER: For any valid V-Mitra query, give an answer. If unsure, preface with “I believe…”
   and cite the context.
4. SAFETY FIRST: If the request is harmful, unsafe, or violates policy, refuse:
   “I’m sorry, but I can’t help with that.”
5. STYLE: Use simple, step-by-step instructions. Prefix steps “Step 1, Step 2, …”.
"""

# ────────────────────────────────
# 7) Session state for query & suggestions
# ────────────────────────────────
if "query" not in st.session_state:
    st.session_state["query"] = ""

if "initial_suggestions" not in st.session_state:
    st.session_state["initial_suggestions"] = random.sample(list(questions), k=5)

# ────────────────────────────────
# 8) Initial suggestions
# ────────────────────────────────
st.subheader("💡 Suggested questions")
col1, col2 = st.columns(2)
for i, q in enumerate(st.session_state["initial_suggestions"]):
    if (col1 if i < 3 else col2).button(q, key=f"init_{i}"):
        st.session_state["query"] = q

# ────────────────────────────────
# 9) Free‐form input
# ────────────────────────────────
user_q = st.text_input("Or ask your own question:", key="query_input")
if st.button("Ask"):
    if user_q.strip():
        st.session_state["query"] = user_q.strip()

# ────────────────────────────────
# 10) Handle query & generate answer
# ────────────────────────────────
if st.session_state["query"]:
    query = st.session_state["query"]

    # 10.1) Retrieval
    q_vec = embedder.encode([query], convert_to_tensor=False)[0]
    q_vec /= np.linalg.norm(q_vec)
    sims  = embs @ q_vec
    idx   = int(np.argmax(sims))
    ctx_q, ctx_a = questions[idx], answers[idx]

    # 10.2) Build API prompt
    user_msg = (
        f"Context Q: {ctx_q}\n"
        f"Context A: {ctx_a}\n\n"
        f"User: {query}\n"
        "Answer:"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_msg},
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # 10.3) Call Groq
    with st.spinner("Thinking..."):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()

    # 10.4) Display answer
    st.markdown("### Answer")
    st.write(answer)

    # 10.5) Show used context
    st.markdown("**Used context**")
    st.write(f"- Q: {ctx_q}")
    st.write(f"- A: {ctx_a}")

    # ────────────────────────────────
    # 11) Related questions
    # ────────────────────────────────
    st.subheader("🔍 Related questions")
    related = []
    sims_flat = sims.flatten()
    for i in np.argsort(sims_flat)[::-1]:
        if questions[i] != query:
            related.append(questions[i])
        if len(related) >= 4:
            break

    cols = st.columns(2)
    for i, rq in enumerate(related):
        if cols[i % 2].button(rq, key=f"rel_{i}"):
            st.session_state["query"] = rq

