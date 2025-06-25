import streamlit as st
import json, os, requests
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ────────────────────────────────
# 0) Load environment & config
# ────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Please set the GROQ_API_KEY in your .env")
    st.stop()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL    = "llama3-8b-8192"

st.set_page_config(page_title=" V-Mitra Chatbot", layout="centered")

# ────────────────────────────────
# 1) Polished CSS + header
# ────────────────────────────────


st.markdown("""
<style> 
/* Limit the header width to match chat-window and center it */
.header-bar {
  max-width: 600px;    /* same as your .chat-window max-width */
  margin: 0 auto;      /* center horizontally */
}

/* Also center the chat-window itself */
.chat-window {
  max-width: 600px;
  margin: 16px auto;   /* keep a bit of top/bottom margin */
}

/* (Optional) Make input box the same width too */
.stChatInput {
  max-width: 600px !important;
  margin: 0 auto !important;
}            
/* Hide the white chat-window background */
.chat-window {
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* If you want to collapse its container completely: */
.chat-window > * {
  margin: 0 !important;
  max-width: 100% !important;
}
            /* 1) Fix the header at top */
.header-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
}

/* 2) Push the rest of the page down to clear the fixed header */
/*    Adjust 80px to match your header-bar height + any gap */
.block-container {
  padding-top: 80px !important;
}

/* 3) Make only the chat area scroll */
/*    Calculate: full viewport minus header (80px) minus input box area (e.g. 80px) */
.chat-window {
  max-height: calc(100vh - 160px) !important;
  overflow-y: auto !important;
  margin-top: 16px;  /* optional spacing under the header */
}

/* 4) (Optional) Style the scrollbar inside chat-window */
.chat-window::-webkit-scrollbar {
  width: 6px;
}
.chat-window::-webkit-scrollbar-thumb {
  background: rgba(74,0,224,0.6);
  border-radius: 3px;
}
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&family=Montserrat:wght@700&display=swap');

html, body, .main, .block-container {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
  font-family: 'Nunito', 'Montserrat', sans-serif !important;
  margin: 0; padding: 0;
}

/* Header */
.header-bar {
  display: flex;
  align-items: center;
  background: linear-gradient(90deg, #8e2de2, #4a00e0);
  padding: 18px;
  border-radius: 0 0 20px 20px;
  margin-bottom: 16px;
  box-shadow: 0 4px 20px rgba(0,0,0,.1);
  animation: slideDown 0.6s ease-out;
}
.header-bar img {
  height: 50px;
  margin-right: 14px;
  border: 3px solid #fff;
  border-radius: 50%;
  background: #fff;
  transition: transform .3s ease;
}
.header-bar img:hover {
  transform: scale(1.1) rotate(5deg);
}
.header-bar .chatbot-title {
  font-size: 1.9rem;
  color: #fff;
  font-weight: 700;
  margin: 0;
}
.header-bar .chatbot-desc {
  font-size: 1rem;
  color: #e0e0e0;
  margin: 4px 0 0 0;
}

/* Chat pane */
.chat-window {
  max-width: 600px;
  margin: auto;
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(8px);
  border-radius: 24px;
  box-shadow: 0 12px 32px rgba(0,0,0,.1);
  padding: 16px;
  min-height: 60vh;
  overflow-y: auto;
  scroll-snap-type: y mandatory;
}
.chat-window::-webkit-scrollbar {
  width: 6px;
}
.chat-window::-webkit-scrollbar-track {
  background: transparent;
}
.chat-window::-webkit-scrollbar-thumb {
  background: rgba(74,0,224,0.6);
  border-radius: 3px;
}
.chat-window::-webkit-scrollbar-thumb:hover {
  background: rgba(74,0,224,0.9);
}

/* Messages */
.message-row {
  display: flex;
  margin: 8px 0;
  opacity: 0;
  animation: fadeIn 0.4s forwards;
  scroll-snap-align: start;
}
.message-row.user {
  flex-direction: row-reverse;
}
.bubble {
  max-width: 75%;
  padding: 14px 20px;
  border-radius: 20px;
  word-break: break-word;
  position: relative;
  font-size: 1rem;
  line-height: 1.4;
  transition: transform .2s, box-shadow .2s;
}
.bubble.user {
  background: linear-gradient(120deg, #00c6ff 0%, #0072ff 100%);
  color: #fff;
  border-bottom-right-radius: 5px;
}
.bubble.bot {
  background: linear-gradient(120deg, #ffe29f 0%, #ffa99f 100%);
  color: #333;
  border-bottom-left-radius: 5px;
}
.bubble:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,.08);
}
/* Pop effect */
.bubble::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 8px;
  height: 8px;
  background: transparent;
  border-radius: 50%;
  transform: translate(-50%,-50%) scale(0);
  animation: pop 0.5s ease-out;
}

/* Avatars */
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  margin: 0 10px;
  flex-shrink: 0;
  transition: box-shadow .3s;
}
.avatar.bot {
  box-shadow: 0 0 0 2px #ffa99f;
}
.avatar.user {
  box-shadow: 0 0 0 2px #0072ff;
}
.avatar:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,.2);
}

/* Input */
.stChatInput {
  border-radius: 24px!important;
  border: 2px solid #0072ff!important;
  padding: 10px;
  font-size: 1rem!important;
  box-shadow: 0 2px 8px rgba(0,0,0,.1)!important;
  transition: box-shadow .3s, border-color .3s;
}
.stChatInput:focus {
  outline: none;
  border-color: #4a00e0!important;
  box-shadow: 0 4px 16px rgba(0,0,0,.15)!important;
}

/* Typing indicator */
.typing-row {
  display: flex;
  align-items: center;
  margin: 8px 0;
  opacity: 0;
  animation: fadeIn 0.4s forwards;
}
.typing-row .bubble {
  background: linear-gradient(120deg, #ffe29f 0%, #ffa99f 100%);
  color: #333;
  border-bottom-left-radius: 5px;
  max-width: 60%;
  padding: 10px 16px;
}
.dot-flashing {
  position: relative;
  width: 30px;
  height: 12px;
  margin-right: 8px;
}
.dot-flashing span, .dot-flashing:before, .dot-flashing:after {
  content: '';
  display: inline-block;
  position: absolute;
  top: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ffa99f;
  animation: dotFlashing 1s infinite ease-in-out;
}
.dot-flashing:before { left: 0; animation-delay: 0s; }
.dot-flashing span  { left:10px; animation-delay: .3s; }
.dot-flashing:after { left:20px; animation-delay: .6s; }

/* Animations */
@keyframes fadeIn {
  0% { opacity: 0; transform: translateY(10px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes slideDown {
  0% { transform: translateY(-20px); opacity: 0; }
  100% { transform: translateY(0); opacity: 1; }
}
@keyframes pop {
  0% { transform: translate(-50%,-50%) scale(0); }
  80% { transform: translate(-50%,-50%) scale(1.2); }
  100% { transform: translate(-50%,-50%) scale(1); }
}
@keyframes dotFlashing {
  0% { opacity: .2; }
  50%,100% { opacity: 1; }
}
/* ---- CONTACT BOX FLEX VERSION ---- */
.contact-box-flex {
  max-width: 700px;
  margin: 32px auto 0 auto;
  background: rgba(255,255,255,0.92);
  border-radius: 20px;
  box-shadow: 0 6px 24px #3c00e010;
  padding: 24px 32px;
  font-family: 'Nunito', 'Montserrat', sans-serif;
  font-size: 1.08rem;
  color: #3e348f;
  text-align: left;
  border-left: 7px solid #7b2ff2;
  border-top: 2px solid #f5f7fa;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 32px;
}
.contact-details {
  flex: 2;
}
.contact-image {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}
.contact-image img {
  width: 200px;
  height: 200px;
  object-fit: contain;
  border-radius: 28px;
  border: 2px solid #7b2ff2;
  box-shadow: 0 2px 10px #7b2ff222;
  background: #f9f9fc;
}
@media (max-width: 950px) {
  .contact-box-flex { flex-direction: column; gap: 12px; }
  .contact-image img { width: 120px; height: 120px; }
}
</style>

<div class="header-bar">
  <img src="https://mpez.co.in/static/assets/img/vmitra_logo_latest.jpeg" alt="bot"/>
  <div>
    <div class="chatbot-title"> V-Mitra Helpdesk AI ChatBot</div>
    <div class="chatbot-desc">वी-मित्र एप्लिकेशन सहायक – आपकी ऐप हेल्पडेस्क</div>
  </div>
</div> 



<!-- New Flex Contact Box with Image -->
<div class="contact-box-flex">
  <div class="contact-details">
    <h3>Contact Us</h3>
    <p>
      Block No. 7, Shakti Bhawan<br>
      PO: Vidyut Nagar, Rampur<br>
      Jabalpur (M.P.) India<br><br>
      <b>Phone:</b> <a href="tel:18002331266">1800-233-1266</a><br>
      <b>Email:</b> <a href="mailto:mpez.nidaan@gmail.com">mpez.nidaan@gmail.com</a>
    </p>
  </div>
  <div class="contact-image">
    <img src="https://mpez.co.in/static/assets/img/vmitra_logo_latest.jpeg" alt="V-Mitra Logo"/>
  </div>
</div>
""", unsafe_allow_html=True)





# ────────────────────────────────
# 2) Load KB & embedder
# ────────────────────────────────
kb = json.load(open("vmitra_knowledge_base.json","r",encoding="utf-8"))["v_mitra_knowledge_base"]
example_qas = []
for sec in kb["sections"]:
    if sec["id"]=="sample_intents_entities":
        for intent in sec["intents"]:
            for ex in intent["examples"]:
                example_qas.append((ex["question"], ex["answer"]))
questions, answers = zip(*example_qas)

@st.cache_resource
def init_embedder():
    m = SentenceTransformer("all-MiniLM-L6-v2")
    e = np.array(m.encode(questions, convert_to_tensor=False))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return m, e

embedder, embs = init_embedder()

# ────────────────────────────────
# 3) System prompt
# ────────────────────────────────
system_prompt = """
You are V-Mitra ChatBot, the official AI assistant for the V-Mitra app user citizen audit app.
Use the given context to answer precisely, courteously, and professionally.

RULES:
1. DOMAIN ONLY: Only answer V-Mitra questions; otherwise ask to reframe.
2. CONTEXT USAGE: Summarize the context (1–2 lines) then answer.
3. ALWAYS ANSWER: Provide best-guess if unsure, citing the context.
4. SAFETY FIRST: Refuse harmful requests politely.
5. STYLE: Use “Step 1, Step 2…” for instructions.
"""

# ────────────────────────────────
# 4) Initialize chat history
# ────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [("bot","👋 Hello! Ask me anything about V-Mitra.")]

# ────────────────────────────────
# 5) Display chat window
# ────────────────────────────────
chat_win = st.container()
with chat_win:
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for role, text in st.session_state.history:
        is_user = (role=="user")
        avatar = ("https://cdn-icons-png.flaticon.com/512/9131/9131546.png"
                  if is_user else
                  "https://cdn-icons-png.flaticon.com/512/4712/4712035.png")
        row_cls = "user" if is_user else "bot"
        bub_cls = "bubble user" if is_user else "bubble bot"
        st.markdown(f"""
          <div class="message-row {row_cls}">
            <img src="{avatar}" class="avatar"/>
            <div class="{bub_cls}">{text}</div>
          </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────
# 6) User input at bottom
# ────────────────────────────────
user_text = st.chat_input("Type your question…", key="input")
if user_text:
    # Append user message
    st.session_state.history.append(("user", user_text))

    # Compute and append bot response under spinner
    with st.spinner("V-Mitra Bot is typing…"):
        # Retrieval
        qv = embedder.encode([user_text])[0]
        qv /= np.linalg.norm(qv)
        sims = embs @ qv
        idx  = int(np.argmax(sims))
        ctx_q, ctx_a = questions[idx], answers[idx]

        # Call Groq
        payload = {
          "model": LLM_MODEL,
          "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":
              f"Context Q: {ctx_q}\nContext A: {ctx_a}\n\nUser: {user_text}\nAnswer:"
            }
          ]
        }
        headers = {
          "Authorization": f"Bearer {GROQ_API_KEY}",
          "Content-Type": "application/json"
        }
        r = requests.post(GROQ_API_URL, headers=headers, json=payload)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()

    # Append bot message
    st.session_state.history.append(("bot", reply))

    # Rerender chat window with new messages
    st.rerun()
