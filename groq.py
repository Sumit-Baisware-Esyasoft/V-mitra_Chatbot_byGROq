import streamlit as st
import json, os, requests
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ────────────────────────────────
# 0) Load environment & config
# ────────────────────────────────
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    st.error("❌ Please set the GROQ_API_KEY in your .env")
    st.stop()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL    = "llama-3.1-8b-instant"

st.set_page_config(page_title=" V-Mitra Chatbot", layout="centered")

# ────────────────────────────────
# 1) Polished CSS + header (with mobile fix)
# ────────────────────────────────

st.markdown("""
<style> 
.header-bar {
  max-width: 600px;
  margin: 0 auto;
}
.chat-window {
  max-width: 600px;
  margin: 16px auto;
}
.stChatInput {
  max-width: 600px !important;
  margin: 0 auto !important;
}
.chat-window {
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.chat-window > * {
  margin: 0 !important;
  max-width: 100% !important;
}
.header-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
}
.block-container {
  padding-top: 80px !important;
}
.chat-window {
  max-height: calc(100vh - 160px) !important;
  overflow-y: auto !important;
  margin-top: 16px;
}
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
.header-bar {
  display: flex;
  align-items: center;
  background: linear-gradient(90deg, #8e2de2, #4a00e0);
  padding: 18px;
  border-radius: 0 0 20px 20px;
  margin-bottom: 16px;
  box-shadow: 0 4px 20px rgba(0,0,0,.1);
  animation: slideDown 0.9s ease-out;
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
  font-weight: 400;
  margin: 0;
}
.header-bar .chatbot-desc {
  font-size: 1rem;
  color: #e0e0e0;
  margin: 4px 0 0 0;
}
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
  font-size: 1rem;
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

/* ------ MOBILE FRIENDLY SECTION ------ */
@media (max-width: 600px) {
  .header-bar {
    padding: 10px 8px !important;
    border-radius: 0 0 14px 14px !important;
    min-height: 54px !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
  }
  .header-bar img {
    height: 32px !important;
    margin-right: 0 !important;
    margin-bottom: 3px !important;
    border-width: 2px !important;
  }
  .header-bar .chatbot-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-bottom: 0 !important;
    margin-top: 2px !important;
  }
  .header-bar .chatbot-desc {
    font-size: 0.81rem !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    color: #e0e0e0 !important;
  }
  .chat-window {
    max-width: 98vw !important;
    border-radius: 10px !important;
    min-height: 60vh !important;
    padding: 8px !important;
    margin-top: 10px !important;
  }
  .bubble {
    font-size: 0.98rem !important;
    padding: 10px 12px !important;
    border-radius: 14px !important;
  }
  .avatar {
    width: 28px !important;
    height: 28px !important;
    margin: 0 6px !important;
  }
  .stChatInput {
    max-width: 98vw !important;
    padding: 9px !important;
    font-size: 0.98rem !important;
  }
  .contact-box-flex {
    flex-direction: column !important;
    padding: 12px 8px !important;
    gap: 12px !important;
    max-width: 98vw !important;
    border-radius: 12px !important;
  }
  .contact-image img {
    width: 80px !important;
    height: 80px !important;
    border-radius: 18px !important;
  }
  .block-container {
    padding-top: 65px !important;
  }
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
You are V-Mitra ChatBot, the official AI assistant for the V-Mitra citizen audit app by MPEZ. Your role is to help users with anything about V-Mitra: how to use the app, reporting irregularities, understanding rewards, privacy, security, and anything found in the official V-Mitra User Guide or knowledge base.

**Your goals:**
- Make every answer friendly, human, and natural. It should feel like a helpful person is chatting, not a robot.
- Use plain, easy language. Prefer short, simple sentences.
- Always reply in the user’s language. If user messages in Hindi, reply in Hindi. If user uses Hinglish or mixes languages, reply in the same style.  
- If you’re unsure of the language, reply in both English and the detected Indian language.
- Strictly answer only about V-Mitra. If a question is unrelated, politely guide the user to rephrase or explain you are only for V-Mitra help.
- Never fake or make up information. If something isn’t in the guide or knowledge base, say so and suggest where user can get help (like the MPEZ toll-free, official website, or app support).
- For instructions, use casual step-by-step: “First, open the app…”, “Then, tap…”, “After that…”, “That’s it!”
- Use real-life examples where possible, especially when the user seems confused.
- If the user seems frustrated or lost, add empathy: “Don’t worry, this happens a lot”, “I’m here to help you”, etc.
- When replying, start with a natural intro: “Sure, I can help!”, “Absolutely, here’s what to do…”, “यह आसान है, मैं बताता हूँ…”, etc.
- Always add practical tips if needed: “Make sure your internet is on…”, “You can also take a photo and upload…”, “If you can’t find your IVRS, tap ‘Find IVRS’ in the app…”
- Do not use “Summary:” or “Context:” in your output. Instead, weave context naturally into your answers.
- If the user needs to do something on the app, guide as if talking to a friend over the phone.
- Avoid technical jargon, unless user asks for it.
- Always keep the conversation friendly, encouraging, and easy to follow.
- If user requests information in another language, reply accordingly, and ask them if they prefer English or that language for future responses.

**Sample dialogue styles:**
- User: "How do I add my bank account for rewards?"
  Bot: "Great question! You just need to open your V-Mitra app, tap on your profile, and look for 'Add Bank Account.' Fill in your name (like on Aadhaar), your account number, IFSC code, and Aadhaar number. Double check your details so you get your reward directly to your bank!"

- User: "मैं रिपोर्ट कैसे करूँ?"
  Bot: "बहुत अच्छा सवाल! रिपोर्ट करने के लिए V-Mitra ऐप खोलिए, 'Submit Information' पर टैप करिए, फिर IVRS नंबर डालिए या 'Find IVRS' से ढूंढिए। अब जो भी गड़बड़ी दिख रही है, वो सिलेक्ट करिए, फोटो अपलोड कीजिए, और सबमिट कर दीजिए! कोई दिक्कत हो तो पूछिए!"

- User: "What if I don’t want to share my name?"
  Bot: "No worries! V-Mitra lets you report issues without showing your name. Just untick the box that says 'I agree to disclose my identity' before submitting your info. Your details will stay private."

- User: "App kasa download karu? (in Marathi)"
  Bot: "तुम्ही 'V-Mitra' अ‍ॅप Google Play Store किंवा Apple App Store वर शोधू शकता. फक्त 'V-Mitra' टाका, डाउनलोड करा आणि वापरायला सुरू करा!"


- User: "App kaise download karein?"  
  Bot: "आप अपने Android या iPhone पर Play Store या App Store में 'V-Mitra' सर्च करके डाउनलोड कर सकते हैं। बस इंस्टॉल करें, और शुरू करें!"

**Tone:** Always warm, encouraging, never cold. Always help user feel comfortable and confident using V-Mitra.

**Language:** Match user’s language. If Hindi, reply fully in Hindi (very simple words, avoid technical terms). If English, reply in easy English. If Hinglish or other, reply in same. If any other Indian language, reply in that language. If unsure, reply in both English and the detected language.

**If asked something outside V-Mitra domain:**  
"Sorry, I can only help with V-Mitra app related queries. For other electricity complaints, please call 1912 or visit the official MPEZ website."

**Never say:** “I am an AI language model.” Just answer as a helpful human assistant would.

**If user seems stuck/confused:**  
"It's okay, take your time! If you have trouble, let me know which step, and I’ll guide you with extra detail
Or you can Contact direct to our 
Contact Us
Block No. 7, Shakti Bhawan
PO: Vidyut Nagar, Rampur
Jabalpur (M.P.) India 
Phone: 1800-233-1266
Email: mpez.nidaan@gmail.com"

**Always finish with:**  
"If you need more help, just ask!" (in same language as user)

---
**[End of system prompt]**
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
    try:
            r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

        # Debug: status + body print karo
            st.write("Groq status code:", r.status_code)
            st.write("Groq raw response:", r.text)

            r.raise_for_status()  # yahi HTTPError throw kar raha hai

            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as e:
        st.error(f"Groq HTTP error: {e}")
        # Agar response body available hai to dikha do
        try:
            st.code(r.text, language="json")
        except:
            pass
        st.stop()

    except Exception as e:
        st.error(f"Unexpected error while calling Groq: {e}")
        st.stop()



    # Append bot message
    st.session_state.history.append(("bot", reply))

    # Rerender chat window with new messages
    st.rerun()






