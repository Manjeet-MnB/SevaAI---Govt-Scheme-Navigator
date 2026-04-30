import streamlit as st
import os
from dotenv import load_dotenv
from rag_agent import retrieve_schemes, generate_answer, answer_followup, check_eligibility_detail

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SevaAI - Govt Scheme Navigator",
    page_icon="🏛️",
    layout="wide"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stButton > button {
        width: 100%;
        background-color: #1a56db;
        color: white;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
    }
    .stButton > button:hover { background-color: #1447c0; }
    .scheme-card {
        background: #f8faff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #1a56db;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .profile-summary {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 14px;
    }
    .chat-user {
        background: #1a56db;
        color: white;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-left: 20%;
        font-size: 14px;
    }
    .chat-bot {
        background: #f3f4f6;
        color: #111;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-right: 20%;
        font-size: 14px;
    }
    h1 { color: #1a56db !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_schemes" not in st.session_state:
    st.session_state.current_schemes = []
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "searched" not in st.session_state:
    st.session_state.searched = False

# ─── Header ──────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("## 🏛️")
with col_title:
    st.markdown("# SevaAI - Govt Scheme Navigator")
    st.caption("AI-powered tool to find Indian government schemes you qualify for")

st.divider()

# ─── Layout: Sidebar (Profile) + Main (Results) ──────────────────────────────

with st.sidebar:
    st.markdown("### 👤 Your Profile")
    st.caption("Fill in your details to find matching schemes")

    age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)

    occupation = st.selectbox("Occupation", [
        "Farmer",
        "Student",
        "Unemployed / Job seeker",
        "Self employed / Small business",
        "Entrepreneur",
        "Rural / Daily wage worker",
        "Salaried / Other"
    ])

    caste = st.selectbox("Caste Category", [
        "General", "OBC", "SC (Scheduled Caste)", "ST (Scheduled Tribe)"
    ])

    income = st.selectbox("Annual Family Income", [
        "Below Rs 1 lakh",
        "Rs 1 – 2.5 lakh",
        "Rs 2.5 – 6 lakh",
        "Rs 6 – 12 lakh",
        "Above Rs 12 lakh"
    ])

    state = st.text_input("State", placeholder="e.g. Maharashtra, UP, Bihar")

    need = st.text_area(
        "What are you looking for?",
        placeholder="e.g. housing loan, education scholarship, business startup loan...",
        height=80
    )

    st.markdown("---")

    # Quick examples
    st.markdown("**Quick Examples**")
    if st.button("🌾 Farmer · SC · Maharashtra"):
        st.session_state["_example"] = "farmer"
        st.rerun()
    if st.button("🎓 Student · OBC · UP"):
        st.session_state["_example"] = "student"
        st.rerun()
    if st.button("💼 Entrepreneur · SC · Karnataka"):
        st.session_state["_example"] = "entrepreneur"

    st.markdown("---")
    search_btn = st.button("🔍 Find My Schemes", type="primary")

# Handle quick examples
examples = {
    "farmer":       {"age": 35, "occupation": "Farmer", "caste": "SC (Scheduled Caste)", "income": "Rs 1 – 2.5 lakh", "state": "Maharashtra", "need": "farming support and housing"},
    "student":      {"age": 19, "occupation": "Student", "caste": "OBC", "income": "Rs 1 – 2.5 lakh", "state": "Uttar Pradesh", "need": "education scholarship"},
    "entrepreneur": {"age": 26, "occupation": "Entrepreneur", "caste": "SC (Scheduled Caste)", "income": "Rs 2.5 – 6 lakh", "state": "Karnataka", "need": "business startup loan"},
}
if "_example" in st.session_state:
    ex = examples[st.session_state.pop("_example")]
    age = ex["age"]
    occupation = ex["occupation"]
    caste = ex["caste"]
    income = ex["income"]
    state = ex["state"]
    need = ex["need"]

# Build profile dict
caste_clean = caste.split(" ")[0].lower()
profile = {
    "age": age,
    "occupation": occupation.split("/")[0].strip().lower(),
    "caste": caste_clean,
    "income": income,
    "state": state or "India",
    "need": need or "any government benefit"
}

# ─── Main Area ───────────────────────────────────────────────────────────────

if search_btn:
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not set. Add it to your .env file and restart.")
        st.stop()

    # Check embeddings exist
    if not os.path.exists("embeddings/faiss_index.pkl"):
        st.error("⚠️ Embeddings not built. Run `python embed_schemes.py` first.")
        st.stop()

    with st.spinner("🔍 Searching schemes and generating your personalized guide..."):
        schemes = retrieve_schemes(profile, top_k=5)
        if not schemes:
            st.warning("No schemes found matching your profile. Try changing occupation or need.")
        else:
            answer = generate_answer(profile, schemes)
            st.session_state.answer = answer
            st.session_state.current_schemes = schemes
            st.session_state.profile = profile
            st.session_state.chat_history = []
            st.session_state.searched = True

# ─── Results ─────────────────────────────────────────────────────────────────

if st.session_state.searched and st.session_state.answer:

    # Profile summary
    p = st.session_state.profile
    st.markdown(f"""
    <div class="profile-summary">
    ✅ Showing results for: <strong>{p['age']} yr old {p['occupation']} · {p['caste'].upper()} · {p['income']} · {p['state']}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Tabs: Results | Eligibility Check | Chat
    tab1, tab2, tab3 = st.tabs(["📋 Scheme Recommendations", "✅ Eligibility Checker", "💬 Ask Follow-up"])

    # ── Tab 1: Recommendations ────────────────────────────────────────────────
    with tab1:
        st.markdown(st.session_state.answer)

        st.divider()
        st.caption(f"Found **{len(st.session_state.current_schemes)}** matching schemes from our database")
        for s in st.session_state.current_schemes:
            st.markdown(f"""
            <div class="scheme-card">
                <strong>{s['name']}</strong><br>
                <span style="color:#6b7280;font-size:13px">{s['ministry']}</span><br>
                <a href="{s['url']}" target="_blank" style="font-size:13px">🔗 {s['url']}</a>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Eligibility Checker ────────────────────────────────────────────
    with tab2:
        st.markdown("**Select a scheme for a detailed eligibility check:**")

        scheme_names = {s["name"]: s["id"] for s in st.session_state.current_schemes}
        selected_scheme = st.selectbox("Choose scheme", list(scheme_names.keys()))

        if st.button("🔍 Check My Eligibility"):
            with st.spinner("Checking eligibility criteria..."):
                result = check_eligibility_detail(
                    scheme_names[selected_scheme],
                    st.session_state.profile
                )
            st.markdown("### Result")
            st.markdown(result)

    # ── Tab 3: Chat Follow-up ─────────────────────────────────────────────────
    with tab3:
        st.markdown("**Ask anything about these schemes:**")
        st.caption("e.g. 'What documents do I need?' · 'How long does it take?' · 'Can I apply offline?'")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_q = st.text_input("Your question", placeholder="Type your question here...")
            send = st.form_submit_button("Send ➤")

        if send and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Thinking..."):
                bot_reply = answer_followup(
                    user_q,
                    st.session_state.current_schemes,
                    st.session_state.chat_history
                )
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            st.rerun()

# ─── Empty State ─────────────────────────────────────────────────────────────

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #9ca3af;">
        <div style="font-size: 52px; margin-bottom: 16px;">🏛️</div>
        <h3 style="color: #6b7280;">Fill your profile and click Find My Schemes</h3>
        <p style="font-size: 14px; margin-top: 8px;">We'll match you with government schemes you qualify for<br>and give you step-by-step application guidance.</p>
        <br>
        <p style="font-size: 12px;">Helping every Indian citizen find the benefits they deserve</p>
    </div>
    """, unsafe_allow_html=True)