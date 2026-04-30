import os
import json
import pickle
from dotenv import load_dotenv

load_dotenv()

# ─── LangChain Imports ────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ─── Config ──────────────────────────────────────────────────────────────────

INDEX_PATH  = "embeddings/faiss_index.pkl"
META_PATH   = "embeddings/scheme_meta.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o"

# ─── LangChain Components (initialized once) ─────────────────────────────────

# LLM
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.3,
    max_tokens=2000,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Embeddings
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Chat history is managed by Streamlit session_state — no LangChain memory needed

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful Indian government scheme advisor named SevaAI.
Your job is to:
1. Look at the user profile and retrieved schemes carefully
2. Rank schemes from most to least relevant for this specific user
3. For each scheme explain: why they qualify, exact benefit amount, and 3-4 concrete next steps
4. Be specific about documents they need
5. Always respond in simple, friendly English. If user prefers Hindi, mention they can ask in Hindi.
6. Format each scheme as:
   🔹 **Scheme Name**
   ✅ Why you qualify: ...
   💰 Benefit: ...
   📋 Next steps: ...
   🔗 Apply at: URL

Keep it practical and actionable. Do not mention schemes the user clearly does not qualify for."""


# ─── FAISS Vectorstore (LangChain) ───────────────────────────────────────────

_vectorstore = None
_meta = None

def _load_vectorstore():
    """
    Load FAISS index. 
    If built by embed_schemes.py (raw faiss), wrap it in LangChain FAISS vectorstore.
    """
    global _vectorstore, _meta

    if _vectorstore is not None:
        return

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Embeddings not found. Run: python embed_schemes.py")

    # Load metadata
    with open(META_PATH, "rb") as f:
        _meta = pickle.load(f)

    # Try loading as LangChain FAISS first
    lc_index_path = "embeddings/lc_faiss"
    if os.path.exists(lc_index_path):
        _vectorstore = FAISS.load_local(
            lc_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Loaded LangChain FAISS vectorstore")
        return

    # First run: convert raw FAISS → LangChain FAISS vectorstore
    print("🔄 Converting raw FAISS index to LangChain vectorstore (one-time)...")
    import faiss
    import numpy as np

    with open(INDEX_PATH, "rb") as f:
        raw_index = faiss.deserialize_index(pickle.load(f))

    # Rebuild as LangChain FAISS using scheme texts as documents
    docs = []
    for m in _meta:
        s = m["scheme"]
        elig = s.get("eligibility", {})
        text = (
            f"Scheme: {s['name']}. "
            f"Ministry: {s.get('ministry', '')}. "
            f"Category: {', '.join(s.get('category', []))}. "
            f"Occupation: {', '.join(elig.get('occupation', []))}. "
            f"Caste: {', '.join(elig.get('caste', []))}. "
            f"Income: {elig.get('income') or 'no restriction'}. "
            f"Benefits: {s.get('benefits', '')}. "
            f"Tags: {s.get('tags', '')}."
        )
        docs.append(Document(
            page_content=text,
            metadata={"id": s["id"], "name": s["name"]}
        ))

    _vectorstore = FAISS.from_documents(docs, embeddings)
    _vectorstore.save_local(lc_index_path)
    print(f"✅ LangChain FAISS vectorstore built and saved ({len(docs)} schemes)")


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_schemes(user_profile: dict, top_k: int = 5) -> list[dict]:
    """
    Use LangChain FAISS similarity search to find matching schemes.
    Then apply hard filters for caste and age.
    """
    _load_vectorstore()

    # Build query string from profile
    query = (
        f"Government schemes for a person who is "
        f"{user_profile.get('age', '')} years old, "
        f"occupation: {user_profile.get('occupation', 'any')}, "
        f"caste: {user_profile.get('caste', 'general')}, "
        f"annual income: {user_profile.get('income', 'unknown')}, "
        f"state: {user_profile.get('state', 'India')}, "
        f"looking for: {user_profile.get('need', 'any benefit')}."
    )

    # LangChain similarity search — returns list of (Document, score)
    docs_with_scores = _vectorstore.similarity_search_with_score(query, k=top_k * 2)

    results = []
    seen = set()

    for doc, score in docs_with_scores:
        scheme_id = doc.metadata.get("id")
        if scheme_id in seen:
            continue
        seen.add(scheme_id)

        # Get full scheme from metadata
        scheme = next((m["scheme"] for m in _meta if m["id"] == scheme_id), None)
        if not scheme:
            continue

        # Hard filter: caste
        caste = user_profile.get("caste", "general").lower()
        allowed = scheme["eligibility"].get("caste", ["general", "sc", "st", "obc"])
        if caste not in allowed and "any" not in allowed:
            continue

        # Hard filter: age
        age = user_profile.get("age")
        if age:
            age_min = scheme["eligibility"].get("age_min", 0) or 0
            age_max = scheme["eligibility"].get("age_max") or 999
            if not (age_min <= int(age) <= age_max):
                continue

        results.append(scheme)
        if len(results) >= top_k:
            break

    return results


# ─── RAG Chain (LangChain LCEL) ───────────────────────────────────────────────

def _build_rag_prompt(inputs: dict) -> list:
    """Build the prompt messages from profile + schemes."""
    profile_text = json.dumps(inputs["profile"], indent=2, ensure_ascii=False)
    schemes_text = json.dumps(inputs["schemes"], indent=2, ensure_ascii=False)

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User Profile:\n{profile_text}\n\n"
            f"Retrieved Schemes:\n{schemes_text}\n\n"
            f"Please provide personalized scheme recommendations. "
            f"Rank by relevance and explain clearly why they qualify and exact steps to apply."
        ))
    ]

# LCEL chain: build prompt → LLM → parse output
rag_chain = (
    RunnableLambda(_build_rag_prompt)
    | llm
    | StrOutputParser()
)


def generate_answer(user_profile: dict, schemes: list[dict], chat_history: list[dict] = None) -> str:
    """
    Use LangChain LCEL RAG chain to generate personalized scheme recommendations.
    """
    return rag_chain.invoke({
        "profile": user_profile,
        "schemes": schemes
    })


# ─── Conversational Follow-up (with LangChain Memory) ────────────────────────

# Prompt template for follow-up chat
followup_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nSchemes currently in context:\n{schemes_context}"
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Follow-up chain
followup_chain = followup_prompt | llm | StrOutputParser()


def answer_followup(question: str, context_schemes: list[dict], chat_history: list[dict]) -> str:
    """
    Handle follow-up questions using LangChain prompt templates.
    Chat history is passed in from Streamlit session_state.
    """
    schemes_context = json.dumps(context_schemes, indent=2, ensure_ascii=False)

    return followup_chain.invoke({
        "question": question,
        "schemes_context": schemes_context
    })


# ─── Eligibility Check Chain ──────────────────────────────────────────────────

eligibility_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
Given this user profile:
{profile}

And this government scheme:
{scheme}

Do a detailed eligibility check:
- List each eligibility criterion and whether the user meets it (✅ or ❌)
- Give overall verdict: Eligible / Likely Eligible / Not Eligible
- If not fully eligible, suggest what they could do to become eligible
- Mention any exceptions or special provisions that might apply
""")
])

eligibility_chain = eligibility_prompt | llm | StrOutputParser()


def check_eligibility_detail(scheme_id: str, user_profile: dict) -> str:
    """
    Use LangChain chain to do a detailed per-criterion eligibility check.
    """
    _load_vectorstore()

    scheme = next((m["scheme"] for m in _meta if m["id"] == scheme_id), None)
    if not scheme:
        return "Scheme not found."

    return eligibility_chain.invoke({
        "profile": json.dumps(user_profile, indent=2),
        "scheme": json.dumps(scheme, indent=2)
    })
