# 🏛️ SevaAI — Govt Scheme Navigator

> AI-powered tool to help Indian citizens discover government schemes they qualify for and get step-by-step application guidance.

**Problem:** People don't know which government schemes they qualify for or how to apply.

**Solution:** Enter your profile → LangChain RAG retrieves matching schemes → GPT-4o gives personalized, step-by-step guidance.

---

## ✨ Features

- 🔍 **Smart matching** — LangChain FAISS vector search across 15+ schemes (expandable to 3000+)
- 🤖 **GPT-4o powered** — Personalized eligibility explanation + application steps
- 🔗 **LangChain LCEL chains** — Modular RAG pipeline with prompt templates
- 💬 **Conversational follow-up** — Ask questions about documents, process, deadlines
- ✅ **Detailed eligibility check** — Per-criterion breakdown for any scheme
- 🖥️ **Streamlit UI** — Clean, interactive interface with tabs and sidebar

---
## Screenshots
<img width="2559" height="1341" alt="1" src="https://github.com/user-attachments/assets/f7530a94-482d-495d-81b8-b57c0d13cdaf" />


<img width="2559" height="1341" alt="2" src="https://github.com/user-attachments/assets/e44f8675-46b5-4786-aa68-5fdfd6a21c8e" />


<img width="2557" height="1341" alt="3" src="https://github.com/user-attachments/assets/502b538f-93d1-46d8-8182-b016aa3d10b7" />


<img width="2559" height="1342" alt="4" src="https://github.com/user-attachments/assets/e5a028ae-cd46-4c5c-9105-dbbe7fb307fc" />


<img width="2559" height="1344" alt="5" src="https://github.com/user-attachments/assets/1dae0733-4b0e-45dd-98f3-c2920e246a3c" />


## Project Demo Link -
https://drive.google.com/file/d/15bQoDO7DA9X2sUPdwnb7N0n_8225ebgf/view?usp=sharing

## 🛠️ Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| LLM | OpenAI GPT-4o | Paid (your key) |
| Embeddings | text-embedding-3-small | Paid (your key) |
| RAG Framework | LangChain (LCEL) | Free |
| Vector DB | FAISS (local) | Free |
| Frontend | Streamlit | Free |
| Hosting | Localhost / any VPS | Free/cheap |

---

## 📁 Project Structure

```
SevaAI/
├── data/
│   └── schemes.json          # Scheme database (15 real Indian schemes)
├── embeddings/
│   ├── faiss_index.pkl       # Raw FAISS index (auto-created)
│   ├── scheme_meta.pkl       # Scheme metadata (auto-created)
│   └── lc_faiss/             # LangChain FAISS vectorstore (auto-created)
├── embed_schemes.py          # One-time embedding + indexing script
├── rag_agent.py              # Core LangChain RAG logic
├── streamlit_app.py          # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Build the vector index (run once only)
```bash
python embed_schemes.py
```
This embeds all schemes using `text-embedding-3-small` and saves a local FAISS index.
**Cost:** ~$0.001 for 15 schemes (negligible).

### 4. Run the app
```bash
streamlit run streamlit_app.py
```

### 5. Open in browser
Streamlit will automatically open:
```
http://localhost:8501
```

---

## 🧠 How It Works

```
User fills profile (age, caste, income, state, need)
        ↓
LangChain builds query string from profile
        ↓
OpenAIEmbeddings embeds the query (text-embedding-3-small)
        ↓
LangChain FAISS vectorstore → similarity_search_with_score()
        ↓
Hard filters applied (caste, age eligibility)
        ↓
Top 5 matching schemes retrieved
        ↓
LCEL Chain: RunnableLambda → ChatOpenAI (GPT-4o) → StrOutputParser
        ↓
Personalized recommendations with exact amounts + steps
        ↓
User can ask follow-up questions via ChatPromptTemplate chain
        ↓
User can click any scheme for detailed eligibility check chain
```

---

## 🔗 LangChain Components Used

| Component | Purpose |
|---|---|
| `ChatOpenAI` | GPT-4o LLM for generating answers |
| `OpenAIEmbeddings` | text-embedding-3-small for query + scheme embedding |
| `FAISS` (langchain_community) | Vector store for similarity search |
| `ChatPromptTemplate` | Structured prompts for follow-up and eligibility check |
| `SystemMessagePromptTemplate` | System role prompt injection |
| `HumanMessagePromptTemplate` | User message formatting |
| `RunnableLambda` | Custom function as LCEL chain step |
| `StrOutputParser` | Parse LLM output to plain string |
| LCEL `\|` operator | Chain: prompt → LLM → parser |

---

## 🗃️ Adding More Schemes

Edit `data/schemes.json` following this format:

```json
{
  "id": "unique-id",
  "name": "Full Scheme Name",
  "ministry": "Ministry Name",
  "category": ["category1", "category2"],
  "eligibility": {
    "occupation": ["farmer", "any"],
    "income": "Below Rs 2.5 lakh",
    "age_min": 18,
    "age_max": null,
    "caste": ["general", "sc", "st", "obc"],
    "states": "all",
    "notes": "Additional eligibility notes"
  },
  "benefits": "What the user gets",
  "documents_needed": ["Aadhaar", "Income certificate"],
  "how_to_apply": ["Step 1", "Step 2", "Step 3"],
  "url": "https://official-site.gov.in",
  "tags": "keywords for search"
}
```

After adding schemes, re-run the embedding script:
```bash
python embed_schemes.py
```
Then delete `embeddings/lc_faiss/` so LangChain rebuilds the vectorstore fresh:
```bash
# Windows
rmdir /s /q embeddings\lc_faiss

# Mac/Linux
rm -rf embeddings/lc_faiss
```

---

## 🌐 Data Sources for Expanding

- **myscheme.gov.in** — 3000+ central + state schemes (official govt portal)
- **india.gov.in** — National scheme directory
- **scholarships.gov.in** — All scholarship schemes

---

## 💰 Cost Estimate

| Operation | Model | Cost |
|---|---|---|
| Embed 15 schemes (one-time) | text-embedding-3-small | ~$0.001 |
| Embed 3000 schemes (one-time) | text-embedding-3-small | ~$0.10 |
| Per user query embedding | text-embedding-3-small | ~$0.000002 |
| Per GPT-4o response | gpt-4o | ~$0.01–0.05 |

**Total running cost per user session: ~$0.05 (₹4)**

---

## 🧪 Sample Test Queries

| Profile | Expected Schemes |
|---|---|
| Farmer, 35yrs, SC, Maharashtra, income 1.5L | PM-KISAN, PMAY-Gramin, PMSBY, PMJJBY |
| Student, 19yrs, OBC, UP, scholarship | Post-Matric OBC Scholarship, PMKVY |
| Entrepreneur, 26yrs, SC, Karnataka, business loan | Stand-Up India, MUDRA Yojana, PMEGP |
| Rural worker, 28yrs, ST, MP, employment | MGNREGS, PMKVY, Atal Pension Yojana |

---

## 🔮 Future Improvements

- [ ] Scrape all 3000+ schemes from myscheme.gov.in
- [ ] Multilingual support (Hindi, Marathi, Tamil, Telugu)
- [ ] LangGraph for stateful multi-turn agent
- [ ] WhatsApp bot integration (Twilio)
- [ ] State-specific scheme filtering
- [ ] User profile saving
- [ ] Scheme application deadline reminders
- [ ] Deploy on Streamlit Cloud (free hosting)

---

## 👨‍💻 Author

Built as part of a GenAI portfolio project demonstrating:
- RAG (Retrieval Augmented Generation)
- LangChain LCEL chains
- Vector embeddings + FAISS
- Streamlit for rapid AI app development
- Real-world civic tech application

---
