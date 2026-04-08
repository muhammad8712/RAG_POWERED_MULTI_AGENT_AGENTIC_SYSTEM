# 🧠 RAG-Powered Multi-Agent Conversational Assistant for ERP Workflows

> **BSc Computer Science Thesis** — Eötvös Loránd University, Faculty of Informatics  
> **Author:** Muhammad Abdullah &nbsp;|&nbsp; **Supervisor:** Md Easin Arafat PhD.  
> **Department:** Data Science and Engineering &nbsp;|&nbsp; **Year:** 2025

---

## 📌 Overview

This project is a multi-agent conversational assistant built specifically for Enterprise Resource Planning (ERP) workflows. Users can ask natural language questions and receive accurate, fully explainable answers drawn from two fundamentally different data sources — a live relational database and a set of policy PDF documents — within a single unified interface.

The system goes beyond a standard RAG chatbot by orchestrating a pipeline of specialised AI agents through a LangGraph state machine, validating evidence quality before surfacing answers, and supporting multi-turn conversation so follow-up questions like *"now filter that by Germany"* are resolved automatically.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔀 Multi-Agent Orchestration | LangGraph state machine routes queries to specialised agents based on classified intent |
| 🗄️ Natural Language to SQL | DatabaseAgent generates validated SQLite queries from plain English |
| 📄 RAG Policy Retrieval | DocumentAgent retrieves answers from 5 ERP policy PDFs via FAISS vector search |
| 🔄 Corrective Validation Loop | CorrectiveValidationAgent re-runs agents automatically when evidence is insufficient |
| 💬 Multi-Turn Conversation | Conversation history passed to all agents — follow-up questions work natively |
| 🛡️ 6-Layer SQL Safety | Blocks hallucinated columns, bad joins, destructive SQL, and prompt injection |
| 🔍 Full Explainability | Every answer shows SQL used, documents retrieved, similarity scores, and agent trace |
| 📊 100-Query Evaluation | Benchmark suite with ground truth, 4 scoring methods, and a visual dashboard |
| 🚫 Adversarial Protection | All destructive and system-manipulation requests are blocked at query level |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Intent Classifier                  │
│  (rule-based keywords → LLM fallback)           │
│  DOCUMENT / DATABASE / API / COMPOSITE          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Orchestrator Agent                 │
│  Generates execution plan: e.g.                 │
│  ['database', 'document', 'reasoning',          │
│   'validate', 'explainability']                 │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   DatabaseAgent  DocumentAgent  APIAgent
   (SQL gen +     (FAISS RAG +   (REST stub)
    validation)    stuff-chain)
          │            │
          └─────┬──────┘
                ▼
         ReasoningAgent
         (synthesises DB +
          document evidence)
                │
                ▼
  CorrectiveValidationAgent
  (PASS / NEEDS_MORE_INFO / FAIL)
  (re-injects corrective steps if weak)
                │
                ▼
       ExplainabilityAgent
       (answer + SQL + sources +
        scores + trace + validation)
                │
                ▼
         Final Response
```

---

## 📂 Project Structure

```
THESIS/
├── agents/
│   ├── api_agent.py               # REST API caller with URL allowlist
│   ├── database_agent.py          # NL→SQL + 6-layer validation + execution
│   ├── document_agent.py          # FAISS retrieval + stuff-chain QA
│   ├── explainability_agent.py    # Response packaging with full provenance
│   └── reasoning_agent.py         # Multi-source synthesis agent
│
├── orchestration/
│   ├── graph.py                   # LangGraph state machine (GraphState + nodes)
│   ├── intent_classifier.py       # Two-tier classifier (keyword + LLM fallback)
│   ├── orchestrator_agent.py      # Execution plan generator
│   └── validator_agent.py         # Corrective validation with security guards
│
├── data/
│   ├── customers.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── products.csv
│   ├── payments.csv
│   ├── Sales_Order.xlsx           # Odoo export
│   ├── Purchase_Order.xlsx        # Odoo export
│   ├── schema.sql                 # SQLite schema definition
│   └── generate_mock_erp.py       # Database builder script
│
├── ingestion/
│   └── document_ingestion.py      # PDF → chunks → FAISS index
│
├── policies/
│   ├── invoice_matching_policy.pdf
│   ├── payment_terms_policy.pdf
│   ├── procurement_guidelines.pdf
│   ├── purchase_order_approval_policy.pdf
│   └── vendor_onboarding_policy.pdf
│
├── evaluation/
│   ├── eval_queries.json          # 100 benchmark queries
│   ├── ground_truth.json          # Reference answers + key facts
│   ├── run_eval.py                # Evaluation runner
│   ├── score_eval.py              # Multi-method scorer
│   └── eval_dashboard.py          # Streamlit evaluation dashboard
│
├── storage/
│   └── vector_store/              # FAISS index (index.faiss + index.pkl)
│
├── logs/
│   ├── events.jsonl               # Query event log
│   └── validation.jsonl           # Validation issue log
│
├── streamlit_app.py               # Web interface
├── Main_Cli.py                    # Terminal interface
├── erp.db                         # SQLite database (generated)
├── requirements.txt
└── .env                           # GROQ_API_KEY goes here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/erp-multi-agent.git
cd erp-multi-agent
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Build the database and vector store (run once)

```bash
# Step 1 — Generate the SQLite ERP database from CSV/Excel files
python data/generate_mock_erp.py

# Step 2 — Ingest policy PDFs into the FAISS vector store
python ingestion/document_ingestion.py
```

### 6. Launch the application

**Web interface (Streamlit):**
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`

**Terminal interface:**
```bash
python Main_Cli.py
```

---

## 💡 Example Queries

### Database queries
```
Who are the top 5 customers by total order value?
Which product categories generate the highest revenue?
What is the average order value by device type?
Which products are purchased together most often?
Show the most recent 10 orders.
```

### Policy document queries
```
What is the standard payment term?
What is the grace period for late invoice payments?
What approval is required for purchase orders above 25,000 EUR?
What documents are required for vendor onboarding?
What is the invoice matching tolerance percentage?
```

### Composite queries (DB + document)
```
Show top customers by revenue and explain the early payment discount policy.
Which payment methods are most common and what is the grace period rule?
Show recent orders and explain the purchase order approval thresholds.
```

### Multi-turn follow-ups
```
User:      "Who are the top 5 customers by revenue?"
Assistant: [returns list]
User:      "Now filter that by Germany."
Assistant: [correctly re-runs with WHERE country = 'Germany']
```

---

## 🗄️ Database Schema

The SQLite database contains 7 tables:

| Table | Description |
|---|---|
| `customers` | Customer profiles with country, age, signup date |
| `products` | Product catalogue with category, price, cost, margin |
| `orders` | Order headers with payment method, device, source, totals |
| `order_items` | Line items linking orders to products with quantities |
| `payments` | Payment transactions linked to customers |
| `sales_orders_odoo` | Odoo CRM export — sales order references and statuses |
| `purchase_orders_odoo` | Odoo procurement export — PO references and statuses |

---

## 📊 Evaluation Framework

The system is evaluated against **100 benchmark queries** across five categories:

| Category | Count | Scoring Method | What is tested |
|---|---|---|---|
| DOCUMENT | 30 | Keyword / LLM judge | Policy retrieval accuracy |
| DATABASE | 40 | Structural | SQL generation + row return |
| COMPOSITE | 20 | Composite | DB rows + document keywords |
| ADVERSARIAL | 10 | Adversarial | Security blocking |

**Score scale:** 2 = PASS · 1 = PARTIAL · 0 = FAIL  
**Weighted score:** `(PASS×2 + PARTIAL×1) / (Total×2) × 100`

### Running the evaluation

```bash
# Run all 100 queries
python evaluation/run_eval.py

# Score results against ground truth
python evaluation/score_eval.py \
    --results evaluation/results/eval_results_TIMESTAMP.jsonl

# Launch visual dashboard
streamlit run evaluation/eval_dashboard.py
```

---

## 🛡️ Security

The system blocks all adversarial and destructive requests at two independent layers:

**Layer 1 — Query-level guard** (CorrectiveValidationAgent):
Detects patterns like `DROP TABLE`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, system prompt extraction, database export requests, and validation bypass attempts. Returns `FAIL` immediately.

**Layer 2 — SQL validation pipeline** (DatabaseAgent):
- Enforces SELECT-only queries
- Whitelist of 7 allowed tables
- Detects hallucinated column references (e.g. `oi.category`)
- Blocks bad joins (e.g. `customer_id = product_id`)
- Prevents multiple statements and forbidden keywords
- Guards against schema-violating queries

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| LLM | `llama-3.1-8b-instant` via Groq API |
| Orchestration | LangGraph + LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (faiss-cpu) |
| Database | SQLite via SQLAlchemy |
| PDF Parsing | pdfplumber + PyMuPDF |
| Web Interface | Streamlit |
| Data Processing | pandas, openpyxl |

---

## 🔮 Future Work

- **Two-model routing** — use `qwen/qwen3-32b` for SQL generation and reasoning (95%+ text-to-SQL accuracy) while keeping `llama-3.1-8b-instant` for fast classification
- **Live ERP API integration** — connect APIAgent to real Odoo or SAP REST endpoints
- **Fine-tuned SQL model** — domain-specific fine-tuning on the ERP schema to eliminate hallucination entirely
- **Persistent session history** — user accounts with saved conversation and query history
- **Multilingual support** — language detection and multilingual embeddings for non-English ERP environments
- **CI/CD evaluation pipeline** — automatic regression testing on every code change

---

## 📄 License

This project is submitted as a BSc thesis at Eötvös Loránd University. All rights reserved.

---

## 🙏 Acknowledgements

- **Supervisor:** Md Easin Arafat PhD., Department of Data Science and Engineering, ELTE
- [LangChain](https://python.langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/) teams for the orchestration framework
- [Groq](https://groq.com) for fast open-source LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Research) for vector similarity search
- [Hugging Face](https://huggingface.co) for the sentence-transformers embedding model
