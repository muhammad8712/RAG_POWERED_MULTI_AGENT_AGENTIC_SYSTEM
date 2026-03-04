````markdown
# 🧠 RAG-Powered Multi-Agent ERP Assistant
**Agentic + Corrective RAG with LangGraph · Documents (FAISS) · SQL (SQLite) · APIs · Explainability**

<p align="center">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Orchestration-blue">
  <img alt="RAG" src="https://img.shields.io/badge/RAG-FAISS%20Vector%20Search-purple">
  <img alt="SQLite" src="https://img.shields.io/badge/Database-SQLite-lightgrey">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b">
</p>

---

## 📌 Overview

This repository contains a **RAG-powered multi-agent conversational assistant** designed for **ERP-style workflows**.  
It answers user questions by selecting the right tools (documents, database, APIs), validating evidence, and returning a final response with traceability.

### What it can do

- 📄 **Policy / process Q&A** from PDFs using **FAISS retrieval** (RAG)
- 🗄️ **ERP data Q&A** using **SQLite** (vendors, invoices, purchase orders, payments)
- 🌐 **Read-only API calls** (GET) for system integrations
- 🧠 **Multi-agent reasoning** for composite queries (docs + database)
- ✅ **Corrective validation loop** that detects weak evidence and re-runs tools when required
- 🔎 **Explainability output**: sources, similarity scores, SQL query, validation report, and execution trace

---

## 🧱 Architecture

### Execution flow

```text
User Query
  ↓
Intent Classifier
  ↓
Orchestrator (Plan Generator)
  ↓
Worker Agents (Document / Database / API)
  ↓
Reasoning Agent (when synthesis is needed)
  ↓
Validator (Corrective Loop)
  ↓
Explainability Layer
  ↓
Final Response
````

---

## 📂 Project Structure

```text
agents/
  document_agent.py
  database_agent.py
  api_agent.py
  reasoning_agent.py
  explainability_agent.py

orchestration/
  graph.py
  orchestrator_agent.py
  intent_classifier.py
  validator_agent.py

data/
  schema.sql
  generate_data.py

evaluation/
  eval_queries.json
  run_eval.py
  results/

logs/
  logger.py

policies/
  generate_policies.py
  (generated mock PDFs)

storage/
  vector_store/  (generated)

document_ingestion.py
streamlit_app.py
main_cli.py
requirements.txt
README.md
```

---

## ⚙️ Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/muhammad8712/RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM.git
cd RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM
```

### 2) Create a virtual environment

**Conda**

```bash
conda create -n myenv python=3.11
conda activate myenv
```

**venv**

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

> Note: ensure the filename is `requirements.txt` (lowercase) in GitHub for compatibility.

### 4) Configure environment variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🗄️ Database Setup (SQLite)

Generate synthetic ERP data:

```bash
python data/generate_data.py
```

Creates:

```text
erp.db
```

Tables:

* `vendors`
* `invoices`
* `purchase_orders`
* `payments`

---

## 📄 Policy PDFs (Mock Data)

Generate mock ERP policy documents:

```bash
python policies/generate_policies.py
```

Outputs PDFs into:

```text
policies/
```

---

## 🔍 Document Ingestion (FAISS Vector Store)

Build the FAISS vector index:

```bash
python document_ingestion.py
```

Outputs:

```text
storage/vector_store/
  ├── index.faiss
  └── index.pkl
```

⚠️ This must be done before running the assistant, otherwise document retrieval will be unavailable.

---

## 🖥️ Run the Application

### 🌐 Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

UI includes:

* Query input
* Final answer
* Document sources + similarity scores
* SQL query + database table output
* Validation report
* Execution trace
* Downloadable JSON output

### 💻 CLI Runner

```bash
python main_cli.py
```

---

## 🧪 Evaluation

Run benchmark queries:

```bash
python evaluation/run_eval.py
```

Results saved to:

```text
evaluation/results/
```

Each record includes:

* Query ID + type
* Final response
* Full orchestration output (agents + trace)

---

## 🧠 Components (What each agent does)

### 📄 Document Agent

* FAISS similarity search over PDF chunks
* Returns grounded answers with sources + similarity scores

### 🗄️ Database Agent

* Converts natural language → SQL
* Enforces **SELECT-only** constraints
* Executes against SQLite and returns structured rows

### 🌐 API Agent

* Read-only GET calls
* Optional allowlist protection
* Structured JSON output

### 🧠 Reasoning Agent

* Synthesizes evidence across multiple agents
* Produces a final decision for composite queries

### ✅ Validator Agent (Corrective RAG)

* Checks evidence sufficiency
* Flags weak retrieval / empty DB / unsupported numeric claims
* Can request corrective tool execution
* Stops after a bounded iteration limit

### 🔎 Explainability Agent

* Packages the final response for UI + evaluation
* Includes:

  * Agents used
  * Document sources + similarity scores
  * SQL query
  * Validation result
  * Execution trace

---

## 📊 Logging

JSONL logs are written to:

```text
logs/events.jsonl
logs/validation.jsonl
```

Typical fields:

* timestamp
* query
* final_response
* validation status

Useful for:

* performance tracking
* error analysis
* thesis evaluation metrics

---

## 🛡️ Safety & Robustness

* SQL injection defenses + SELECT-only enforcement
* API allowlist support
* Evidence-backed validation
* Iteration-limited corrective loop
* Transparent trace output

---

## 🌍 Deployment

> ⚠️ GitHub Pages cannot host Streamlit (static-only hosting).

### ✅ Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub
2. Create a new app on Streamlit Community Cloud
3. Select:

   * Repository + branch
   * Main file: `streamlit_app.py`
4. Add secret in **App Settings → Secrets**:

```text
GROQ_API_KEY = "your_key_here"
```

Also ensure your deployment builds runtime artifacts:

* `erp.db` (run DB generator)
* `storage/vector_store/` (run ingestion)

---

## 🎓 Research Context

This project demonstrates:

* Agentic RAG orchestration with LangGraph
* Corrective validation loops to reduce hallucination
* Evidence-grounded multi-source reasoning
* Explainability and traceability for enterprise QA systems

```
```
