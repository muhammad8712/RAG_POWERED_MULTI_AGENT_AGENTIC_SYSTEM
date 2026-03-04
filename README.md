# RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM


```markdown
# 🧠 RAG-Powered Multi-Agent ERP Assistant  
### Agentic + Corrective RAG Architecture with LangGraph

<p align="center">
  <b>Multi-Agent Conversational System for Enterprise ERP Workflows</b><br>
  Retrieval-Augmented Generation · Structured Data Querying · Corrective Validation · Explainability
</p>

---

## 🚀 Overview

This project implements a **RAG-powered multi-agent conversational assistant for ERP systems**.

The system intelligently answers questions using:

- 📄 **Policy Documents (PDFs)** via FAISS vector search  
- 🗄️ **Structured ERP Database (SQLite)**  
- 🌐 **External APIs** (read-only GET integration)  
- 🧠 **Multi-agent reasoning** for composite queries  
- ✅ **Corrective validation loop** to prevent unsupported responses  
- 🔎 **Explainability layer** with full execution trace  

Built with **LangGraph**, this system demonstrates a production-style Agentic RAG pipeline suitable for enterprise knowledge workflows.

---

# 🏗️ System Architecture

## Execution Flow

```

User Query
↓
Intent Classifier
↓
Orchestrator (Plan Generator)
↓
Worker Agents (Document / Database / API)
↓
Reasoning Agent (if needed)
↓
Validator (Corrective Loop)
↓
Explainability Layer
↓
Final Response

````

---

# 📁 Project Structure

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
````

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/muhammad8712/RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM.git
cd RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM
```

---

## 2️⃣ Create Virtual Environment

### Using Conda

```bash
conda create -n myenv python=3.11
conda activate myenv
```

### Using venv

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r Requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

# 🗄️ Database Setup

Generate synthetic ERP data:

```bash
python data/generate_data.py
```

This creates:

```
erp.db
```

Tables included:

* vendors
* invoices
* purchase_orders
* payments

---

# 📄 Policy Document Setup

Generate mock ERP policies:

```bash
python policies/generate_policies.py
```

PDFs will be created inside:

```
policies/
```

---

# 🔍 Document Ingestion (Vector Store)

Build the FAISS index:

```bash
python document_ingestion.py
```

This generates:

```
storage/vector_store/
  ├── index.faiss
  └── index.pkl
```

⚠️ Must be executed before running the assistant.

---

# 🖥️ Running the Application

## 🌐 Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

### UI Features

* Natural language query input
* Document citation display
* Similarity scores
* SQL query + result table
* Validation report
* Execution trace
* Downloadable JSON output

---

## 💻 Command Line Interface

```bash
python main_cli.py
```

---

# 🧪 Evaluation Pipeline

Run benchmark queries:

```bash
python evaluation/run_eval.py
```

Results saved to:

```
evaluation/results/
```

Each result contains:

* Query ID
* Query type
* Final response
* Full orchestration trace

---

# 🧠 Core System Components

## 📄 Document Agent

* Vector retrieval via FAISS
* Top-k chunk selection
* Context-grounded generation
* Source + similarity tracking

## 🗄️ Database Agent

* LLM-to-SQL conversion
* SELECT-only enforcement
* SQL injection protection
* Structured result output

## 🌐 API Agent

* Safe GET-only calls
* Optional allowlist restriction
* Structured JSON responses

## 🧠 Reasoning Agent

* Multi-source synthesis
* Combines structured + unstructured evidence
* Produces final decision + explanation

## ✅ Validator Agent (Corrective RAG)

* Checks evidence sufficiency
* Detects unsupported numeric claims
* Triggers corrective tool execution
* Iteration-limited retry loop

## 🔎 Explainability Agent

* Final answer packaging
* Lists agents used
* Includes:

  * Document sources
  * SQL query
  * Similarity scores
  * Validation result
  * Execution trace

---

# 📊 Logging & Monitoring

Logs stored as JSONL:

```
logs/events.jsonl
logs/validation.jsonl
```

Each log entry includes:

* Timestamp
* Query
* Final response
* Validation status

Useful for:

* Performance evaluation
* Failure analysis
* Research metrics

---

# 🛡️ Safety & Robustness

* SQL injection protection
* Restricted SQL operations (SELECT-only)
* API allowlist validation
* Evidence-backed validation loop
* Bounded corrective iteration
* Transparent execution trace

---

# 🌐 Deployment

> ⚠ GitHub Pages cannot host Streamlit apps (static only).

## Recommended: Streamlit Community Cloud

1. Push project to GitHub
2. Open Streamlit Community Cloud
3. Create new app
4. Select repository + branch
5. Set main file: `streamlit_app.py`
6. Add secret:

```
GROQ_API_KEY = "your_key_here"
```

Ensure:

* Database exists (generate during deployment)
* Vector store is built (run ingestion during deployment)

---

# 🎓 Research Context

This project demonstrates:

* Agentic RAG Architecture
* Corrective Validation Loop
* Multi-Agent Orchestration with LangGraph
* Evidence-Grounded Reasoning
* Explainability for Enterprise AI Systems

