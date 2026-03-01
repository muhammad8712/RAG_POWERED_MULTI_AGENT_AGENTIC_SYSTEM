#RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM

````markdown
# RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM

## Overview

This project implements a **RAG-powered multi-agent conversational assistant for ERP systems**.

The assistant can answer questions using:

- 📄 Policy documents (PDFs) via vector search (FAISS)
- 🗄️ Structured ERP database (SQLite)
- 🌐 External APIs (read-only GET requests)
- 🧠 Multi-agent reasoning for composite queries
- ✅ Corrective validation loop to prevent unsupported answers
- 🔎 Explainability layer with traceability and source reporting

The architecture is built using **LangGraph**, combining orchestration, retrieval, database querying, validation, and explainability.

---

## System Architecture

Workflow:

1. User Query
2. Intent Classifier
3. Orchestrator (Plan Generator)
4. Worker Agents (Document / Database / API)
5. Reasoning Agent (if needed)
6. Validator (Corrective Loop)
7. Explainability Layer
8. Final Response

---

## Project Structure

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
  (PDF files go here)
  generate_policies.py

storage/
  vector_store/  (generated)

document_ingestion.py
streamlit_app.py
main_cli.py
requirements.txt
README.md
````

---

## Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/muhammad8712/RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM.git
cd RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM
```

### 2) Create and activate a virtual environment

#### Option A: Conda

```bash
conda create -n myenv python=3.11
conda activate myenv
```

#### Option B: venv (recommended if you do not use Conda)

```bash
python -m venv myenv
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Set environment variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

**Do not commit** `.env` to GitHub.

---

## Database Setup (SQLite)

Generate synthetic ERP data:

```bash
python data/generate_data.py
```

This creates:

* `erp.db`

Tables:

* `vendors`
* `invoices`
* `purchase_orders`
* `payments`

---

## Policies (Mock PDFs)

Generate mock policy PDFs:

```bash
python policies/generate_policies.py
```

This creates PDFs inside:

* `policies/`

---

## Document Ingestion (Vector Store)

Build the FAISS vector index:

```bash
python document_ingestion.py
```

This creates:

```text
storage/vector_store/
  index.faiss
  index.pkl
```

**Note:** Ingestion must be done before running the assistant.

---

## Running the Application

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

UI features:

* Query input
* Answer display
* Document sources + similarity scores
* SQL query + database results
* Validation output
* Execution trace
* JSON download

### Option 2: Command Line Interface

```bash
python main_cli.py
```

Enter your ERP query when prompted.

---

## Running Evaluation

Run benchmark queries:

```bash
python evaluation/run_eval.py
```

Results are saved to:

```text
evaluation/results/
```

Each result contains:

* Query ID
* Query type
* Final response
* Full orchestration output

---

## Core Components

### Document Agent

* Performs FAISS similarity search
* Retrieves top-k document chunks
* Generates grounded answers using retrieved context
* Returns sources and similarity scores

### Database Agent

* Converts user question → SQL query
* Enforces SELECT-only queries
* Blocks unsafe SQL keywords
* Returns structured row results

### API Agent

* Safe GET-only requests
* Optional allowlist validation
* Structured response

### Reasoning Agent

* Combines document + database evidence
* Produces final decision and explanation
* Used for composite queries

### Validator Agent (Corrective RAG)

* Checks if answer is supported by evidence
* Detects:

  * Empty retrieval
  * Low similarity scores
  * Missing DB results
  * Unsupported numeric claims
* Can trigger corrective tool re-execution
* Stops after configurable iteration limit

### Explainability Agent

* Packages final answer
* Lists agents used
* Includes:

  * Document sources
  * Similarity scores
  * SQL query
  * Validation result
  * Execution trace

---

## Logging

Logs are stored as JSONL:

```text
logs/events.jsonl
logs/validation.jsonl
```

Each entry may contain:

* Timestamp
* Query
* Final response
* Validation status

Used for evaluation and analysis.

---

## Safety Features

* SQL injection protection
* SELECT-only enforcement
* Blocked unsafe SQL keywords
* API allowlist support
* Evidence-backed validation
* Iteration-limited corrective loop

---

## Deployment (Streamlit Community Cloud)

**Note:** GitHub Pages cannot host Streamlit (Streamlit is not a static site).

Steps:

1. Push the project to GitHub
2. Go to Streamlit Community Cloud
3. Create a new app
4. Select repo + branch
5. Set main file: `streamlit_app.py`
6. Add secrets in Streamlit Cloud settings:

```text
GROQ_API_KEY = "your_key_here"
```

**Important:** Your deployment must have access to:

* `erp.db` (generate in deployment or ship a demo DB)
* `storage/vector_store/` (build in deployment by running ingestion)

---

## Research Context

This system demonstrates:

* Agentic RAG architecture
* Corrective validation loop
* Multi-agent orchestration
* Evidence-grounded reasoning
* Explainability in enterprise QA systems

````
