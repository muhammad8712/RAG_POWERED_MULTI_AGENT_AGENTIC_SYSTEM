# RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM

📌 Overview
This project implements a RAG-powered multi-agent conversational assistant for ERP systems.
The system can answer questions using:
•	📄 Policy documents (PDFs) via vector search (FAISS)
•	🗄️ Structured ERP database (SQLite)
•	🌐 External APIs (read-only GET requests)
•	🧠 Multi-agent reasoning for composite queries
•	✅ Corrective validation loop to prevent unsupported answers
•	🔎 Explainability layer with traceability and source reporting
The architecture is built using LangGraph, combining agent orchestration, retrieval augmentation, database querying, validation, and explainability.
________________________________________
🏗️ System Architecture
The system workflow:
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
________________________________________
📁 Project Structure
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
________________________________________
🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/muhammad8712/RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM.git
cd RAG_POWERED_MULTI_AGENT_AGENTIC_SYSTEM
________________________________________
2️⃣ Create Virtual Environment
conda create --name myenv python=3.13
conda activate myenv\
________________________________________
3️⃣ Install Dependencies
pip install -r Requirements.txt
________________________________________
4️⃣ Set Environment Variables
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
________________________________________
🗄️ Database Setup
Generate synthetic ERP data:
python data/generate_data.py
This creates:
erp.db
Tables:
•	vendors
•	invoices
•	purchase_orders
•	payments
________________________________________
📄 Document Ingestion (Vector Store)
Place your ERP policy PDFs inside:
policies/
Then build the FAISS vector index:
python document_ingestion.py
This creates:
storage/vector_store/
    index.faiss
    index.pkl
⚠️ This must be run before launching the assistant.
________________________________________
🖥️ Running the Application
Option 1: Streamlit Web Interface (Recommended)
streamlit run streamlit_app.py
Features:
•	Query input
•	Answer display
•	Document sources
•	Similarity scores
•	SQL query + database results
•	Validation output
•	Execution trace
•	JSON download
________________________________________
Option 2: Command Line Interface
python main_cli.py
Enter your ERP query when prompted.
________________________________________
🧪 Running Evaluation
To run benchmark queries:
python evaluation/run_eval.py
Results are saved to:
evaluation/results/
Each result contains:
•	Query ID
•	Query type
•	Final response
•	Full orchestration output
________________________________________
🔍 Core Components
📄 Document Agent
•	Performs FAISS similarity search
•	Retrieves top-k document chunks
•	Uses RAG to generate grounded answers
•	Returns sources and similarity scores
________________________________________
🗄️ Database Agent
•	Converts user question → SQL query
•	Enforces SELECT-only queries
•	Blocks unsafe SQL
•	Returns structured row results
________________________________________
🌐 API Agent
•	Safe GET-only requests
•	Optional allowlist validation
•	Structured JSON response
________________________________________
🧠 Reasoning Agent
•	Combines document + database evidence
•	Produces final decision and explanation
•	Used for composite queries
________________________________________
✅ Validator Agent (Corrective RAG)
•	Checks if answer is supported by evidence
•	Detects:
o	Empty retrieval
o	Low similarity scores
o	Missing DB results
o	Unsupported numeric claims
•	Can trigger corrective tool re-execution
•	Stops after configurable iteration limit
________________________________________
🔎 Explainability Agent
•	Packages final answer
•	Lists agents used
•	Includes:
o	Document sources
o	Similarity scores
o	SQL query
o	Validation result
o	Execution trace
________________________________________
📊 Logging
Logs are stored as JSONL:
logs/events.jsonl
logs/validation.jsonl
Each entry contains:
•	Timestamp
•	Query
•	Final response
•	Validation status
Used for evaluation and analysis.
________________________________________
🛡️ Safety Features
•	SQL injection protection
•	SELECT-only enforcement
•	Blocked unsafe SQL keywords
•	API allowlist
•	Evidence-backed validation
•	Iteration-limited corrective loop
________________________________________
🌐 Deployment
Streamlit Community Cloud (Recommended)
1.	Push project to GitHub
2.	Go to Streamlit Community Cloud
3.	Create new app
4.	Select repo + branch
5.	Set main file: streamlit_app.py
6.	Add secret:
GROQ_API_KEY = your_key_here
⚠️ Ensure database and vector store exist or build them during deployment.
________________________________________
🎓 Research Context
This system demonstrates:
•	Agentic RAG architecture
•	Corrective validation loop
•	Multi-agent orchestration
•	Evidence-grounded reasoning
•	Explainability in enterprise QA systems
