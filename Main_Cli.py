import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents.document_agent import DocumentAgent
from agents.database_agent import DatabaseAgent
from agents.api_agent import APIAgent
from agents.reasoning_agent import ReasoningAgent
from agents.explainability_agent import ExplainabilityAgent

from orchestration.graph import build_graph
from orchestration.orchestrator_agent import OrchestratorAgent
from orchestration.validator_agent import CorrectiveValidationAgent

from logs.logger import log_event


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set.")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.load_local(
    "storage/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

engine = create_engine("sqlite:///erp.db", echo=False)

document_agent = DocumentAgent(vector_db, llm, top_k=5)
database_agent = DatabaseAgent(engine, llm)

api_agent = APIAgent(
    base_urls=["http://localhost:8000/"],
    timeout=15,
)

reasoning_agent = ReasoningAgent(llm)
explainability_agent = ExplainabilityAgent()

orchestrator_agent = OrchestratorAgent(llm)
validator_agent = CorrectiveValidationAgent()

graph = build_graph(
    document_agent=document_agent,
    database_agent=database_agent,
    api_agent=api_agent,
    reasoning_agent=reasoning_agent,
    explainability_agent=explainability_agent,
    orchestrator_agent=orchestrator_agent,
    validator_agent=validator_agent,
)


if __name__ == "__main__":
    query = input("Enter your ERP query: ").strip()
    result = graph.invoke({"query": query})

    log_event({
        "type": "query_run",
        "query": query,
        "final_response": result.get("final_response"),
    })

    fr = result.get("final_response") or {}
    val = fr.get("validation") or {}
    if val.get("status") in ("NEEDS_MORE_INFO", "FAIL"):
        log_event(
            {
                "type": "validation_issue",
                "query": query,
                "validation": val,
            },
            filename="validation.jsonl",
        )

    print("\n========== FINAL RESPONSE ==========\n")
    print(result.get("final_response"))