from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import text


class DatabaseAgent:
    def __init__(self, db_engine, llm):
        self.engine = db_engine
        self.llm = llm

        self.schema_description = """
Tables (SQLite):

vendors(
  vendor_id TEXT PRIMARY KEY,
  vendor_name TEXT,
  country TEXT,
  credit_rating TEXT,
  onboarding_date TEXT
)

invoices(
  invoice_id TEXT PRIMARY KEY,
  vendor_id TEXT,
  amount REAL,
  currency TEXT,
  issue_date TEXT,
  due_date TEXT,
  status TEXT,
  payment_date TEXT,
  FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
)

purchase_orders(
  po_id TEXT PRIMARY KEY,
  vendor_id TEXT,
  order_date TEXT,
  amount REAL,
  status TEXT,
  expected_delivery_date TEXT,
  actual_delivery_date TEXT,
  FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
)

payments(
  payment_id TEXT PRIMARY KEY,
  invoice_id TEXT,
  amount REAL,
  payment_date TEXT,
  method TEXT,
  FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
)
""".strip()

        self.prompt = ChatPromptTemplate.from_template(
            """
You generate a single SQLite SELECT query using only the schema below.

Schema:
{schema}

Constraints:
- Output only SQL (no markdown, no explanation).
- One statement only.
- SELECT only (no INSERT/UPDATE/DELETE/DDL/PRAGMA).
- Prefer explicit columns when reasonable.

Question:
{question}
""".strip()
        )

        self.chain = self.prompt | self.llm

    def generate_sql(self, query: str) -> str:
        response = self.chain.invoke({"schema": self.schema_description, "question": query})
        sql = (response.content or "").strip()
        return sql.replace("```sql", "").replace("```", "").strip()

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        sql_stripped = (sql or "").strip()
        sql_upper = sql_stripped.upper()
        sql_lower = sql_stripped.lower()

        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries allowed."

        if ";" in sql_stripped.rstrip().rstrip(";"):
            return False, "Multiple SQL statements not allowed."

        forbidden = (
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "PRAGMA",
            "ATTACH",
            "DETACH",
        )
        for word in forbidden:
            if word in sql_upper:
                return False, f"Forbidden keyword: {word}"

        allowed_tables = ("vendors", "invoices", "purchase_orders", "payments")
        if not any(t in sql_lower for t in allowed_tables):
            return False, "No known table referenced."

        return True, "Valid"

    def execute_sql(self, sql: str) -> list[dict[str, Any]] | dict[str, str]:
        try:
            final_sql = (sql or "").strip()
            if "LIMIT" not in final_sql.upper():
                final_sql += " LIMIT 50"

            with self.engine.connect() as conn:
                result = conn.execute(text(final_sql))
                rows = result.fetchall()
                columns = result.keys()

            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return {"error": str(e)}

    def run(self, query: str) -> dict[str, Any]:
        sql_query = self.generate_sql(query)
        is_valid, message = self.validate_sql(sql_query)

        if not is_valid:
            return {"sql_query": sql_query, "error": message, "result": [], "row_count": 0}

        result = self.execute_sql(sql_query)

        if isinstance(result, dict) and "error" in result:
            return {"sql_query": sql_query, "error": result["error"], "result": [], "row_count": 0}

        return {"sql_query": sql_query, "error": None, "result": result, "row_count": len(result)}