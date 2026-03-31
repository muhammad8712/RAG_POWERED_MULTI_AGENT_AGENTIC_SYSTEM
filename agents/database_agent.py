from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import text


def _format_history(history: list[dict]) -> str:
    """
    Convert conversation history list into a compact plain-text block
    for injection into the SQL generation prompt.

    All turns (up to the last 10) are used to keep full context.
    """
    if not history:
        return "None"

    recent = history[-10:]
    lines: list[str] = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = str(turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "None"


class DatabaseAgent:
    def __init__(self, db_engine, llm):
        self.engine = db_engine
        self.llm = llm

        self.schema_description = """
Tables (SQLite):

customers(
  customer_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE,
  country TEXT,
  age INTEGER,
  signup_date TEXT,
  marketing_opt_in BOOLEAN
)

products(
  product_id INTEGER PRIMARY KEY,
  category TEXT,
  name TEXT NOT NULL,
  price_usd REAL,
  cost_usd REAL,
  margin_usd REAL
)

orders(
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  order_time TEXT,
  payment_method TEXT,
  discount_pct REAL,
  subtotal_usd REAL,
  total_usd REAL,
  country TEXT,
  device TEXT,
  source TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)

order_items(
  order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  unit_price_usd REAL,
  quantity INTEGER,
  line_total_usd REAL,
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
)

payments(
  transaction_id INTEGER PRIMARY KEY,
  payment_date TEXT,
  customer_id INTEGER NOT NULL,
  amount REAL,
  type TEXT,
  description TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)

sales_orders_odoo(
  sales_order_ref TEXT PRIMARY KEY,
  creation_date TEXT,
  customer_name TEXT,
  salesperson TEXT,
  company TEXT,
  total REAL,
  status TEXT
)

purchase_orders_odoo(
  purchase_order_ref TEXT PRIMARY KEY,
  priority TEXT,
  vendor_name TEXT,
  company TEXT,
  buyer TEXT,
  order_deadline TEXT,
  total REAL,
  status TEXT
)

Valid relationships:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_items.product_id = products.product_id
- payments.customer_id = customers.customer_id

Important limitations:
- payments are linked to customer_id, not order_id
- do not invent columns
- do not invent joins
- do not join Odoo export tables to relational tables unless the relationship is explicit in the schema
- document/policy questions should not be answered from the database if no structured evidence exists
""".strip()

        # ── prompt now includes optional conversation history ─────────────────
        self.prompt = ChatPromptTemplate.from_template(
            """
You generate exactly one SQLite SELECT query using only the schema below.

Schema:
{schema}

Conversation History (most recent turns, for context resolution):
{history}

CONTEXT RESOLUTION RULES:
- If the current question uses pronouns or references like "that", "those", "it", "them",
  "the same", "filter that", "now show", resolve them using the conversation history above.
- Example: history says "top 5 customers by revenue", current question says
  "now filter that by Germany" → generate SQL for top customers by revenue WHERE country = 'Germany'.
- Example: history says "show orders", current question says "how many were from mobile?" →
  generate SQL for COUNT(*) FROM orders WHERE device = 'mobile'.
- If history is "None" or irrelevant, treat the question as standalone.

Rules:
- Output only SQL.
- One statement only.
- SELECT only.
- Use only tables and columns that exist in the schema above.
- Do NOT use tables named: vendors, invoices, contracts, purchase_lines, budgets, accounts.
  These do not exist. Return: SELECT 'INSUFFICIENT_DB_EVIDENCE' AS message if needed.
- Do not invent relationships.
- Do not infer payment-to-order matching.
- Do not use markdown.
- Do not add a trailing semicolon.

PLURAL vs SINGULAR:
- Questions using plural nouns ("which customers", "which products", "which countries", "top selling",
  "most common", "most frequently") want a ranked list — use LIMIT 10.
- Only use LIMIT 1 when explicitly asked for a single result.

COLUMN RULES — avoid these common mistakes:
- order_items does NOT have a 'category' column. To get category data, JOIN with products:
  SELECT p.category, SUM(oi.line_total_usd) FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category
- products does NOT have 'line_total_usd'. That column belongs to order_items.
- sales_orders_odoo does NOT have a 'country' column. For sales by country use the orders table.
- Do NOT join orders.customer_id = products.product_id — those are unrelated keys.

COMPOSITE QUESTIONS:
- When a question mixes database facts with policy/document questions, generate SQL for the DB part only.
- "Which customers made the highest payments AND what is the payment term policy?"
  → SELECT c.name, SUM(p.amount) AS total FROM customers c JOIN payments p ON c.customer_id = p.customer_id GROUP BY c.customer_id ORDER BY total DESC LIMIT 10
- "Top selling products AND explain the invoice tolerance rule"
  → SELECT p.name, SUM(oi.line_total_usd) AS rev FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.name ORDER BY rev DESC LIMIT 10
- "Which payment methods are most common AND what is the grace period?"
  → SELECT payment_method, COUNT(*) AS cnt FROM orders GROUP BY payment_method ORDER BY cnt DESC LIMIT 10
- "Recent customer orders AND explain the purchase approval process"
  → SELECT o.order_id, c.name, o.order_time, o.total_usd FROM orders o JOIN customers c ON o.customer_id = c.customer_id ORDER BY o.order_time DESC LIMIT 10
- "Which customers generate the most revenue AND what vendor documents are required?"
  → SELECT c.name, SUM(o.total_usd) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY revenue DESC LIMIT 10

VALID STATUS VALUES:
- sales_orders_odoo.status: 'Sales Order', 'Quotation', 'Quotation Sent'
- purchase_orders_odoo.status: 'Purchase Order', 'Draft Purchase Order', 'Request for Quotation'
- Do NOT use status = 'confirmed' — that value does not exist.

CO-PURCHASE ANALYSIS:
- "Which products are purchased together most often" requires a self-join:
  SELECT p1.name, p2.name, COUNT(*) AS cnt
  FROM order_items oi1
  JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
  JOIN products p1 ON oi1.product_id = p1.product_id
  JOIN products p2 ON oi2.product_id = p2.product_id
  GROUP BY p1.product_id, p2.product_id ORDER BY cnt DESC LIMIT 10

- If the question is not answerable from the schema, return:
SELECT 'INSUFFICIENT_DB_EVIDENCE' AS message

Non-database questions (return INSUFFICIENT_DB_EVIDENCE):
- What is the standard payment term? / grace period? / late fee? / tolerance? / approval required?
- What is the invoice matching tolerance? / price variance tolerance?
- What are the procurement rules? / vendor onboarding requirements?

Question:
{question}
""".strip()
        )

        self.chain = self.prompt | self.llm

        self.allowed_tables = {
            "customers",
            "products",
            "orders",
            "order_items",
            "payments",
            "sales_orders_odoo",
            "purchase_orders_odoo",
        }

    def generate_sql(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        response = self.chain.invoke(
            {
                "schema": self.schema_description,
                "history": _format_history(conversation_history or []),
                "question": query,
            }
        )
        sql = (response.content or "").strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        sql_stripped = (sql or "").strip()
        sql_upper = sql_stripped.upper()

        if not sql_stripped:
            return False, "Empty SQL generated."

        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries allowed"

        body = re.sub(r";\s*$", "", sql_stripped)
        if ";" in body:
            return False, "Multiple statements detected"

        forbidden_patterns = (
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bUPDATE\b",
            r"\bINSERT\b",
            r"\bALTER\b",
            r"\bTRUNCATE\b",
            r"\bCREATE\b",
            r"\bPRAGMA\b",
            r"\bATTACH\b",
            r"\bDETACH\b",
        )

        for pattern in forbidden_patterns:
            if re.search(pattern, sql_stripped, flags=re.IGNORECASE):
                return False, f"Forbidden SQL detected: {pattern}"

        referenced_tables = set(
            match.group(1).lower()
            for match in re.finditer(
                r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                sql_stripped,
                flags=re.IGNORECASE,
            )
        )

        if not referenced_tables:
            return False, "No known table referenced"

        unknown_tables = referenced_tables - self.allowed_tables
        if unknown_tables:
            return False, f"Unknown table(s) referenced: {sorted(unknown_tables)}"

        sql_lower = sql_stripped.lower()

        if "orders" in referenced_tables and "payments" in referenced_tables:
            if "customer_id" not in sql_lower:
                return False, "Orders and payments may only be related through customer_id"

        bad_patterns = [
            r"sales_orders_odoo\s+\w*\s*on\s+.*product_id.*sales_order_ref",
            r"payments\s+\w*\s*on\s+.*order_id",
            r"customer_name\s*=\s*\w+\.customer_id",
            r"\bvalue\b",
        ]
        for pattern in bad_patterns:
            if re.search(pattern, sql_lower, flags=re.IGNORECASE):
                return False, f"Likely hallucinated SQL pattern detected: {pattern}"

        hallucinated_column_patterns = [
            (r"(?:oi|order_items)\s*\.\s*category", "order_items has no 'category' column; join with products"),
            (r"(?:p|products)\s*\.\s*line_total_usd", "products has no 'line_total_usd' column; use order_items"),
            (r"(?:sales_orders_odoo|\bsoo\b|\bso\b)\s*\.\s*country", "sales_orders_odoo has no 'country' column"),
            (r"on\s+\w+\.customer_id\s*=\s*\w+\.product_id", "Wrong join: customer_id != product_id"),
            (r"on\s+\w+\.product_id\s*=\s*\w+\.customer_id", "Wrong join: product_id != customer_id"),
        ]
        for pattern, msg in hallucinated_column_patterns:
            if re.search(pattern, sql_lower, flags=re.IGNORECASE):
                return False, f"Hallucinated column or join: {msg}"

        if (re.search(r"\bfrom\s+order_items\b", sql_lower) and
                not re.search(r"\bjoin\s+products\b", sql_lower) and
                re.search(r"(?<!p\.)(?<!products\.)\bcategory\b", sql_lower)):
            return False, "Hallucinated column: 'category' must come from products via JOIN, not directly from order_items"

        return True, "Valid"

    def execute_sql(self, sql: str) -> list[dict[str, Any]] | dict[str, str]:
        try:
            final_sql = re.sub(r";\s*$", "", (sql or "").strip()).strip()

            if "LIMIT" not in final_sql.upper():
                final_sql += "\nLIMIT 50"

            with self.engine.connect() as conn:
                result = conn.execute(text(final_sql))
                rows = result.fetchall()
                columns = result.keys()

            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            return {"error": str(e)}

    def run(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        sql_query = self.generate_sql(query, conversation_history=conversation_history)
        is_valid, message = self.validate_sql(sql_query)

        if not is_valid:
            return {
                "sql_query": sql_query,
                "error": message,
                "result": [],
                "row_count": 0,
            }

        result = self.execute_sql(sql_query)

        if isinstance(result, dict) and "error" in result:
            return {
                "sql_query": sql_query,
                "error": result["error"],
                "result": [],
                "row_count": 0,
            }

        return {
            "sql_query": sql_query,
            "error": None,
            "result": result,
            "row_count": len(result),
        }