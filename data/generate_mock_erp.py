from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "erp.db"
SCHEMA_PATH = ROOT / "data" / "schema.sql"

random.seed(42)

COUNTRIES = ["HU", "DE", "AT", "PL", "RO", "CZ", "SK", "IT", "FR", "NL"]
CURRENCIES = ["EUR", "USD", "HUF"]
CREDIT_RATINGS = ["A", "B", "C", "D"]
PAYMENT_METHODS = ["BANK_TRANSFER", "CARD", "CASH", "CHECK"]
INVOICE_STATUSES = ["OPEN", "PAID", "OVERDUE", "PARTIALLY_PAID", "DISPUTED"]
PO_STATUSES = ["OPEN", "APPROVED", "DELIVERED", "CANCELLED", "LATE"]

VENDOR_PREFIX = ["Alpha", "Beta", "Gamma", "Delta", "Omni", "Nova", "Apex", "Zen"]
VENDOR_SUFFIX = ["Supply", "Trading", "Services", "Logistics", "Manufacturing", "Consulting"]


def rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


def money(mu: float = 1200, sigma: float = 900, min_val: float = 50, max_val: float = 25000) -> float:
    val = random.gauss(mu, sigma)
    return round(max(min_val, min(max_val, val)), 2)


def main(vendors_n: int = 50, invoices_n: int = 600, pos_n: int = 250) -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema.sql at {SCHEMA_PATH}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        today = date.today()
        two_years_ago = today - timedelta(days=730)

        vendors: list[tuple[str, str, str, str, str]] = []
        for i in range(1, vendors_n + 1):
            vendor_id = f"V{i:04d}"
            vendor_name = f"{random.choice(VENDOR_PREFIX)} {random.choice(VENDOR_SUFFIX)} {i}"
            country = random.choice(COUNTRIES)
            credit = random.choices(CREDIT_RATINGS, weights=[45, 30, 18, 7])[0]
            onboarding = rand_date(two_years_ago, today - timedelta(days=30)).isoformat()
            vendors.append((vendor_id, vendor_name, country, credit, onboarding))

        conn.executemany(
            "INSERT INTO vendors(vendor_id, vendor_name, country, credit_rating, onboarding_date) VALUES (?,?,?,?,?)",
            vendors,
        )

        vendor_ids = [v[0] for v in vendors]

        invoices: list[tuple[str, str, float, str, str, str, str, str | None]] = []
        payments: list[tuple[str, str, float, str, str]] = []

        for i in range(1, invoices_n + 1):
            invoice_id = f"INV{i:06d}"
            vendor_id = random.choice(vendor_ids)

            issue = rand_date(today - timedelta(days=365), today)
            term_days = random.choices([15, 30, 45, 60], weights=[15, 55, 20, 10])[0]
            due = issue + timedelta(days=term_days)

            amount = money()
            currency = random.choices(CURRENCIES, weights=[75, 20, 5])[0]
            status = random.choices(INVOICE_STATUSES, weights=[20, 50, 15, 10, 5])[0]

            payment_date: str | None = None

            if status == "PAID":
                payment_date = rand_date(issue, min(today, due + timedelta(days=20))).isoformat()
                payments.append(
                    (f"PAY{i:06d}", invoice_id, amount, payment_date, random.choice(PAYMENT_METHODS))
                )
            elif status == "PARTIALLY_PAID":
                partial = round(amount * random.uniform(0.2, 0.8), 2)
                payment_date = rand_date(issue, min(today, due + timedelta(days=40))).isoformat()
                payments.append(
                    (f"PAY{i:06d}", invoice_id, partial, payment_date, random.choice(PAYMENT_METHODS))
                )
            elif status == "OVERDUE":
                if due >= today:
                    issue = today - timedelta(days=random.randint(60, 200))
                    due = issue + timedelta(days=term_days)

            invoices.append(
                (invoice_id, vendor_id, amount, currency, issue.isoformat(), due.isoformat(), status, payment_date)
            )

        conn.executemany(
            "INSERT INTO invoices(invoice_id, vendor_id, amount, currency, issue_date, due_date, status, payment_date) VALUES (?,?,?,?,?,?,?,?)",
            invoices,
        )

        pos: list[tuple[str, str, str, float, str, str, str | None]] = []
        for i in range(1, pos_n + 1):
            po_id = f"PO{i:06d}"
            vendor_id = random.choice(vendor_ids)
            order_dt = rand_date(today - timedelta(days=365), today)

            amount = money(mu=3000, sigma=2500, min_val=100, max_val=80000)
            status = random.choices(PO_STATUSES, weights=[20, 25, 35, 10, 10])[0]

            lead = random.choices([7, 14, 21, 30, 45], weights=[20, 35, 20, 15, 10])[0]
            expected = order_dt + timedelta(days=lead)

            actual: str | None = None
            if status == "DELIVERED":
                actual = rand_date(order_dt + timedelta(days=3), expected + timedelta(days=15)).isoformat()
            elif status == "LATE":
                actual = (expected + timedelta(days=random.randint(5, 40))).isoformat()

            pos.append((po_id, vendor_id, order_dt.isoformat(), amount, status, expected.isoformat(), actual))

        conn.executemany(
            "INSERT INTO purchase_orders(po_id, vendor_id, order_date, amount, status, expected_delivery_date, actual_delivery_date) VALUES (?,?,?,?,?,?,?)",
            pos,
        )

        conn.executemany(
            "INSERT INTO payments(payment_id, invoice_id, amount, payment_date, method) VALUES (?,?,?,?,?)",
            payments,
        )

        conn.commit()

        cur = conn.cursor()
        print("Created", DB_PATH)
        for table in ("vendors", "invoices", "purchase_orders", "payments"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"{table}: {cur.fetchone()[0]}")

        cur.execute("SELECT COUNT(*) FROM invoices WHERE status='OVERDUE'")
        print("overdue_invoices:", cur.fetchone()[0])

    finally:
        conn.close()


if __name__ == "__main__":
    main()