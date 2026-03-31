from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "erp.db"
SCHEMA_PATH = ROOT / "data" / "schema.sql"
DATA_DIR = ROOT / "data"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def load_excel(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_excel(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def prepare_customers() -> pd.DataFrame:
    df = clean_columns(load_csv("customers.csv"))
    return df[
        [
            "customer_id",
            "name",
            "email",
            "country",
            "age",
            "signup_date",
            "marketing_opt_in",
        ]
    ]


def prepare_products() -> pd.DataFrame:
    df = clean_columns(load_csv("products.csv"))
    return df[
        [
            "product_id",
            "category",
            "name",
            "price_usd",
            "cost_usd",
            "margin_usd",
        ]
    ]


def prepare_orders() -> pd.DataFrame:
    df = clean_columns(load_csv("orders.csv"))
    return df[
        [
            "order_id",
            "customer_id",
            "order_time",
            "payment_method",
            "discount_pct",
            "subtotal_usd",
            "total_usd",
            "country",
            "device",
            "source",
        ]
    ]


def prepare_order_items() -> pd.DataFrame:
    df = clean_columns(load_csv("order_items.csv"))

    # order_item_id is AUTOINCREMENT in SQLite, so do not insert it
    return df[
        [
            "order_id",
            "product_id",
            "unit_price_usd",
            "quantity",
            "line_total_usd",
        ]
    ]


def prepare_payments() -> pd.DataFrame:
    df = clean_columns(load_csv("payments.csv"))

    # Support either "date" or "payment_date"
    if "date" in df.columns and "payment_date" not in df.columns:
        df = df.rename(columns={"date": "payment_date"})

    return df[
        [
            "transaction_id",
            "payment_date",
            "customer_id",
            "amount",
            "type",
            "description",
        ]
    ]


def prepare_sales_orders_odoo() -> pd.DataFrame:
    df = clean_columns(load_excel("Sales Order.xlsx"))

    df = df.rename(
        columns={
            "order_reference": "sales_order_ref",
            "creation_date": "creation_date",
            "customer": "customer_name",
            "salesperson": "salesperson",
            "company": "company",
            "total": "total",
            "status": "status",
        }
    )

    return df[
        [
            "sales_order_ref",
            "creation_date",
            "customer_name",
            "salesperson",
            "company",
            "total",
            "status",
        ]
    ]


def prepare_purchase_orders_odoo() -> pd.DataFrame:
    df = clean_columns(load_excel("Purchase Order.xlsx"))

    df = df.rename(
        columns={
            "order_reference": "purchase_order_ref",
            "priority": "priority",
            "vendor": "vendor_name",
            "company": "company",
            "buyer": "buyer",
            "order_deadline": "order_deadline",
            "total": "total",
            "status": "status",
        }
    )

    return df[
        [
            "purchase_order_ref",
            "priority",
            "vendor_name",
            "company",
            "buyer",
            "order_deadline",
            "total",
            "status",
        ]
    ]


def main() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        customers = prepare_customers()
        products = prepare_products()
        orders = prepare_orders()
        order_items = prepare_order_items()
        payments = prepare_payments()
        sales_orders = prepare_sales_orders_odoo()
        purchase_orders = prepare_purchase_orders_odoo()

        customers.to_sql("customers", conn, if_exists="append", index=False)
        products.to_sql("products", conn, if_exists="append", index=False)
        orders.to_sql("orders", conn, if_exists="append", index=False)
        order_items.to_sql("order_items", conn, if_exists="append", index=False)
        payments.to_sql("payments", conn, if_exists="append", index=False)
        sales_orders.to_sql("sales_orders_odoo", conn, if_exists="append", index=False)
        purchase_orders.to_sql("purchase_orders_odoo", conn, if_exists="append", index=False)

        conn.commit()

        print("Database created:", DB_PATH)
        print("customers:", len(customers))
        print("products:", len(products))
        print("orders:", len(orders))
        print("order_items:", len(order_items))
        print("payments:", len(payments))
        print("sales_orders_odoo:", len(sales_orders))
        print("purchase_orders_odoo:", len(purchase_orders))

    finally:
        conn.close()


if __name__ == "__main__":
    main()