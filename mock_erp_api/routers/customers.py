# mock_erp_api/routers/customers.py
"""Customer resource endpoints."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/customers", tags=["Customers"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/customers ─────────────────────────────────────────────────────
@router.get("", summary="List customers")
def list_customers(
    country: Optional[str] = Query(None, description="Filter by country"),
    limit: int = Query(20, ge=1, le=500, description="Max records to return"),
    db: sqlite3.Connection = Depends(get_db),
):
    if country:
        cur = db.execute(
            "SELECT * FROM customers WHERE country = ? LIMIT ?",
            (country, limit),
        )
    else:
        cur = db.execute("SELECT * FROM customers LIMIT ?", (limit,))
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/customers/top ─────────────────────────────────────────────────
@router.get("/top", summary="Top customers by total spend")
def top_customers(
    limit: int = Query(10, ge=1, le=100),
    db: sqlite3.Connection = Depends(get_db),
):
    cur = db.execute(
        """
        SELECT
            c.customer_id,
            c.name,
            c.email,
            c.country,
            COUNT(DISTINCT o.order_id)   AS total_orders,
            ROUND(SUM(o.total_usd), 2)   AS total_spend_usd
        FROM customers c
        JOIN orders o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id
        ORDER BY total_spend_usd DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/customers/{id} ────────────────────────────────────────────────
@router.get("/{customer_id}", summary="Get customer by ID")
def get_customer(customer_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    return {"status": "success", "data": dict(row)}


# ── GET /api/v1/customers/{id}/orders ─────────────────────────────────────────
@router.get("/{customer_id}/orders", summary="Orders for a customer")
def customer_orders(
    customer_id: int,
    limit: int = Query(20, ge=1, le=200),
    db: sqlite3.Connection = Depends(get_db),
):
    # Verify customer exists
    cur = db.execute(
        "SELECT customer_id FROM customers WHERE customer_id = ?", (customer_id,)
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")

    cur = db.execute(
        """
        SELECT
            o.order_id,
            o.order_time,
            o.payment_method,
            o.discount_pct,
            o.subtotal_usd,
            o.total_usd,
            o.country,
            o.device,
            o.source,
            COUNT(oi.order_item_id) AS item_count
        FROM orders o
        LEFT JOIN order_items oi ON oi.order_id = o.order_id
        WHERE o.customer_id = ?
        GROUP BY o.order_id
        ORDER BY o.order_time DESC
        LIMIT ?
        """,
        (customer_id, limit),
    )
    rows = _rows(cur)
    return {"status": "success", "customer_id": customer_id, "count": len(rows), "data": rows}
