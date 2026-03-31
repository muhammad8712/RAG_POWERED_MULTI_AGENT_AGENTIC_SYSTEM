# mock_erp_api/routers/orders.py
"""Order resource endpoints."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/orders", tags=["Orders"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/orders ────────────────────────────────────────────────────────
@router.get("", summary="List orders")
def list_orders(
    country: Optional[str] = Query(None),
    payment_method: Optional[str] = Query(None),
    device: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    filters, params = [], []
    if country:
        filters.append("o.country = ?")
        params.append(country)
    if payment_method:
        filters.append("o.payment_method = ?")
        params.append(payment_method)
    if device:
        filters.append("o.device = ?")
        params.append(device)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(limit)

    cur = db.execute(
        f"""
        SELECT
            o.*,
            c.name AS customer_name
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        {where}
        ORDER BY o.order_time DESC
        LIMIT ?
        """,
        params,
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/orders/recent ─────────────────────────────────────────────────
@router.get("/recent", summary="Most recent orders")
def recent_orders(
    limit: int = Query(10, ge=1, le=100),
    db: sqlite3.Connection = Depends(get_db),
):
    cur = db.execute(
        """
        SELECT
            o.order_id,
            o.order_time,
            c.name  AS customer_name,
            o.country,
            o.payment_method,
            o.total_usd
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        ORDER BY o.order_time DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/orders/summary ────────────────────────────────────────────────
@router.get("/summary", summary="Aggregate order statistics")
def orders_summary(db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        """
        SELECT
            COUNT(*)                        AS total_orders,
            ROUND(SUM(total_usd), 2)        AS total_revenue_usd,
            ROUND(AVG(total_usd), 2)        AS avg_order_value_usd,
            ROUND(MIN(total_usd), 2)        AS min_order_value_usd,
            ROUND(MAX(total_usd), 2)        AS max_order_value_usd,
            COUNT(DISTINCT customer_id)     AS unique_customers,
            COUNT(DISTINCT country)         AS countries_served
        FROM orders
        """
    )
    row = cur.fetchone()
    return {"status": "success", "data": dict(row)}


# ── GET /api/v1/orders/{id} ───────────────────────────────────────────────────
@router.get("/{order_id}", summary="Get order by ID with line items")
def get_order(order_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        """
        SELECT o.*, c.name AS customer_name, c.email AS customer_email
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        WHERE o.order_id = ?
        """,
        (order_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")

    order = dict(row)

    # Fetch line items
    cur2 = db.execute(
        """
        SELECT
            oi.order_item_id,
            oi.product_id,
            p.name      AS product_name,
            p.category,
            oi.quantity,
            oi.unit_price_usd,
            oi.line_total_usd
        FROM order_items oi
        JOIN products p ON p.product_id = oi.product_id
        WHERE oi.order_id = ?
        ORDER BY oi.order_item_id
        """,
        (order_id,),
    )
    cols = [d[0] for d in cur2.description]
    order["line_items"] = [dict(zip(cols, r)) for r in cur2.fetchall()]

    return {"status": "success", "data": order}
