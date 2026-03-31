# mock_erp_api/routers/sales_orders.py
"""Odoo-style Sales Order endpoints (sales_orders_odoo table)."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/sales-orders", tags=["Sales Orders (Odoo)"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/sales-orders ─────────────────────────────────────────────────
@router.get("", summary="List Odoo sales orders")
def list_sales_orders(
    status: Optional[str] = Query(None, description="Filter by status (e.g. draft, sale, done, cancel)"),
    customer_name: Optional[str] = Query(None, description="Filter by customer name (case-insensitive partial match)"),
    salesperson: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    filters, params = [], []
    if status:
        filters.append("status = ?")
        params.append(status)
    if customer_name:
        filters.append("LOWER(customer_name) LIKE ?")
        params.append(f"%{customer_name.lower()}%")
    if salesperson:
        filters.append("LOWER(salesperson) LIKE ?")
        params.append(f"%{salesperson.lower()}%")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(limit)

    cur = db.execute(
        f"""
        SELECT *
        FROM sales_orders_odoo
        {where}
        ORDER BY creation_date DESC
        LIMIT ?
        """,
        params,
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/sales-orders/summary ─────────────────────────────────────────
@router.get("/summary", summary="Sales order statistics by status")
def sales_orders_summary(db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        """
        SELECT
            status,
            COUNT(*)                    AS count,
            ROUND(SUM(total), 2)        AS total_value,
            ROUND(AVG(total), 2)        AS avg_value
        FROM sales_orders_odoo
        GROUP BY status
        ORDER BY count DESC
        """
    )
    rows = _rows(cur)

    cur2 = db.execute(
        "SELECT COUNT(*) AS total_orders, ROUND(SUM(total), 2) AS grand_total FROM sales_orders_odoo"
    )
    overall = dict(cur2.fetchone())

    return {
        "status": "success",
        "data": {
            "overall": overall,
            "by_status": rows,
        },
    }


# ── GET /api/v1/sales-orders/{ref} ───────────────────────────────────────────
@router.get("/{sales_order_ref}", summary="Get sales order by reference")
def get_sales_order(sales_order_ref: str, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        "SELECT * FROM sales_orders_odoo WHERE sales_order_ref = ?",
        (sales_order_ref,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Sales order '{sales_order_ref}' not found.",
        )
    return {"status": "success", "data": dict(row)}
