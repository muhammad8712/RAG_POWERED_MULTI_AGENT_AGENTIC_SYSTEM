# mock_erp_api/routers/payments.py
"""Payment / transaction resource endpoints."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/payments", tags=["Payments"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/payments ──────────────────────────────────────────────────────
@router.get("", summary="List payment transactions")
def list_payments(
    type: Optional[str] = Query(None, description="Filter by payment type"),
    customer_id: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    filters, params = [], []
    if type:
        filters.append("p.type = ?")
        params.append(type)
    if customer_id:
        filters.append("p.customer_id = ?")
        params.append(customer_id)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(limit)

    cur = db.execute(
        f"""
        SELECT
            p.transaction_id,
            p.payment_date,
            p.customer_id,
            c.name  AS customer_name,
            p.amount,
            p.type,
            p.description
        FROM payments p
        JOIN customers c ON c.customer_id = p.customer_id
        {where}
        ORDER BY p.payment_date DESC
        LIMIT ?
        """,
        params,
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/payments/summary ──────────────────────────────────────────────
@router.get("/summary", summary="Payment aggregate statistics")
def payments_summary(db: sqlite3.Connection = Depends(get_db)):
    # Overall stats
    cur = db.execute(
        """
        SELECT
            COUNT(*)                    AS total_transactions,
            ROUND(SUM(amount), 2)       AS total_amount_usd,
            ROUND(AVG(amount), 2)       AS avg_amount_usd,
            ROUND(MIN(amount), 2)       AS min_amount_usd,
            ROUND(MAX(amount), 2)       AS max_amount_usd
        FROM payments
        """
    )
    overall = dict(cur.fetchone())

    # Breakdown by type
    cur2 = db.execute(
        """
        SELECT
            type,
            COUNT(*)                    AS count,
            ROUND(SUM(amount), 2)       AS total_usd
        FROM payments
        GROUP BY type
        ORDER BY total_usd DESC
        """
    )
    breakdown = _rows(cur2)

    return {
        "status": "success",
        "data": {
            "overall": overall,
            "by_type": breakdown,
        },
    }


# ── GET /api/v1/payments/{id} ─────────────────────────────────────────────────
@router.get("/{transaction_id}", summary="Get payment transaction by ID")
def get_payment(transaction_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        """
        SELECT p.*, c.name AS customer_name
        FROM payments p
        JOIN customers c ON c.customer_id = p.customer_id
        WHERE p.transaction_id = ?
        """,
        (transaction_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404, detail=f"Transaction {transaction_id} not found."
        )
    return {"status": "success", "data": dict(row)}
