# mock_erp_api/routers/purchase_orders.py
"""Odoo-style Purchase Order endpoints (purchase_orders_odoo table)."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/purchase-orders", tags=["Purchase Orders (Odoo)"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/purchase-orders ──────────────────────────────────────────────
@router.get("", summary="List Odoo purchase orders")
def list_purchase_orders(
    status: Optional[str] = Query(None, description="Filter by status (e.g. draft, purchase, done, cancel)"),
    priority: Optional[str] = Query(None, description="Filter by priority (0=Normal, 1=Urgent)"),
    vendor_name: Optional[str] = Query(None, description="Partial match on vendor name"),
    buyer: Optional[str] = Query(None, description="Partial match on buyer name"),
    limit: int = Query(20, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    filters, params = [], []
    if status:
        filters.append("status = ?")
        params.append(status)
    if priority:
        filters.append("priority = ?")
        params.append(priority)
    if vendor_name:
        filters.append("LOWER(vendor_name) LIKE ?")
        params.append(f"%{vendor_name.lower()}%")
    if buyer:
        filters.append("LOWER(buyer) LIKE ?")
        params.append(f"%{buyer.lower()}%")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(limit)

    cur = db.execute(
        f"""
        SELECT *
        FROM purchase_orders_odoo
        {where}
        ORDER BY order_deadline DESC
        LIMIT ?
        """,
        params,
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/purchase-orders/summary ──────────────────────────────────────
@router.get("/summary", summary="Purchase order statistics")
def purchase_orders_summary(db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        """
        SELECT
            status,
            COUNT(*)                    AS count,
            ROUND(SUM(total), 2)        AS total_value,
            ROUND(AVG(total), 2)        AS avg_value
        FROM purchase_orders_odoo
        GROUP BY status
        ORDER BY count DESC
        """
    )
    rows = _rows(cur)

    cur2 = db.execute(
        "SELECT COUNT(*) AS total_orders, ROUND(SUM(total), 2) AS grand_total FROM purchase_orders_odoo"
    )
    overall = dict(cur2.fetchone())

    # High-value POs (>= 10,000)
    cur3 = db.execute(
        "SELECT COUNT(*) AS count FROM purchase_orders_odoo WHERE total >= 10000"
    )
    high_value = dict(cur3.fetchone())

    return {
        "status": "success",
        "data": {
            "overall": overall,
            "high_value_pos": high_value,
            "by_status": rows,
        },
    }


# ── GET /api/v1/purchase-orders/{ref} ────────────────────────────────────────
@router.get("/{purchase_order_ref}", summary="Get purchase order by reference")
def get_purchase_order(purchase_order_ref: str, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        "SELECT * FROM purchase_orders_odoo WHERE purchase_order_ref = ?",
        (purchase_order_ref,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Purchase order '{purchase_order_ref}' not found.",
        )
    return {"status": "success", "data": dict(row)}
