# mock_erp_api/routers/products.py
"""Product resource endpoints."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from mock_erp_api.database import get_db

router = APIRouter(prefix="/api/v1/products", tags=["Products"])


def _rows(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ── GET /api/v1/products ──────────────────────────────────────────────────────
@router.get("", summary="List products")
def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    if category:
        cur = db.execute(
            "SELECT * FROM products WHERE category = ? ORDER BY name LIMIT ?",
            (category, limit),
        )
    else:
        cur = db.execute("SELECT * FROM products ORDER BY name LIMIT ?", (limit,))
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/products/top-selling ─────────────────────────────────────────
@router.get("/top-selling", summary="Top products by quantity sold")
def top_selling(
    limit: int = Query(10, ge=1, le=100),
    db: sqlite3.Connection = Depends(get_db),
):
    cur = db.execute(
        """
        SELECT
            p.product_id,
            p.name,
            p.category,
            p.price_usd,
            p.cost_usd,
            p.margin_usd,
            SUM(oi.quantity)               AS total_qty_sold,
            ROUND(SUM(oi.line_total_usd), 2) AS total_revenue_usd
        FROM products p
        JOIN order_items oi ON oi.product_id = p.product_id
        GROUP BY p.product_id
        ORDER BY total_qty_sold DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/products/categories ──────────────────────────────────────────
@router.get("/categories", summary="List distinct product categories")
def list_categories(db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        "SELECT DISTINCT category, COUNT(*) AS product_count "
        "FROM products GROUP BY category ORDER BY category"
    )
    rows = _rows(cur)
    return {"status": "success", "count": len(rows), "data": rows}


# ── GET /api/v1/products/{id} ─────────────────────────────────────────────────
@router.get("/{product_id}", summary="Get product by ID")
def get_product(product_id: int, db: sqlite3.Connection = Depends(get_db)):
    cur = db.execute(
        "SELECT * FROM products WHERE product_id = ?", (product_id,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
    return {"status": "success", "data": dict(row)}
