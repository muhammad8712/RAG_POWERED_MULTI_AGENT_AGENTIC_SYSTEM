# mock_erp_api/routers/health.py
"""Health-check and API meta endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from mock_erp_api.database import get_db, DB_PATH

router = APIRouter(tags=["Health"])

_ENDPOINTS = [
    "GET /api/v1/customers",
    "GET /api/v1/customers/top",
    "GET /api/v1/customers/{id}",
    "GET /api/v1/customers/{id}/orders",
    "GET /api/v1/orders",
    "GET /api/v1/orders/recent",
    "GET /api/v1/orders/summary",
    "GET /api/v1/orders/{id}",
    "GET /api/v1/products",
    "GET /api/v1/products/top-selling",
    "GET /api/v1/products/categories",
    "GET /api/v1/products/{id}",
    "GET /api/v1/payments",
    "GET /api/v1/payments/summary",
    "GET /api/v1/payments/{id}",
    "GET /api/v1/sales-orders",
    "GET /api/v1/sales-orders/summary",
    "GET /api/v1/sales-orders/{ref}",
    "GET /api/v1/purchase-orders",
    "GET /api/v1/purchase-orders/summary",
    "GET /api/v1/purchase-orders/{ref}",
]


@router.get("/", summary="Welcome")
def root():
    return {
        "name": "Mock ERP REST API",
        "version": "1.0.0",
        "description": "Simulated Odoo-style ERP endpoints for thesis demonstration.",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", summary="Health check")
def health(db: sqlite3.Connection = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
        db_file = str(DB_PATH)
        db_size_mb = round(DB_PATH.stat().st_size / 1_048_576, 2)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "disconnected", "detail": str(e)},
        )
    return {
        "status": "ok",
        "db": db_status,
        "db_path": db_file,
        "db_size_mb": db_size_mb,
    }


@router.get("/api/v1/info", summary="API metadata")
def info():
    return {
        "name": "Mock ERP REST API",
        "version": "1.0.0",
        "base_url": "http://localhost:8000",
        "available_endpoints": _ENDPOINTS,
        "total_endpoints": len(_ENDPOINTS),
    }
