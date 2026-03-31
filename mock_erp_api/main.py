# mock_erp_api/main.py
"""
FastAPI application factory for the Mock ERP REST API.

Run with:
    python mock_erp_api/run_server.py
or directly:
    uvicorn mock_erp_api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mock_erp_api.routers import (
    customers,
    health,
    orders,
    payments,
    products,
    purchase_orders,
    sales_orders,
)

# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mock ERP REST API",
    description=(
        "A simulated Odoo-style ERP REST API backed by a real SQLite database. "
        "Exposes customers, orders, products, payments, and Odoo-style sales/purchase orders "
        "for multi-agent RAG system demonstration."
    ),
    version="1.0.0",
    contact={
        "name": "Thesis Project",
    },
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS — allow all origins so the Streamlit app can call the API freely ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(customers.router)
app.include_router(orders.router)
app.include_router(products.router)
app.include_router(payments.router)
app.include_router(sales_orders.router)
app.include_router(purchase_orders.router)
