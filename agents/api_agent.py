# agents/api_agent.py
"""
APIAgent — makes HTTP GET calls to the Mock ERP REST API.

The run() method accepts a natural-language query and uses keyword-based
routing (Option A: deterministic, demo-safe) to map it to the most relevant
ERP endpoint, then calls it via self.get().
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

import requests

# ── Base URL for the mock ERP server ─────────────────────────────────────────
ERP_BASE_URL = "http://localhost:8000"

# ── Keyword → (endpoint, default_params) routing table ───────────────────────
#
# Rules are checked top-to-bottom; first match wins.
# Each rule is: (list_of_keywords_any_must_match, endpoint_path, default_params)
#
_ROUTES: list[tuple[list[str], str, dict]] = [
    # ── Sales Orders (Odoo) ─────────────────────────────────────────────────
    (["sales order", "sales orders", "so ", " so,", "odoo sales"],
     "api/v1/sales-orders", {"limit": 10}),

    # ── Purchase Orders (Odoo) ──────────────────────────────────────────────
    # Summary/statistics sub-route MUST come before the generic list route
    (["purchase order statistic", "purchase order summary", "purchase order stat",
      "po statistic", "po summary", "po stat",
      "purchase order aggregate", "purchase orders aggregate",
      "purchase order overview"],
     "api/v1/purchase-orders/summary", {}),

    (["purchase order", "purchase orders", "po ", " po,", "odoo purchase"],
     "api/v1/purchase-orders", {"limit": 10}),


    # ── Orders — specific sub-routes first ──────────────────────────────────
    (["order summary", "order statistics", "revenue summary", "total orders",
      "average order"],
     "api/v1/orders/summary", {}),

    (["recent order", "latest order", "newest order", "last order"],
     "api/v1/orders/recent", {"limit": 10}),

    # ── Customers — sub-routes first ────────────────────────────────────────
    (["top customer", "best customer", "highest spending", "most spending",
      "highest spend", "biggest customer", "customer ranking"],
     "api/v1/customers/top", {"limit": 10}),

    # ── Products — sub-routes first ─────────────────────────────────────────
    (["top product", "best selling", "best-selling", "most sold", "top selling",
      "most popular product", "highest quantity"],
     "api/v1/products/top-selling", {"limit": 10}),

    (["product categor", "categories"],
     "api/v1/products/categories", {}),

    # ── Payments — sub-routes first ─────────────────────────────────────────
    (["payment summary", "payment statistics", "total payment",
      "average payment", "payment breakdown"],
     "api/v1/payments/summary", {}),

    # ── Generic list routes ──────────────────────────────────────────────────
    (["customer", "customers"],
     "api/v1/customers", {"limit": 10}),

    (["order", "orders"],
     "api/v1/orders", {"limit": 10}),

    (["product", "products"],
     "api/v1/products", {"limit": 10}),

    (["payment", "payments", "transaction", "transactions"],
     "api/v1/payments", {"limit": 10}),

    # ── Health / info (fallback) ─────────────────────────────────────────────
    (["health", "status", "info", "available endpoint", "api info"],
     "api/v1/info", {}),
]


def _match_route(query: str) -> tuple[str, dict]:
    """Return (endpoint, params) for the first matching route, or the info
    endpoint as a safe fallback."""
    q = query.lower()
    for keywords, endpoint, params in _ROUTES:
        if any(kw in q for kw in keywords):
            # Pull out an inline limit like "top 5" / "last 3" from the query
            m = re.search(r"\b(?:top|last|recent|limit)\s+(\d+)\b", q)
            if m:
                params = dict(params)             # copy so we don't mutate constant
                params["limit"] = int(m.group(1))
            return endpoint, params
    return "api/v1/info", {}


class APIAgent:
    def __init__(
        self,
        base_urls: list[str] | None = None,
        timeout: int = 15,
        erp_base_url: str = ERP_BASE_URL,
    ):
        self.base_urls = base_urls or []
        self.timeout = timeout
        self.erp_base_url = erp_base_url

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _is_allowed_url(self, full_url: str) -> bool:
        if not self.base_urls:
            return True
        return any(full_url.startswith(b) for b in self.base_urls)

    def get(
        self,
        base_url: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        base = base_url if base_url.endswith("/") else base_url + "/"
        full_url = urljoin(base, endpoint.lstrip("/"))

        if not self._is_allowed_url(full_url):
            return {
                "endpoint": full_url,
                "status_code": None,
                "params": params or {},
                "error": f"URL not allowed: {full_url}",
                "result": None,
            }

        try:
            resp = requests.get(
                full_url, params=params, headers=headers, timeout=self.timeout
            )
            content_type = (resp.headers.get("content-type") or "").lower()

            if "application/json" in content_type:
                result: Any = resp.json()
            else:
                result = resp.text

            return {
                "endpoint": full_url,
                "status_code": resp.status_code,
                "params": params or {},
                "result": result,
            }
        except Exception as e:
            return {
                "endpoint": full_url,
                "status_code": None,
                "params": params or {},
                "error": str(e),
                "result": None,
            }

    # ── Public run() — called by the orchestration graph ────────────────────

    def run(self, query: str) -> dict[str, Any]:
        """
        Map a natural-language query to a mock ERP endpoint and return
        the JSON response in a format compatible with the rest of the
        multi-agent pipeline (validate → explainability).
        """
        endpoint, params = _match_route(query)
        response = self.get(self.erp_base_url, endpoint, params=params)

        return {
            "query": query,
            "endpoint": response["endpoint"],
            "status_code": response.get("status_code"),
            "params": response.get("params", {}),
            "result": response.get("result"),
            "error": response.get("error"),
        }