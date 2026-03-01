# agents/api_agent.py

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests


class APIAgent:
    def __init__(self, base_urls: list[str] | None = None, timeout: int = 15):
        self.base_urls = base_urls or []
        self.timeout = timeout

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
            resp = requests.get(full_url, params=params, headers=headers, timeout=self.timeout)
            content_type = (resp.headers.get("content-type") or "").lower()

            result: Any
            if "application/json" in content_type:
                result = resp.json()
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

    def run(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "note": "API Agent is active but not yet mapped to real ERP endpoints.",
            "endpoint": None,
            "status_code": None,
            "result": None,
        }