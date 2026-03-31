# mock_erp_api/run_server.py
"""
Convenience launcher for the Mock ERP REST API.

Usage:
    python mock_erp_api/run_server.py
    python mock_erp_api/run_server.py --port 8001
    python mock_erp_api/run_server.py --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the Python path so imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Mock ERP REST API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Mock ERP REST API")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Swagger UI  ->  http://{args.host}:{args.port}/docs")
    print(f"  Health      ->  http://{args.host}:{args.port}/health")
    print(f"{'='*60}\n")

    uvicorn.run(
        "mock_erp_api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
