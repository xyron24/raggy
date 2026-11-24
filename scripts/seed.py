"""Seed the running API with demo FAQ documents.

Usage:
    docker compose up -d              # or: uvicorn app.main:app
    python scripts/seed.py            # talks to http://localhost:8000
    RAGGY_URL=http://host:8000 python scripts/seed.py

Requires OPENAI_API_KEY in the API's environment (embeddings are billed).
"""

import os
import sys

import httpx

BASE_URL = os.environ.get("RAGGY_URL", "http://localhost:8000")

DOCS = [
    {
        "doc_id": "faq-refunds",
        "title": "Refund Policy",
        "text": (
            "Acme Cloud offers a 30-day money-back guarantee on all monthly plans, "
            "no questions asked. Annual plans are refunded on a prorated basis for "
            "unused full months. To request a refund, email billing@acmecloud.example "
            "from your account address; refunds land within 5 business days."
        ),
    },
    {
        "doc_id": "faq-security",
        "title": "Security Overview",
        "text": (
            "All customer data is encrypted at rest with AES-256 and in transit with "
            "TLS 1.3. Acme Cloud is SOC 2 Type II certified and undergoes annual "
            "penetration tests by an independent auditor. Data residency options "
            "cover the US, EU, and APAC regions, selectable per workspace."
        ),
    },
    {
        "doc_id": "guide-sso",
        "title": "Setting Up SSO",
        "text": (
            "Single sign-on is available on the Team plan and above. Acme Cloud "
            "supports SAML 2.0 and OpenID Connect identity providers, including "
            "Okta, Entra ID, and Google Workspace. Setup takes about 10 minutes: "
            "paste your IdP metadata URL into Admin Settings, then Security, then "
            "Single Sign-On, and assign the Acme Cloud app to your users."
        ),
    },
    {
        "doc_id": "faq-limits",
        "title": "API Rate Limits",
        "text": (
            "The Starter plan allows 100 API requests per minute per key, Team "
            "allows 1,000, and Enterprise is custom. Exceeding the limit returns "
            "HTTP 429 with a Retry-After header in seconds. Batch endpoints accept "
            "up to 50 items per call and count as one request against the quota."
        ),
    },
    {
        "doc_id": "faq-support",
        "title": "Support Tiers",
        "text": (
            "Starter customers get community forum support with a 48-hour median "
            "response. Team adds email support with a 4-hour response SLA on "
            "business days. Enterprise adds a named engineer, a 30-minute critical "
            "response SLA, and 24/7 phone support with quarterly reviews."
        ),
    },
]

SAMPLE_QUERY = "How do refunds work for annual plans?"


def main() -> int:
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError):
        print(f"API not reachable at {BASE_URL}. Start it first:")
        print("  docker compose up -d   # or: uvicorn app.main:app")
        return 1

    for doc in DOCS:
        response = httpx.post(f"{BASE_URL}/documents/ingest", json=doc, timeout=60)
        if response.status_code == 503:
            print("API has no OpenAI key. Set OPENAI_API_KEY in .env, restart it:")
            print("  docker compose up -d --force-recreate api")
            return 1
        response.raise_for_status()
        body = response.json()
        print(f"ingested {body['doc_id']}: {body['num_chunks']} chunk(s)")

    stats = httpx.get(f"{BASE_URL}/documents/stats", timeout=10).json()
    print(f"collection '{stats['collection']}' now holds {stats['count']} point(s)")
    print("Try a query:")
    print(f"  curl -X POST {BASE_URL}/query -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"query\": \"{SAMPLE_QUERY}\"}}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
