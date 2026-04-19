"""
optimizer_billing.py
Credit reservation, deduction, and balance checks.
Connects to the Node.js backend via HTTP.
All billing operations fail safe — never run analysis if check fails.
"""

import os
import httpx
from typing import Optional

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3001")
AGENT_SECRET = os.environ.get("AGENT_SECRET", "")


def get_store_credits(store_id: str) -> Optional[int]:
    """Fetch current AI credit balance for a store."""
    try:
        response = httpx.get(
            f"{API_BASE_URL}/api/billing/credits/{store_id}",
            headers={"X-Agent-Secret": AGENT_SECRET},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("credits_remaining", 0)
        print(f"[Billing] Failed to fetch credits: {response.status_code}")
        return None
    except Exception as e:
        print(f"[Billing] Credits fetch error: {e}")
        return None


def reserve_credits(store_id: str, amount: int, run_id: str) -> bool:
    """Reserve credits before run starts. Returns True if successful."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/billing/credits/reserve",
            headers={"X-Agent-Secret": AGENT_SECRET},
            json={"storeId": store_id, "amount": amount, "runId": run_id},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"[Billing] Reserve credits error: {e}")
        return False


def finalize_credits(store_id: str, run_id: str, actual_amount: int) -> bool:
    """Finalize actual credit deduction after run completes."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/billing/credits/finalize",
            headers={"X-Agent-Secret": AGENT_SECRET},
            json={"storeId": store_id, "runId": run_id, "actualAmount": actual_amount},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"[Billing] Finalize credits error: {e}")
        return False


def release_reserved_credits(store_id: str, run_id: str) -> bool:
    """Release reserved credits if run fails or is cancelled."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/billing/credits/release",
            headers={"X-Agent-Secret": AGENT_SECRET},
            json={"storeId": store_id, "runId": run_id},
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"[Billing] Release credits error: {e}")
        return False


def check_and_reserve(store_id: str, required_credits: int, run_id: str) -> dict:
    """
    Full billing gate:
    1. Check balance
    2. Compare to required
    3. Reserve if sufficient
    4. Return structured result for UI
    """
    current = get_store_credits(store_id)

    # If billing service is unreachable, block for safety
    if current is None:
        return {
            "allowed": False,
            "status": "billing_check_failed",
            "message": "Could not verify credit balance. Please try again.",
            "credits_required": required_credits,
            "credits_available": 0,
            "credits_missing": required_credits,
            "top_up_recommended": True,
            "top_up_cta_label": "Top Up Credits"
        }

    if current < required_credits:
        missing = required_credits - current
        return {
            "allowed": False,
            "status": "blocked_insufficient_credits",
            "message": (
                f"You need {required_credits} credits to run this analysis, "
                f"but you only have {current} credits. "
                f"Please top up {missing} or more credits to continue."
            ),
            "credits_required": required_credits,
            "credits_available": current,
            "credits_missing": missing,
            "top_up_recommended": True,
            "top_up_cta_label": "Top Up Credits",
            "top_up_url": "/billing?action=topup"
        }

    # Try to reserve
    reserved = reserve_credits(store_id, required_credits, run_id)
    if not reserved:
        return {
            "allowed": False,
            "status": "reservation_failed",
            "message": "Could not reserve credits. Please try again.",
            "credits_required": required_credits,
            "credits_available": current,
            "credits_missing": 0,
            "top_up_recommended": False
        }

    return {
        "allowed": True,
        "status": "credits_reserved",
        "credits_required": required_credits,
        "credits_available": current,
        "credits_remaining_after": current - required_credits,
        "credits_missing": 0,
        "top_up_recommended": False
    }
