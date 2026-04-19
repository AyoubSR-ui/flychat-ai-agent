"""
optimizer_pricing.py
Token estimation, cost calculation, and margin protection.
FlyChat COD must always maintain >= 25% gross margin on optimizer runs.
"""

import math

# ── Platform credit pricing ───────────────────────────────
# 1 credit = how many USD FlyChat charges the store
# This is the platform's internal pricing unit
CREDIT_VALUE_USD = 0.001        # 1 credit = $0.001 charged to store
MARGIN_TARGET = 0.25            # 25% gross margin minimum
SAFETY_BUFFER = 0.10            # 10% safety buffer for token underestimation

# ── Token estimation constants ────────────────────────────
# Average tokens per conversation message (content only)
AVG_TOKENS_PER_MESSAGE = 45
# Prompt overhead per conversation (system instructions, JSON structure)
PROMPT_OVERHEAD_PER_CONV = 350
# Expected output tokens per conversation analysis
AVG_OUTPUT_TOKENS_PER_CONV = 400
# Tokens for improvement generation (one call for whole batch)
IMPROVEMENT_GEN_INPUT_TOKENS = 800
IMPROVEMENT_GEN_OUTPUT_TOKENS = 600


def estimate_tokens_per_conversation(message_count: int) -> dict:
    """Estimate input/output tokens for analyzing one conversation."""
    input_tokens = (message_count * AVG_TOKENS_PER_MESSAGE) + PROMPT_OVERHEAD_PER_CONV
    output_tokens = AVG_OUTPUT_TOKENS_PER_CONV
    return {"input": input_tokens, "output": output_tokens}


def estimate_run_tokens(conversations: list, model_config: dict) -> dict:
    """
    Estimate total token usage for an optimizer run.
    Includes per-conversation analysis + improvement generation call.
    """
    total_input = 0
    total_output = 0

    for conv in conversations:
        msg_count = conv.get("message_count", 8)
        est = estimate_tokens_per_conversation(msg_count)
        total_input += est["input"]
        total_output += est["output"]

    # Add improvement generation call
    total_input += IMPROVEMENT_GEN_INPUT_TOKENS
    total_output += IMPROVEMENT_GEN_OUTPUT_TOKENS

    return {
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "conversation_count": len(conversations),
        "model": model_config["model_id"]
    }


def calculate_provider_cost(input_tokens: int, output_tokens: int, model_config: dict) -> float:
    """Calculate raw provider API cost in USD."""
    input_cost = (input_tokens / 1000) * model_config["input_cost_per_1k"]
    output_cost = (output_tokens / 1000) * model_config["output_cost_per_1k"]
    return round(input_cost + output_cost, 6)


def calculate_credit_charge(provider_cost_usd: float) -> dict:
    """
    Calculate how many credits to charge the store.

    Margin formula:
    charged_usd >= provider_cost / (1 - margin_target)
    charged_usd >= provider_cost / 0.75  →  25% gross margin

    Add safety buffer for token underestimation.
    Round UP to nearest integer credit (never round down).
    """
    # Apply safety buffer to provider cost
    buffered_cost = provider_cost_usd * (1 + SAFETY_BUFFER)

    # Minimum charge to hit margin target
    minimum_charge_usd = buffered_cost / (1 - MARGIN_TARGET)

    # Convert to credits (round UP — never underprice)
    credits_required = math.ceil(minimum_charge_usd / CREDIT_VALUE_USD)

    # Actual margin after rounding
    actual_charged_usd = credits_required * CREDIT_VALUE_USD
    actual_margin = (actual_charged_usd - provider_cost_usd) / actual_charged_usd if actual_charged_usd > 0 else 0

    return {
        "estimated_provider_cost_usd": round(provider_cost_usd, 6),
        "buffered_provider_cost_usd": round(buffered_cost, 6),
        "minimum_charge_usd": round(minimum_charge_usd, 6),
        "credits_required": credits_required,
        "actual_charged_usd": round(actual_charged_usd, 6),
        "actual_margin_pct": round(actual_margin * 100, 1),
        "margin_target_pct": MARGIN_TARGET * 100,
        "safety_buffer_pct": SAFETY_BUFFER * 100,
        "margin_protected": actual_margin >= MARGIN_TARGET
    }


def build_cost_estimate(conversations: list, model_config: dict) -> dict:
    """Full cost estimation for a run. Called before any API call."""
    token_est = estimate_run_tokens(conversations, model_config)

    provider_cost = calculate_provider_cost(
        token_est["estimated_input_tokens"],
        token_est["estimated_output_tokens"],
        model_config
    )

    credit_charge = calculate_credit_charge(provider_cost)

    return {
        **token_est,
        **credit_charge,
        "model_label": model_config.get("label", "Standard"),
        "ready": True
    }
