"""
optimizer_batches.py
Batch processing logic for optimizer runs.
Processes conversations in chunks, saves progress per batch.
"""

import json
import re
import time
from datetime import datetime
from typing import Optional

from optimizer_provider import get_optimizer_client, get_optimizer_provider, get_model_config, call_model


def chunk_conversations(conversations: list, batch_size: int) -> list[list]:
    """Split conversation list into batches of batch_size."""
    return [
        conversations[i:i + batch_size]
        for i in range(0, len(conversations), batch_size)
    ]


def format_conversation_for_analysis(messages: list) -> str:
    lines = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        content = msg.get("content", "")
        if not content:
            continue
        if content.startswith("📷") or content.startswith("🎤"):
            continue
        if "replied to an ad" in content:
            continue
        if "Auto-label" in content:
            continue
        role = "AGENT" if sender in ["bot", "assistant", "agent"] else "CUSTOMER"
        lines.append(f"{role}: {content}")
    return "\n".join(lines[:40])  # Cap at 40 lines per conversation


def analyze_single_conversation(
    conv: dict,
    client,
    provider,
    model_id: str,
    max_tokens: int = 1200
) -> tuple[Optional[dict], int, int]:
    """
    Analyze one conversation. Returns (result, input_tokens, output_tokens).
    Returns (None, 0, 0) on failure.
    """
    messages = conv.get("messages", [])
    if len(messages) < 4:
        return None, 0, 0

    conv_text = format_conversation_for_analysis(messages)
    outcome = conv.get("outcome", "unknown")
    lead_stage = conv.get("lead_stage", "interested")
    order_confirmed = conv.get("order_confirmed", False)
    msg_count = conv.get("message_count", len(messages))

    prompt = f"""Analyze this COD e-commerce sales conversation from Algeria.
The agent handles customers in Algerian darija (Arabic/Latin/French mix).

CONVERSATION:
{conv_text}

OUTCOME: lead_stage={lead_stage} | order_confirmed={order_confirmed} | messages={msg_count}

Respond ONLY with valid JSON:
{{
  "quality_score": {{
    "clarity": <1-10>,
    "naturalness": <1-10>,
    "progression": <1-10>,
    "qualification_speed": <1-10>,
    "closing_effectiveness": <1-10>,
    "darija_handling": <1-10>,
    "overall": <1-10>
  }},
  "strengths": ["<specific strength>"],
  "weaknesses": ["<specific weakness>"],
  "language_detected": "<arabic|latin|french|mixed>",
  "messages_to_qualify": <int or 0>,
  "patterns_found": ["<pattern_name>"],
  "best_reply_example": "<best agent reply from conversation>",
  "worst_reply_example": "<worst agent reply from conversation>",
  "improvement_suggestion": "<one specific actionable suggestion>"
}}"""

    try:
        text, input_tokens, output_tokens = call_model(
            client, provider, model_id, prompt, max_tokens
        )
        clean = re.sub(r'```json\n?|```\n?', '', text.strip())
        result = json.loads(clean)
        result["conversation_id"] = conv.get("conversation_id", "")
        result["store_id"] = conv.get("store_id", "")
        result["outcome"] = outcome
        result["analyzed_at"] = datetime.now().isoformat()
        return result, input_tokens, output_tokens
    except Exception as e:
        print(f"[Batch] Failed to analyze conv {conv.get('conversation_id')}: {e}")
        return None, 0, 0


def process_batch(
    batch: list,
    batch_index: int,
    model_config: dict,
    client,
    provider
) -> dict:
    """
    Process one batch of conversations.
    Returns batch result with per-conversation analysis and token totals.
    """
    print(f"[Batch] Processing batch {batch_index + 1} ({len(batch)} conversations)...")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    failed_count = 0

    for conv in batch:
        result, in_tok, out_tok = analyze_single_conversation(
            conv,
            client,
            provider,
            model_config["model_id"]
        )
        if result:
            results.append(result)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
        else:
            failed_count += 1
        time.sleep(0.2)  # Small delay to respect rate limits

    return {
        "batch_index": batch_index,
        "conversations_in": len(batch),
        "conversations_analyzed": len(results),
        "conversations_failed": failed_count,
        "results": results,
        "actual_input_tokens": total_input_tokens,
        "actual_output_tokens": total_output_tokens,
        "status": "completed" if failed_count == 0 else "partially_completed",
        "processed_at": datetime.now().isoformat()
    }


def summarize_run_cost(batch_results: list, model_config: dict) -> dict:
    """Summarize actual token usage and cost across all batches."""
    from optimizer_pricing import calculate_provider_cost, calculate_credit_charge

    total_input = sum(b.get("actual_input_tokens", 0) for b in batch_results)
    total_output = sum(b.get("actual_output_tokens", 0) for b in batch_results)
    total_analyzed = sum(b.get("conversations_analyzed", 0) for b in batch_results)
    total_failed = sum(b.get("conversations_failed", 0) for b in batch_results)

    actual_cost = calculate_provider_cost(total_input, total_output, model_config)
    credit_info = calculate_credit_charge(actual_cost)

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_conversations_analyzed": total_analyzed,
        "total_conversations_failed": total_failed,
        "actual_provider_cost_usd": actual_cost,
        "actual_credit_charge": credit_info["credits_required"],
        "margin_protected": credit_info["margin_protected"],
        "actual_margin_pct": credit_info["actual_margin_pct"]
    }
