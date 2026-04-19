"""
communication_optimizer.py
Main orchestrator for the FlyChat COD communication optimization pipeline.

Files:
- optimizer_provider.py  → provider/model config
- optimizer_pricing.py   → cost estimation and margin protection
- optimizer_billing.py   → credit checks and deduction
- optimizer_batches.py   → batch processing

Safety rules enforced:
1. Never runs without passing billing gate
2. Never auto-approves outputs
3. Never modifies product/price/shipping facts
4. Never crashes agent.py (fail safe empty string)
5. Always targets >= 25% gross margin
"""

import json
import os
import uuid
import re
from datetime import datetime
from typing import Optional

from optimizer_provider import (
    get_optimizer_provider, get_model_config, get_optimizer_client,
    get_plan_limits, call_model
)
from optimizer_pricing import build_cost_estimate
from optimizer_billing import check_and_reserve, finalize_credits, release_reserved_credits
from optimizer_batches import chunk_conversations, process_batch, summarize_run_cost

APPROVED_IMPROVEMENTS_FILE = "approved_improvements.json"
OPTIMIZER_RUNS_FILE = "optimizer_runs.json"


# ============================================================
# RUN RECORD MANAGEMENT
# ============================================================

def create_run_record(store_id: str, model_config: dict, conversations: list, cost_estimate: dict) -> dict:
    return {
        "run_id": f"run_{uuid.uuid4().hex[:12]}",
        "store_id": store_id,
        "selected_model": model_config["model_id"],
        "provider": str(model_config.get("provider", "openai")),
        "conversations_requested": len(conversations),
        "conversations_processed": 0,
        "batch_count": 0,
        "status": "pending",
        "estimated_input_tokens": cost_estimate.get("estimated_input_tokens", 0),
        "estimated_output_tokens": cost_estimate.get("estimated_output_tokens", 0),
        "actual_input_tokens": 0,
        "actual_output_tokens": 0,
        "estimated_provider_cost_usd": cost_estimate.get("estimated_provider_cost_usd", 0),
        "actual_provider_cost_usd": 0,
        "estimated_credit_charge": cost_estimate.get("credits_required", 0),
        "actual_credit_charge": 0,
        "credits_reserved": cost_estimate.get("credits_required", 0),
        "credits_finalized": 0,
        "margin_target": 25.0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }


def save_run_record(record: dict):
    runs = {}
    if os.path.exists(OPTIMIZER_RUNS_FILE):
        try:
            with open(OPTIMIZER_RUNS_FILE, 'r') as f:
                runs = json.load(f)
        except Exception:
            pass
    runs[record["run_id"]] = record
    with open(OPTIMIZER_RUNS_FILE, 'w', encoding='utf-8') as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)


# ============================================================
# ESTIMATION ENDPOINT (called before run)
# ============================================================

def estimate_optimizer_run(conversations: list, store_id: str, plan: str = "starter") -> dict:
    """
    Estimates cost and checks if store has enough credits.
    Called when user opens the optimizer UI — before any API calls.
    Returns frontend-ready payload.
    """
    plan_limits = get_plan_limits(plan)
    mode = plan_limits["mode"]
    max_convs = plan_limits["max_conversations"]

    # Limit to plan max
    limited = conversations[:max_convs]
    model_config = get_model_config(mode)

    cost_estimate = build_cost_estimate(limited, model_config)

    return {
        "conversations_available": len(conversations),
        "conversations_to_analyze": len(limited),
        "plan_max_conversations": max_convs,
        "model": model_config["model_id"],
        "model_label": model_config["label"],
        "estimated_input_tokens": cost_estimate["estimated_input_tokens"],
        "estimated_output_tokens": cost_estimate["estimated_output_tokens"],
        "estimated_provider_cost_usd": cost_estimate["estimated_provider_cost_usd"],
        "credits_required": cost_estimate["credits_required"],
        "margin_target_pct": cost_estimate["margin_target_pct"],
        "store_id": store_id,
        "ready_to_run": True
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_optimization_pipeline(
    conversations: list,
    store_id: str,
    plan: str = "starter",
    auto_approve: bool = False
) -> dict:
    """
    Main pipeline with billing gate.

    Flow:
    1. Estimate cost
    2. Check + reserve credits (BLOCKS if insufficient)
    3. Process in batches
    4. Generate improvements
    5. Finalize credits
    6. Save as pending_approval
    """
    print(f"[Optimizer] Starting pipeline — store={store_id} conversations={len(conversations)}")

    # ── Step 1: Plan limits ──────────────────────────────
    plan_limits = get_plan_limits(plan)
    mode = plan_limits["mode"]
    batch_size = plan_limits["batch_size"]
    max_convs = plan_limits["max_conversations"]

    limited_convs = conversations[:max_convs]
    model_config = get_model_config(mode)
    provider = get_optimizer_provider()

    # ── Step 2: Cost estimate ─────────────────────────────
    cost_estimate = build_cost_estimate(limited_convs, model_config)
    required_credits = cost_estimate["credits_required"]

    # ── Step 3: Create run record ─────────────────────────
    run = create_run_record(store_id, model_config, limited_convs, cost_estimate)
    run["status"] = "estimating"
    save_run_record(run)
    run_id = run["run_id"]

    # ── Step 4: Billing gate (BLOCKS if insufficient) ─────
    billing_check = check_and_reserve(store_id, required_credits, run_id)

    if not billing_check["allowed"]:
        run["status"] = billing_check["status"]
        run["completed_at"] = datetime.now().isoformat()
        save_run_record(run)

        print(f"[Optimizer] BLOCKED — {billing_check['status']}")
        return {
            "status": billing_check["status"],
            "blocked": True,
            "run_id": run_id,
            "billing": billing_check,
            "cost_estimate": cost_estimate
        }

    print(f"[Optimizer] Credits reserved: {required_credits} credits for run {run_id}")
    run["status"] = "running"
    save_run_record(run)

    # ── Step 5: Process in batches ────────────────────────
    batches = chunk_conversations(limited_convs, batch_size)
    all_results = []
    batch_records = []

    try:
        client = get_optimizer_client(provider)

        for i, batch in enumerate(batches):
            batch_result = process_batch(batch, i, model_config, client, provider)
            batch_records.append(batch_result)
            all_results.extend(batch_result["results"])

            # Save progress after each batch
            run["conversations_processed"] = len(all_results)
            run["batch_count"] = i + 1
            run["status"] = "running"
            save_run_record(run)

            print(f"[Optimizer] Batch {i+1}/{len(batches)} done — {batch_result['conversations_analyzed']} analyzed")

    except Exception as e:
        print(f"[Optimizer] Pipeline failed: {e}")
        run["status"] = "failed"
        run["completed_at"] = datetime.now().isoformat()
        save_run_record(run)
        release_reserved_credits(store_id, run_id)
        return {"status": "failed", "error": str(e), "run_id": run_id}

    if not all_results:
        run["status"] = "failed"
        save_run_record(run)
        release_reserved_credits(store_id, run_id)
        return {"status": "failed", "error": "No conversations could be analyzed", "run_id": run_id}

    # ── Step 6: Generate improvements ────────────────────
    improvements = generate_improvements_from_results(all_results, model_config, client, provider)

    # ── Step 7: Calculate actual cost ────────────────────
    cost_summary = summarize_run_cost(batch_records, model_config)
    actual_credits = cost_summary["actual_credit_charge"]

    # ── Step 8: Finalize credits ──────────────────────────
    finalize_credits(store_id, run_id, actual_credits)

    # ── Step 9: Update run record ─────────────────────────
    run.update({
        "status": "completed",
        "conversations_processed": cost_summary["total_conversations_analyzed"],
        "actual_input_tokens": cost_summary["total_input_tokens"],
        "actual_output_tokens": cost_summary["total_output_tokens"],
        "actual_provider_cost_usd": cost_summary["actual_provider_cost_usd"],
        "actual_credit_charge": actual_credits,
        "credits_finalized": actual_credits,
        "completed_at": datetime.now().isoformat()
    })
    save_run_record(run)

    # ── Step 10: Save improvements as pending ─────────────
    improvements["status"] = "approved" if auto_approve else "pending_approval"
    improvements["run_id"] = run_id
    improvements["store_id"] = store_id
    improvements["generated_at"] = datetime.now().isoformat()

    _save_pending_improvements(improvements, store_id)

    print(f"[Optimizer] Pipeline complete — {cost_summary['total_conversations_analyzed']} analyzed")
    print(f"[Optimizer] Actual credits used: {actual_credits}")
    print(f"[Optimizer] Margin protected: {cost_summary['margin_protected']}")
    print(f"[Optimizer] Status: {improvements['status']}")

    return {
        "status": improvements["status"],
        "run_id": run_id,
        "conversations_analyzed": cost_summary["total_conversations_analyzed"],
        "cost_summary": cost_summary,
        "improvements_summary": improvements.get("improvement_summary", ""),
        "confidence_score": improvements.get("confidence_score", 0),
        "blocked": False
    }


def generate_improvements_from_results(results: list, model_config: dict, client, provider) -> dict:
    """
    Generates structured behavior improvements from all analyzed conversations.
    Uses ONE final API call on aggregated patterns.
    """
    top = sorted(results, key=lambda x: x.get("quality_score", {}).get("overall", 0), reverse=True)[:10]
    weak = sorted(results, key=lambda x: x.get("quality_score", {}).get("overall", 0))[:10]

    best_replies = [c.get("best_reply_example", "") for c in top if c.get("best_reply_example")][:5]
    worst_replies = [c.get("worst_reply_example", "") for c in weak if c.get("worst_reply_example")][:5]
    suggestions = list(set(c.get("improvement_suggestion", "") for c in results if c.get("improvement_suggestion")))[:5]

    all_strengths = [s for c in top for s in c.get("strengths", [])]
    all_weaknesses = [w for c in weak for w in c.get("weaknesses", [])]
    avg_score = sum(c.get("quality_score", {}).get("overall", 0) for c in results) / max(len(results), 1)

    prompt = f"""Sales communication improvement specialist for COD e-commerce Algeria.

ANALYSIS SUMMARY:
- Conversations analyzed: {len(results)}
- Average quality score: {avg_score:.1f}/10
- Top strengths: {json.dumps(all_strengths[:8])}
- Top weaknesses: {json.dumps(all_weaknesses[:8])}
- Best reply examples: {json.dumps(best_replies)}
- Worst reply examples: {json.dumps(worst_replies)}
- Improvement suggestions: {json.dumps(suggestions)}

Generate improvements. Respond ONLY with valid JSON:
{{
  "stable_behavior_rules": ["<rule 1>", "<rule 2>", "<rule 3>", "<rule 4>", "<rule 5>"],
  "communication_style_guidance": ["<style tip 1>", "<style tip 2>", "<style tip 3>"],
  "qualification_improvements": ["<improvement 1>", "<improvement 2>"],
  "closing_improvements": ["<improvement 1>", "<improvement 2>"],
  "patterns_to_avoid": ["<pattern 1>", "<pattern 2>", "<pattern 3>"],
  "quick_win_suggestions": ["<quick win 1>", "<quick win 2>"],
  "prompt_addon": "<max 5 lines generic communication guidance for the AI agent. No brand names, no prices.>",
  "confidence_score": <0.0-1.0>,
  "improvement_summary": "<2-3 sentences summarizing main findings>"
}}

CONSTRAINTS:
- Only improve communication behavior (tone, speed, naturalness)
- Never suggest changing prices, shipping, or product facts
- Keep all rules generic — any Algerian COD store
- No brand names, no specific prices"""

    try:
        text, _, _ = call_model(client, provider, model_config["model_id"], prompt, 1500)
        clean = re.sub(r'```json\n?|```\n?', '', text.strip())
        return json.loads(clean)
    except Exception as e:
        print(f"[Optimizer] Improvement generation failed: {e}")
        return {"status": "generation_failed", "improvement_summary": "Generation failed"}


def _save_pending_improvements(improvements: dict, store_id: str):
    existing = {}
    if os.path.exists(APPROVED_IMPROVEMENTS_FILE):
        try:
            with open(APPROVED_IMPROVEMENTS_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            pass
    key = store_id or 'global'
    existing[key] = improvements
    with open(APPROVED_IMPROVEMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def approve_improvements(store_id: str = None) -> bool:
    key = store_id or 'global'
    try:
        with open(APPROVED_IMPROVEMENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if key not in data:
            return False
        data[key]["status"] = "approved"
        data[key]["approved_at"] = datetime.now().isoformat()
        with open(APPROVED_IMPROVEMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Optimizer] Approve failed: {e}")
        return False


def get_behavior_improvement_layer(store_id: str = None, context: str = None) -> str:
    """
    Called by agent.py on every request.
    Returns approved improvement layer or empty string.
    NEVER crashes. NEVER returns unapproved content.
    """
    try:
        if not os.path.exists(APPROVED_IMPROVEMENTS_FILE):
            return ""
        with open(APPROVED_IMPROVEMENTS_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        improvements = all_data.get(store_id) or all_data.get('global')
        if not improvements or improvements.get("status") != "approved":
            return ""

        rules = improvements.get("stable_behavior_rules", [])
        style = improvements.get("communication_style_guidance", [])
        avoid = improvements.get("patterns_to_avoid", [])
        addon = improvements.get("prompt_addon", "")

        if not rules and not addon:
            return ""

        layer = "\n=== COMMUNICATION IMPROVEMENT LAYER ===\n"
        if rules:
            layer += "BEHAVIOR RULES:\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules[:5]))
        if style:
            layer += "\n\nSTYLE:\n" + "\n".join(f"- {s}" for s in style[:3])
        if avoid:
            layer += "\n\nAVOID:\n" + "\n".join(f"✗ {p}" for p in avoid[:3])
        if addon:
            layer += f"\n\n{addon}"
        layer += "\n=== END IMPROVEMENT LAYER ===\n"
        return layer
    except Exception:
        return ""  # Always fail safe


def get_optimizer_status(store_id: str = None) -> dict:
    status = {
        "has_pending": False, "has_approved": False,
        "last_run_id": None, "last_run_at": None,
        "conversations_analyzed": 0, "improvement_summary": None,
        "confidence_score": None
    }
    try:
        if os.path.exists(APPROVED_IMPROVEMENTS_FILE):
            with open(APPROVED_IMPROVEMENTS_FILE, 'r') as f:
                data = json.load(f)
            key = store_id or 'global'
            item = data.get(key, {})
            s = item.get("status")
            status["has_pending"] = s == "pending_approval"
            status["has_approved"] = s == "approved"
            status["last_run_at"] = item.get("generated_at")
            status["improvement_summary"] = item.get("improvement_summary")
            status["confidence_score"] = item.get("confidence_score")
            status["last_run_id"] = item.get("run_id")
    except Exception:
        pass
    return status
