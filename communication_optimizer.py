"""
FlyChat COD — Communication Optimizer
======================================
Analyzes filtered conversation data using AI to detect quality patterns.
Generates reusable communication guidance injected into agent.py.

SAFETY RULES:
- Never modifies product prices, stock, or shipping facts
- Never auto-rewrites the base agent prompt
- Only generates approved structured improvement layers
- All outputs must be explicitly approved before injection
- Never runs live — always a separate background process
"""

import json
import os
import re
from datetime import datetime
from typing import Optional
import anthropic

# ============================================================
# CONSTANTS
# ============================================================

OPTIMIZER_OUTPUT_FILE = "optimizer_outputs.json"
APPROVED_IMPROVEMENTS_FILE = "approved_improvements.json"

MIN_MESSAGES_TO_ANALYZE = 4
MAX_CONVERSATIONS_PER_RUN = 100
QUALITY_SCORE_THRESHOLD = 7.0  # out of 10

# ============================================================
# DATA MODELS
# ============================================================

def empty_quality_score():
    return {
        "clarity": 0,
        "naturalness": 0,
        "progression": 0,
        "qualification_speed": 0,
        "closing_effectiveness": 0,
        "darija_handling": 0,
        "overall": 0,
    }

def empty_analysis_result():
    return {
        "conversation_id": "",
        "store_id": "",
        "quality_score": empty_quality_score(),
        "strengths": [],
        "weaknesses": [],
        "outcome": "unknown",
        "messages_to_qualify": 0,
        "messages_to_close": 0,
        "language_detected": "unknown",
        "patterns_found": [],
        "analyzed_at": datetime.now().isoformat(),
    }

# ============================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================

def analyze_conversation_quality(conversation_data: dict) -> dict | None:
    """
    Uses Claude AI to score and analyze a single conversation.

    Input: {
        conversation_id, store_id, messages, outcome,
        lead_stage, order_confirmed, message_count,
        qualified_at, confirmed_at, channel
    }

    Output: analysis_result dict with scores and patterns
    """
    messages = conversation_data.get("messages", [])
    if len(messages) < MIN_MESSAGES_TO_ANALYZE:
        return None

    conv_text = format_conversation_for_analysis(messages)
    outcome = conversation_data.get("outcome", "unknown")
    lead_stage = conversation_data.get("lead_stage", "interested")
    order_confirmed = conversation_data.get("order_confirmed", False)
    msg_count = conversation_data.get("message_count", len(messages))

    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    analysis_prompt = f"""You are analyzing a COD e-commerce sales conversation from Algeria.
The sales agent handles customers in Algerian darija (Arabic/Latin/French mix).

CONVERSATION:
{conv_text}

OUTCOME DATA:
- Lead stage reached: {lead_stage}
- Order confirmed: {order_confirmed}
- Total messages: {msg_count}

Analyze this conversation and respond ONLY with a JSON object:
{{
  "quality_score": {{
    "clarity": <1-10: were replies clear and direct?>,
    "naturalness": <1-10: did replies feel human, not robotic?>,
    "progression": <1-10: did conversation move forward each turn?>,
    "qualification_speed": <1-10: how fast were key info collected?>,
    "closing_effectiveness": <1-10: how well did agent close or attempt to close?>,
    "darija_handling": <1-10: how well did agent handle darija language mix?>,
    "overall": <1-10: overall quality>
  }},
  "strengths": [<list of specific good things the agent did>],
  "weaknesses": [<list of specific bad things: repetition, missed signals, etc>],
  "language_detected": "<arabic|latin|french|mixed>",
  "messages_to_qualify": <number of messages until qualifying info was collected, or 0>,
  "patterns_found": [<list of detected patterns: "fast_qualifier", "robotic_reply", "good_closer", "missed_buying_signal", "good_objection_handling", etc>],
  "best_reply_example": "<copy the single best AI reply from this conversation>",
  "worst_reply_example": "<copy the single worst AI reply from this conversation>",
  "improvement_suggestion": "<one specific actionable suggestion for this type of conversation>"
}}

Be specific. Do not be generic. Base everything on the actual conversation text."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": analysis_prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json\n?", "", raw)
        raw = re.sub(r"```\n?", "", raw)

        analysis = json.loads(raw)

        result = empty_analysis_result()
        result.update({
            "conversation_id": conversation_data.get("conversation_id", ""),
            "store_id": conversation_data.get("store_id", ""),
            "quality_score": analysis.get("quality_score", empty_quality_score()),
            "strengths": analysis.get("strengths", []),
            "weaknesses": analysis.get("weaknesses", []),
            "outcome": outcome,
            "messages_to_qualify": analysis.get("messages_to_qualify", 0),
            "messages_to_close": conversation_data.get("messages_to_close", 0),
            "language_detected": analysis.get("language_detected", "unknown"),
            "patterns_found": analysis.get("patterns_found", []),
            "best_reply_example": analysis.get("best_reply_example", ""),
            "worst_reply_example": analysis.get("worst_reply_example", ""),
            "improvement_suggestion": analysis.get("improvement_suggestion", ""),
            "analyzed_at": datetime.now().isoformat(),
        })

        return result

    except Exception as e:
        print(f"[Optimizer] Analysis failed for {conversation_data.get('conversation_id')}: {e}")
        return None


def rank_best_examples(conversations: list) -> dict:
    """
    Ranks analyzed conversations by quality.
    Separates top performers from weak performers.
    Returns ranked lists for pattern extraction.
    """
    analyzed = [c for c in conversations if c and c.get("quality_score")]

    ranked = sorted(
        analyzed,
        key=lambda x: x["quality_score"].get("overall", 0),
        reverse=True,
    )

    top_performers = [c for c in ranked if c["quality_score"].get("overall", 0) >= QUALITY_SCORE_THRESHOLD]
    weak_performers = [c for c in ranked if c["quality_score"].get("overall", 0) < 5.0]

    best_qualifiers = sorted(
        analyzed,
        key=lambda x: x["quality_score"].get("qualification_speed", 0),
        reverse=True,
    )[:5]

    best_closers = sorted(
        [c for c in analyzed if c.get("outcome") in ("confirmed", "qualified_lead")],
        key=lambda x: x["quality_score"].get("closing_effectiveness", 0),
        reverse=True,
    )[:5]

    best_darija = sorted(
        analyzed,
        key=lambda x: x["quality_score"].get("darija_handling", 0),
        reverse=True,
    )[:5]

    return {
        "top_performers": top_performers[:10],
        "weak_performers": weak_performers[:10],
        "best_qualifiers": best_qualifiers,
        "best_closers": best_closers,
        "best_darija_handlers": best_darija,
        "total_analyzed": len(analyzed),
        "average_score": sum(c["quality_score"].get("overall", 0) for c in analyzed) / max(len(analyzed), 1),
        "ranked_at": datetime.now().isoformat(),
    }


def extract_top_patterns(ranked_data: dict) -> dict:
    """
    Extracts recurring patterns from top and weak performers.
    Groups patterns by type for behavior generation.
    """
    top = ranked_data.get("top_performers", [])
    weak = ranked_data.get("weak_performers", [])

    good_patterns: dict[str, int] = {}
    for conv in top:
        for pattern in conv.get("patterns_found", []):
            good_patterns[pattern] = good_patterns.get(pattern, 0) + 1

    common_weaknesses: dict[str, int] = {}
    for conv in weak:
        for weakness in conv.get("weaknesses", []):
            key = weakness.lower()[:80]
            common_weaknesses[key] = common_weaknesses.get(key, 0) + 1

    best_replies = [
        conv.get("best_reply_example", "")
        for conv in top
        if conv.get("best_reply_example")
    ][:10]

    worst_replies = [
        conv.get("worst_reply_example", "")
        for conv in weak
        if conv.get("worst_reply_example")
    ][:10]

    suggestions = list({
        conv.get("improvement_suggestion", "")
        for conv in (top + weak)
        if conv.get("improvement_suggestion")
    })

    languages: dict[str, int] = {}
    for conv in top + weak + ranked_data.get("best_qualifiers", []):
        lang = conv.get("language_detected", "unknown")
        languages[lang] = languages.get(lang, 0) + 1

    return {
        "good_patterns_frequency": dict(sorted(good_patterns.items(), key=lambda x: x[1], reverse=True)),
        "common_weaknesses_frequency": dict(sorted(common_weaknesses.items(), key=lambda x: x[1], reverse=True)),
        "best_reply_examples": best_replies,
        "worst_reply_examples": worst_replies,
        "improvement_suggestions": suggestions,
        "language_distribution": languages,
        "avg_messages_to_qualify": sum(c.get("messages_to_qualify", 0) for c in top) / max(len(top), 1),
        "extracted_at": datetime.now().isoformat(),
    }


def generate_behavior_improvements(patterns: dict, store_id: str | None = None) -> dict:
    """
    Uses Claude AI to generate structured behavior improvement rules
    from extracted patterns.

    SAFETY: Only generates communication behavior improvements.
    Never modifies prices, stock, shipping, or product facts.
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    good_patterns = json.dumps(patterns.get("good_patterns_frequency", {}), indent=2)
    weaknesses = json.dumps(patterns.get("common_weaknesses_frequency", {}), indent=2)
    best_replies = "\n".join([f"- {r}" for r in patterns.get("best_reply_examples", [])[:5]])
    worst_replies = "\n".join([f"- {r}" for r in patterns.get("worst_reply_examples", [])[:5]])
    suggestions = "\n".join([f"- {s}" for s in patterns.get("improvement_suggestions", [])[:5]])
    avg_qualify = patterns.get("avg_messages_to_qualify", 0)

    generation_prompt = f"""You are a sales communication improvement specialist for COD e-commerce in Algeria.
Based on real conversation analysis data, generate actionable behavior improvements.

ANALYSIS DATA:

Good patterns found in top-performing conversations:
{good_patterns}

Common weaknesses in poor conversations:
{weaknesses}

Best reply examples from top conversations:
{best_replies}

Worst reply examples from weak conversations:
{worst_replies}

Improvement suggestions extracted:
{suggestions}

Average messages to qualify a lead: {avg_qualify:.1f}

Generate improvements and respond ONLY with this JSON structure:
{{
  "stable_behavior_rules": [
    "<rule 1: a clear, actionable behavior rule that applies to all conversations>",
    "<rule 2>",
    "<rule 3>",
    "<rule 4>",
    "<rule 5>"
  ],
  "communication_style_guidance": [
    "<style tip 1: how to phrase replies better>",
    "<style tip 2>",
    "<style tip 3>"
  ],
  "qualification_improvements": [
    "<specific improvement for qualifying leads faster>",
    "<specific improvement 2>"
  ],
  "closing_improvements": [
    "<specific improvement for closing orders>",
    "<specific improvement 2>"
  ],
  "patterns_to_avoid": [
    "<specific bad pattern to avoid>",
    "<specific bad pattern 2>",
    "<specific bad pattern 3>"
  ],
  "quick_win_suggestions": [
    "<one quick change that would immediately improve quality>",
    "<quick win 2>"
  ],
  "prompt_addon": "<a short paragraph (max 5 lines) that can be added to the agent system prompt to improve communication quality based on this analysis. Must be generic, not store-specific. Must not mention prices or products.>",
  "confidence_score": <0.0-1.0: how confident are you in these improvements based on the data quality>,
  "improvement_summary": "<2-3 sentence summary of the main findings and what will improve>"
}}

IMPORTANT CONSTRAINTS:
- Only improve communication behavior (tone, speed, naturalness, progression)
- Do NOT suggest changing product prices, shipping fees, or stock info
- Keep all rules generic enough for any Algerian COD store
- Do not hardcode any brand name, product name, or specific price
- Rules must work for Arabic darija, Latin darija, and French"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": generation_prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json\n?", "", raw)
        raw = re.sub(r"```\n?", "", raw)

        improvements = json.loads(raw)
        improvements["store_id"] = store_id
        improvements["generated_at"] = datetime.now().isoformat()
        improvements["status"] = "pending_approval"  # Must be approved before injection

        return improvements

    except Exception as e:
        print(f"[Optimizer] Improvement generation failed: {e}")
        return {}


def save_approved_improvements(improvements: dict, store_id: str | None = None) -> bool:
    """
    Saves approved improvements to file.
    Status must be 'approved' before agent.py can use them.

    SAFETY: Only approved improvements are ever injected into agent.py.
    """
    try:
        existing: dict = {}
        if os.path.exists(APPROVED_IMPROVEMENTS_FILE):
            with open(APPROVED_IMPROVEMENTS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)

        key = store_id or "global"

        if improvements.get("status") != "approved":
            print(f"[Optimizer] Improvements not saved — status is '{improvements.get('status')}', must be 'approved'")
            return False

        existing[key] = {
            **improvements,
            "approved_at": datetime.now().isoformat(),
        }

        with open(APPROVED_IMPROVEMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"[Optimizer] Improvements saved for store: {key}")
        return True

    except Exception as e:
        print(f"[Optimizer] Save failed: {e}")
        return False


def get_behavior_improvement_layer(store_id: str | None = None, context: str | None = None) -> str:
    """
    Called by agent.py on every request.
    Returns the approved improvement layer to inject into the system prompt.

    Returns empty string if no approved improvements exist.
    This is the ONLY function agent.py should call from this module.
    Always fails safe — never crashes agent.py.
    """
    try:
        if not os.path.exists(APPROVED_IMPROVEMENTS_FILE):
            return ""

        with open(APPROVED_IMPROVEMENTS_FILE, "r", encoding="utf-8") as f:
            all_improvements = json.load(f)

        # Store-specific first, fall back to global
        improvements = all_improvements.get(store_id) if store_id else None
        improvements = improvements or all_improvements.get("global")

        if not improvements:
            return ""

        if improvements.get("status") != "approved":
            return ""

        rules = improvements.get("stable_behavior_rules", [])
        style = improvements.get("communication_style_guidance", [])
        avoid = improvements.get("patterns_to_avoid", [])
        prompt_addon = improvements.get("prompt_addon", "")

        if not rules and not prompt_addon:
            return ""

        layer = "\n=== COMMUNICATION IMPROVEMENT LAYER ===\n"
        layer += "# Generated from real conversation analysis. Follow these improvements:\n\n"

        if rules:
            layer += "BEHAVIOR RULES:\n"
            for i, rule in enumerate(rules[:5], 1):
                layer += f"{i}. {rule}\n"

        if style:
            layer += "\nCOMMUNICATION STYLE:\n"
            for tip in style[:3]:
                layer += f"- {tip}\n"

        if avoid:
            layer += "\nAVOID THESE PATTERNS:\n"
            for pattern in avoid[:3]:
                layer += f"✗ {pattern}\n"

        if prompt_addon:
            layer += f"\n{prompt_addon}\n"

        layer += "=== END IMPROVEMENT LAYER ===\n"

        return layer

    except Exception as e:
        print(f"[Optimizer] Failed to load improvement layer: {e}")
        return ""  # Always fail safe — return empty, never crash agent


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_optimization_pipeline(
    conversations: list,
    store_id: str | None = None,
    auto_approve: bool = False,
) -> dict:
    """
    Main pipeline. Runs full optimization cycle on a batch of conversations.

    Steps:
    1. Analyze each conversation with Claude AI
    2. Rank by quality score
    3. Extract recurring patterns
    4. Generate behavior improvements
    5. Save as pending (requires manual approval unless auto_approve=True)

    auto_approve should only be True in testing.
    In production, always leave as False — store owner approves in dashboard.
    """
    print(f"[Optimizer] Starting pipeline for {len(conversations)} conversations...")

    # Step 1 — Analyze
    analyzed = []
    limit = min(len(conversations), MAX_CONVERSATIONS_PER_RUN)
    for i, conv in enumerate(conversations[:limit]):
        print(f"[Optimizer] Analyzing {i + 1}/{limit}...")
        result = analyze_conversation_quality(conv)
        if result:
            analyzed.append(result)

    if not analyzed:
        print("[Optimizer] No conversations could be analyzed.")
        return {"status": "no_data"}

    # Step 2 — Rank
    ranked = rank_best_examples(analyzed)

    # Step 3 — Extract patterns
    patterns = extract_top_patterns(ranked)

    # Step 4 — Generate improvements
    improvements = generate_behavior_improvements(patterns, store_id)

    if not improvements:
        print("[Optimizer] Could not generate improvements.")
        return {"status": "generation_failed"}

    # Step 5 — Set status
    if auto_approve:
        improvements["status"] = "approved"
    else:
        improvements["status"] = "pending_approval"

    if auto_approve:
        save_approved_improvements(improvements, store_id)

    # Save raw output for review
    output = {
        "store_id": store_id,
        "analyzed_count": len(analyzed),
        "ranked_summary": {
            "total": ranked["total_analyzed"],
            "average_score": ranked["average_score"],
            "top_count": len(ranked["top_performers"]),
            "weak_count": len(ranked["weak_performers"]),
        },
        "patterns": patterns,
        "improvements": improvements,
        "run_at": datetime.now().isoformat(),
    }

    with open(OPTIMIZER_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Optimizer] Pipeline complete. Status: {improvements['status']}")
    print(f"[Optimizer] Avg quality score: {ranked['average_score']:.1f}/10")
    print(f"[Optimizer] Confidence: {improvements.get('confidence_score', 0):.0%}")

    if not auto_approve:
        print("[Optimizer] Improvements are PENDING APPROVAL.")
        print("[Optimizer] Review optimizer_outputs.json then call approve_improvements()")

    return output


def approve_improvements(store_id: str | None = None) -> bool:
    """
    Approves pending improvements for injection into agent.py.
    Should be called after reviewing optimizer_outputs.json.
    In production: triggered by the dashboard "Approve" button.
    """
    try:
        if not os.path.exists(OPTIMIZER_OUTPUT_FILE):
            print("[Optimizer] No pending outputs found.")
            return False

        with open(OPTIMIZER_OUTPUT_FILE, "r", encoding="utf-8") as f:
            output = json.load(f)

        improvements = output.get("improvements", {})
        improvements["status"] = "approved"

        return save_approved_improvements(improvements, store_id)

    except Exception as e:
        print(f"[Optimizer] Approval failed: {e}")
        return False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_conversation_for_analysis(messages: list) -> str:
    """Formats message list into readable text for AI analysis."""
    lines = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        content = msg.get("content", "")
        if not content or content.startswith("📷") or content.startswith("🎤"):
            continue
        role = "AGENT" if sender in ("bot", "assistant", "agent") else "CUSTOMER"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def get_optimizer_status(store_id: str | None = None) -> dict:
    """Returns current optimizer status for dashboard display."""
    status = {
        "has_pending": False,
        "has_approved": False,
        "last_run": None,
        "avg_score": None,
        "analyzed_count": 0,
        "improvement_summary": None,
        "confidence_score": None,
    }

    try:
        if os.path.exists(OPTIMIZER_OUTPUT_FILE):
            with open(OPTIMIZER_OUTPUT_FILE, "r", encoding="utf-8") as f:
                output = json.load(f)
            status["has_pending"] = output.get("improvements", {}).get("status") == "pending_approval"
            status["last_run"] = output.get("run_at")
            status["avg_score"] = output.get("ranked_summary", {}).get("average_score")
            status["analyzed_count"] = output.get("analyzed_count", 0)
            status["improvement_summary"] = output.get("improvements", {}).get("improvement_summary")
            status["confidence_score"] = output.get("improvements", {}).get("confidence_score")

        if os.path.exists(APPROVED_IMPROVEMENTS_FILE):
            with open(APPROVED_IMPROVEMENTS_FILE, "r", encoding="utf-8") as f:
                approved = json.load(f)
            key = store_id or "global"
            status["has_approved"] = key in approved

    except Exception:
        pass

    return status
