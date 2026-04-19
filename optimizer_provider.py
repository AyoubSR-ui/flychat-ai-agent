"""
optimizer_provider.py
Handles AI provider selection, model config, and client initialization.
Never mixes providers. Fails clearly if API key is missing.
"""

import os
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ── Model config ──────────────────────────────────────────
MODELS = {
    "default": {
        "provider": Provider.OPENAI,
        "model_id": "gpt-4o-mini",        # cost-safe default
        "input_cost_per_1k": 0.000150,    # USD per 1K input tokens
        "output_cost_per_1k": 0.000600,   # USD per 1K output tokens
        "context_window": 128000,
        "label": "Standard Analysis"
    },
    "deep": {
        "provider": Provider.OPENAI,
        "model_id": "gpt-4o",             # premium deep analysis
        "input_cost_per_1k": 0.002500,
        "output_cost_per_1k": 0.010000,
        "context_window": 128000,
        "label": "Deep Analysis"
    },
    "anthropic_default": {
        "provider": Provider.ANTHROPIC,
        "model_id": "claude-haiku-4-5-20251001",
        "input_cost_per_1k": 0.000800,
        "output_cost_per_1k": 0.004000,
        "context_window": 200000,
        "label": "Standard Analysis (Anthropic)"
    }
}

# ── Plan model limits ─────────────────────────────────────
PLAN_MODEL_LIMITS = {
    "free":     {"mode": "default", "max_conversations": 10,  "batch_size": 5},
    "starter":  {"mode": "default", "max_conversations": 30,  "batch_size": 10},
    "pro":      {"mode": "default", "max_conversations": 75,  "batch_size": 20},
    "agency":   {"mode": "deep",    "max_conversations": 150, "batch_size": 25},
}


def get_optimizer_provider() -> Provider:
    raw = os.environ.get("OPTIMIZER_PROVIDER", "openai").lower()
    try:
        return Provider(raw)
    except ValueError:
        raise ValueError(f"[Provider] Unknown provider: '{raw}'. Use 'openai' or 'anthropic'.")


def get_optimizer_mode() -> str:
    return os.environ.get("OPTIMIZER_MODE", "default")


def get_model_config(mode: str = None) -> dict:
    mode = mode or get_optimizer_mode()
    provider = get_optimizer_provider()

    # If anthropic provider explicitly set, use anthropic model
    if provider == Provider.ANTHROPIC:
        cfg = MODELS["anthropic_default"].copy()
        override = os.environ.get("OPTIMIZER_MODEL")
        if override:
            cfg["model_id"] = override
        return cfg

    cfg = MODELS.get(mode, MODELS["default"]).copy()
    override = os.environ.get("OPTIMIZER_MODEL")
    if override:
        cfg["model_id"] = override
    return cfg


def get_plan_limits(plan: str) -> dict:
    return PLAN_MODEL_LIMITS.get(plan, PLAN_MODEL_LIMITS["starter"])


def get_optimizer_client(provider: Provider = None):
    """Returns initialized client for selected provider. Fails clearly."""
    provider = provider or get_optimizer_provider()

    if provider == Provider.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[Provider] OPTIMIZER_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    elif provider == Provider.ANTHROPIC:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[Provider] OPTIMIZER_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
            )
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    raise ValueError(f"[Provider] Unsupported provider: {provider}")


def call_model(client, provider: Provider, model_id: str, prompt: str, max_tokens: int = 1500) -> tuple[str, int, int]:
    """
    Unified model call. Returns (response_text, input_tokens, output_tokens).
    Never mixes providers.
    """
    if provider == Provider.OPENAI:
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens

    elif provider == Provider.ANTHROPIC:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return text, input_tokens, output_tokens

    raise ValueError(f"[Provider] Cannot call unknown provider: {provider}")
