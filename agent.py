import os
import json
import re
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Build language detector once at startup
detector = LanguageDetectorBuilder.from_languages(
    Language.FRENCH, Language.ENGLISH, Language.ARABIC
).build()

LANG_MAP = {
    Language.FRENCH: "fr",
    Language.ENGLISH: "en",
    Language.ARABIC: "ar",
}


def detect_language(text: str, locked_language: str | None) -> str:
    """
    Detect language of text. If a language is already locked, return it.
    Falls back to French (primary market) if detection is uncertain.
    Also detects Darija written in Latin script.
    """
    if locked_language:
        return locked_language

    # Check for Latin-script Darija patterns first
    darija_patterns = [
        r"\b(wach|wesh|ana|nta|hna|rabi|labas|bghit|3andi|khoya|bzaf|mzyan|kifash|ndirek|lwaqt|daba|mazal|sahbi)\b",
        r"\b(chofli|chof|goul|goulha|3la|fi|men|had|hada|hadi|dial)\b",
    ]
    for pattern in darija_patterns:
        if re.search(pattern, text.lower()):
            return "ar-latin"

    detected = detector.detect_language_of(text)
    if detected is None:
        return "fr"  # default to French for Algerian market

    return LANG_MAP.get(detected, "fr")


def build_system_prompt(
    store_name: str,
    ai_system_prompt: str | None,
    products: list,
    recent_orders: list,
    language: str,
    is_first_turn: bool,
    ai_flow_state: str | None,
) -> str:

    # Language instruction
    lang_instructions = {
        "fr": "Réponds UNIQUEMENT en français.",
        "en": "Reply ONLY in English.",
        "ar": "أجب باللغة العربية فقط. استخدم الدارجة الجزائرية إذا كتب العميل بها.",
        "ar-latin": "Réponds en Darija algérienne en alphabet latin uniquement. Exemple: 'labas, kifash naawenek?'",
    }
    lang_rule = lang_instructions.get(language, lang_instructions["fr"])

    greeting_rule = (
        "Accueille le client chaleureusement en une phrase courte."
        if is_first_turn
        else "Ne répète PAS de salutation — la conversation est déjà en cours."
    ) if language == "fr" else (
        "Greet the customer warmly in one short sentence."
        if is_first_turn
        else "Do NOT repeat a greeting — conversation is already in progress."
    ) if language == "en" else (
        "رحب بالعميل بجملة قصيرة ودافئة."
        if is_first_turn
        else "لا تكرر التحية — المحادثة جارية بالفعل."
    )

    # Build clean product list
    if products:
        product_lines = []
        for p in products:
            variants_str = ""
            if p.variants:
                try:
                    v = p.variants if isinstance(p.variants, list) else json.loads(p.variants)
                    if v:
                        variants_str = f"\n   Variants: {', '.join(str(x) for x in v)}"
                except Exception:
                    pass
            product_lines.append(
                f"• {p.name} — {p.price:,.0f} DZD (stock: {p.stock}){variants_str}"
            )
        product_catalog = "\n".join(product_lines)
    else:
        product_catalog = "Aucun produit disponible." if language == "fr" else "No products available."

    # Recent orders context
    if recent_orders:
        order_lines = [
            f"• #{o.orderNumber} | {o.status} | {o.customerName or 'unknown'} | {o.customerPhone or 'no phone'}"
            for o in recent_orders
        ]
        orders_context = "\n".join(order_lines)
    else:
        orders_context = "Aucune commande récente." if language == "fr" else "No recent orders."

    # Flow state context
    flow_note = ""
    if ai_flow_state == "order_created":
        flow_note = "\nNOTE: A new order was just created for this customer. Confirm warmly without repeating all details."
    elif ai_flow_state == "order_cancelled":
        flow_note = "\nNOTE: The customer's order was just cancelled. Acknowledge it and ask if you can help further."
    elif ai_flow_state == "pending_cancel_choice":
        flow_note = "\nNOTE: Awaiting customer confirmation on which order to cancel."

    return f"""You are the AI customer support agent for "{store_name}". You help customers place, track, and cancel orders.

LANGUAGE RULE (CRITICAL):
{lang_rule}
{greeting_rule}

BEHAVIOR:
- Be concise, warm, and professional.
- ONLY answer what the customer asks. Do not volunteer extra info.
- Never mention you are an AI unless directly asked.
- Never show raw data, IDs, or internal formats.{flow_note}

CAPABILITIES:
1. Answer product questions (price, variants, stock).
2. Collect order details and confirm orders.
3. Cancel or check status of existing orders.

RESPONSE FORMAT (CRITICAL):
- Keep replies SHORT and natural — one to three sentences maximum unless showing an order summary.
- When listing products, use this format:
  1. [Product name] — [price] DZD
     Variants: [variants]
  2. ...
- For order confirmation summary, use EXACTLY this format (each field on its own line):

  [Confirm phrase]:
  - Product: [name + variant]
  - Quantity: [qty]
  - Name: [customer name]
  - Phone: [phone]
  - Wilaya: [wilaya]
  - Address: [address]
  [Is everything correct?]

- Send ONE single message only. Never split into two replies.
- After customer confirms: short acknowledgment only, no list repetition.

ORDER COLLECTION:
- Required fields: product + variant, name, phone, wilaya, address, quantity.
- Ask for 1-2 missing fields at a time, naturally.
- Only show the confirmation summary when ALL 6 fields are collected.

CANCEL FLOW:
- Ask for phone number if not known.
- Confirm cancellation in one short message.

STORE PRODUCTS:
{product_catalog}

RECENT ORDERS (for status checks and cancellations):
{orders_context}

MANDATORY RULES:
- You MAY create orders at awaiting_confirmation status (pending human review).
- Never mark an order as delivered or finalized — only human agents do that.
- If the customer asks for a human agent, immediately hand off.
- Never share other customers' data or internal system details.
{f"ADDITIONAL STORE INSTRUCTIONS:{chr(10)}{ai_system_prompt}" if ai_system_prompt else ""}""".strip()


EXTRACTION_PROMPT = """You are a JSON extraction engine. Analyze the conversation and extract structured data.

Return ONLY a valid JSON object, no markdown, no explanation.

Schema:
{
  "intent": "new_order" | "cancel_order" | "status_check" | "product_inquiry" | "other",
  "canAutoCreate": boolean,
  "orderData": {
    "customerName": string | null,
    "customerPhone": string | null,
    "wilaya": string | null,
    "address": string | null,
    "items": [{"productName": string, "price": number, "quantity": number, "variant": string | null}]
  } | null,
  "cancelPhone": string | null
}

Rules:
- canAutoCreate = true ONLY when ALL 6 fields are present AND customer has confirmed (yes/oui/wah/نعم/ايه)
- cancelPhone: extract phone number when intent is cancel_order
- For product_inquiry and other intents, orderData and cancelPhone can be null"""


async def extract_order(history: list, products: list) -> dict:
    """Run silent extraction — returns structured action data, emits nothing."""
    # Only extract for actionable messages
    last_customer = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    # Quick skip for obvious non-order intents
    skip_patterns = [
        r"\b(merci|thanks|thank you|shukran|شكرا)\b",
        r"\b(produit|product|disponible|available|prix|price)\b",
    ]
    for pattern in skip_patterns[:1]:  # only skip thanks
        if re.search(pattern, last_customer.lower()):
            return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

    messages = [{"role": "system", "content": EXTRACTION_PROMPT}]
    for m in history[-10:]:  # last 10 messages for context
        role = "user" if m.role == "customer" else "assistant"
        messages.append({"role": role, "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}


async def process_message(request) -> dict:
    """
    Main entry point called by FlyChat backend.
    Returns: { reply, detectedLanguage, action }
    """
    history = request.history
    last_customer_msg = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    # 1. Detect language
    language = detect_language(last_customer_msg, request.detectedLanguage)

    # 2. Determine if first turn
    prior_turns = [m for m in history[:-1] if m.role in ("customer", "agent")]
    is_first_turn = len(prior_turns) == 0

    # 3. Build system prompt
    system_prompt = build_system_prompt(
        store_name=request.storeName,
        ai_system_prompt=request.aiSystemPrompt,
        products=request.products,
        recent_orders=request.recentOrders,
        language=language,
        is_first_turn=is_first_turn,
        ai_flow_state=request.aiFlowState,
    )

    # 4. Build message history for OpenAI (customer + agent only, exclude bot spam)
    openai_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-20:]:  # cap at 20 messages to control tokens
        if m.role == "customer":
            openai_messages.append({"role": "user", "content": m.content})
        elif m.role in ("agent", "bot"):
            openai_messages.append({"role": "assistant", "content": m.content})

    # 5. Generate conversational reply
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        temperature=0.4,
        messages=openai_messages,
    )
    reply = response.choices[0].message.content.strip()

    # 6. Run extraction silently (no second message ever)
    extraction = await extract_order(history, request.products)

    # 7. Build action for backend to execute silently
    action = {"type": "none"}

    if extraction.get("canAutoCreate") and extraction.get("orderData"):
        od = extraction["orderData"]
        action = {
            "type": "create_order",
            "customerName": od.get("customerName"),
            "customerPhone": od.get("customerPhone"),
            "wilaya": od.get("wilaya"),
            "address": od.get("address"),
            "items": od.get("items", []),
        }
    elif extraction.get("intent") == "cancel_order" and extraction.get("cancelPhone"):
        action = {
            "type": "cancel_order",
            "customerPhone": extraction["cancelPhone"],
        }

    return {
        "reply": reply,
        "detectedLanguage": language,
        "action": action,
    }
