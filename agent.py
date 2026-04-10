import os
import json
import re
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder

LANGUAGE_CHOICE_PROMPT = "1️⃣ 🇩🇿 Darija\n2️⃣ دارجة\n3️⃣ 🇫🇷 Français\n4️⃣ 🇬🇧 English"
LANGUAGE_CHOICE_TRIGGER = "kifach thibbs ntkallam m3ak? / بأي لغة تحب نتكلمو؟\n\n" + LANGUAGE_CHOICE_PROMPT

def detect_language_choice(text: str) -> str | None:
    t = text.strip().lower()
    if t in ("1", "darija", "درجة", "darja", "1️⃣"):
        return "ar-latin"
    if t in ("2", "دارجة", "darija arabic", "2️⃣"):
        return "ar"
    if t in ("3", "français", "francais", "fr", "french", "3️⃣"):
        return "fr"
    if t in ("4", "english", "en", "inglizi", "4️⃣"):
        return "en"
    return None

def is_language_uncertain(text: str) -> bool:
    text_lower = text.lower().strip()
    words = text_lower.split()
    if len(words) <= 1:
        return True
    darija_pattern = r"\b(wach|wesh|3andi|3andek|3andkom|kifash|kifach|labas|bghit|na9der|9der|kayn|rani|chno|9oli|nqdar|n3awnk|daba|safi|wakha|yallah|bzaf|sahbi|a5i|a7i|mzyan|zin)\b"
    if re.search(darija_pattern, text_lower):
        return False
    if re.search(r'[39782]', text):
        return False
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if arabic_chars > 2:
        return False
    if len(words) <= 2:
        return True
    return False

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.ENGLISH, Language.ARABIC
        ).build()
    return _detector

LANG_MAP = {
    Language.FRENCH: "fr",
    Language.ENGLISH: "en",
    Language.ARABIC: "ar",
}

def detect_language(text: str, locked_language: str | None) -> str:
    text_lower = text.lower().strip()
    darija_numbers = bool(re.search(r'[39782]', text))
    darija_words = [
        r"\b(wach|wesh|wash|ana|nta|ntia|la3ziz|hna|rabi|labas|bghit|bghina|3andi|3andek|3andkom|khoya|bzaf|mzyan|kifash|ndirek|lwaqt|doka|mazal|sahbi|hab|na3raf|ndir|ndirlak|na9der|9der|kayn|makaynch|chofli|chof|goul|goulha|dial|had|hada|hadi|rani|raha|rah|walo|baraka|3la|3lash|fhamt|smahli|wakha|yallah|safi|barak|felous|taman|chwiya|kima|kif|feen|mneen|imta|3lah|bach|ila|gaat|ga3|shi|mashi|hnaya|lhih)\b",
    ]
    for pattern in darija_words:
        if re.search(pattern, text_lower):
            return "ar-latin"
    if darija_numbers and len(text.split()) <= 6:
        return "ar-latin"
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if arabic_chars > 2:
        return "ar"
    french_words = r"\b(bonjour|merci|oui|non|je|tu|il|nous|vous|est|les|des|une|pour|avec|sur|mtn|dans|que|qui|comment|quel|quelle|parle|veux|voulez|pouvez|disponible|avez|avoir|salut|bonsoir|produit|commande|annuler|livraison)\b"
    if re.search(french_words, text_lower):
        return "fr"
    english_words = r"\b(hello|hi|yes|no|please|thank|thanks|order|cancel|product|available|price|want|need|help|what|how|can)\b"
    if re.search(english_words, text_lower):
        return "en"
    detected = get_detector().detect_language_of(text)
    lang = LANG_MAP.get(detected, None) if detected else None
    if lang is None:
        return "ar"  # DEFAULT: Darija Arabic script
    if lang == "fr" and locked_language in ("ar", "ar-latin"):
        return locked_language
    return lang


def build_system_prompt(
    store_name: str,
    ai_system_prompt: str | None,
    products: list,
    recent_orders: list,
    language: str,
    is_first_turn: bool,
    ai_flow_state: str | None,
    shipping_options: dict | None = None,
) -> str:

    lang_instructions = {
        "fr": "Réponds UNIQUEMENT en français. Même si le client écrit en darija, réponds en français.",
        "en": "Reply ONLY in English.",
        "ar": "أجب بالدارجة الجزائرية باللغة العربية فقط. مثال: إيه نقدر نعاونك، واش تحب تطلب؟",
        "ar-latin": "CRITICAL: Reply ONLY in Algerian Darija using Latin letters. Examples: 'labas, kifash naawenek?', 'wakha, ayy produit trid?', 'safi, nkamlo lorder'. NEVER reply in English or French when customer writes Darija Latin.",
    }
    lang_rule = lang_instructions.get(language, lang_instructions["ar"])

    if language == "fr":
        greeting_rule = (
            "Accueille le client chaleureusement en une phrase courte."
            if is_first_turn
            else "Ne répète PAS de salutation — la conversation est déjà en cours."
        )
    elif language == "en":
        greeting_rule = (
            "Greet the customer warmly in one short sentence."
            if is_first_turn
            else "Do NOT repeat a greeting — conversation is already in progress."
        )
    elif language == "ar":
        greeting_rule = (
            "رحب بالعميل بجملة قصيرة ودافئة بالدارجة الجزائرية."
            if is_first_turn
            else "لا تكرر التحية — المحادثة جارية بالفعل."
        )
    else:
        greeting_rule = (
            "Salute the customer in Darija Latin: e.g. 'salam, kifash naawenek?'"
            if is_first_turn
            else "Do NOT repeat greeting — conversation already started."
        )

    # Build product catalog
    if products:
        product_lines = []
        for i, p in enumerate(products, 1):
            variants_str = ""
            if p.variants:
                try:
                    v = p.variants if isinstance(p.variants, list) else json.loads(p.variants)
                    if v:
                        # Clean variant display: "Color: Bleu, Size: L" → "Colors: Bleu | Sizes: L, XL, XXL"
                        colors = [str(x).split(":")[-1].strip() for x in v if ":" in str(x) and not str(x).lower().startswith("size")]
                        sizes = [str(x).split(":")[-1].strip() for x in v if str(x).lower().startswith("size:")]
                        other = [str(x) for x in v if ":" not in str(x)]
                        parts = []
                        if colors:
                            parts.append("Colors: " + ", ".join(colors))
                        if sizes:
                            parts.append("Sizes: " + ", ".join(sizes))
                        if other:
                            parts.append(", ".join(other))
                        if parts:
                            variants_str = "\n   Variants: " + " | ".join(parts)
                except Exception:
                    pass
            image_str = ""
            if getattr(p, 'imageUrl', None):
                image_str = "\n   Image: " + str(p.imageUrl)
            desc_str = ""
            if getattr(p, 'description', None):
                desc_str = "\n   Description: " + str(p.description)
            product_lines.append(
                f"{i}. {p.name} — {float(p.price):,.0f} DZD (stock: {p.stock}){variants_str}{desc_str}{image_str}"
            )
        product_catalog = "\n\n".join(product_lines)
    else:
        product_catalog = "Aucun produit disponible." if language == "fr" else "لا توجد منتجات متاحة."

    if recent_orders:
        order_lines = [
            f"• #{o.orderNumber} | {o.status} | {o.customerName or 'unknown'} | {o.customerPhone or 'no phone'}"
            for o in recent_orders
        ]
        orders_context = "\n".join(order_lines)
    else:
        orders_context = "لا توجد طلبات حديثة."

    # Build shipping options section
    shipping_section = ""
    if shipping_options:
        home_price = shipping_options.get("home_delivery_price", 0)
        pickup_price = shipping_options.get("pickup_price", 0)
        has_home = shipping_options.get("home_delivery_enabled", True)
        has_pickup = shipping_options.get("pickup_enabled", False)

        if has_home and has_pickup:
            shipping_section = f"""
SHIPPING OPTIONS (show these to customer when they ask or when collecting order):
- الى البيت (Home Delivery): {home_price:,.0f} DZD
- من الفرع (Pickup from Branch): {pickup_price:,.0f} DZD
Ask customer which option they prefer."""
        elif has_home:
            shipping_section = f"""
SHIPPING: Home Delivery only — {home_price:,.0f} DZD added to order total."""
        elif has_pickup:
            shipping_section = f"""
SHIPPING: Pickup from Branch only — {pickup_price:,.0f} DZD."""
    else:
        shipping_section = "\nSHIPPING: Ask customer for delivery preference (home delivery or pickup)."

    flow_note = ""
    if ai_flow_state == "order_created":
        flow_note = "\nNOTE: A new order was just created for this customer. Confirm warmly without repeating all details."
    elif ai_flow_state == "order_cancelled":
        flow_note = "\nNOTE: The customer's order was just cancelled. Acknowledge it and ask if you can help further."
    elif ai_flow_state == "pending_cancel_choice":
        flow_note = "\nNOTE: Awaiting customer confirmation on which order to cancel."

    return f"""أنت وكيل دعم العملاء بالذكاء الاصطناعي لـ "{store_name}". تساعد العملاء على تقديم الطلبات وتتبعها وإلغائها.

اللغة الافتراضية: الدارجة الجزائرية بالخط العربي — هذه هي لغتك الأساسية حتى تكتشف لغة العميل.

LANGUAGE RULE (CRITICAL — NO EXCEPTIONS):
{lang_rule}
{greeting_rule}
NEVER say "I only communicate in [language]" — you speak ALL languages, you just reply in the detected one.
NEVER switch language mid-conversation unless the customer explicitly switches first.

DEFAULT LANGUAGE RULE:
- If you cannot detect the language → use Darija Arabic script (دارجة عربية)
- NEVER default to English or French
- When confused: "إيه نقدر نعاونك، واش تحب تطلب؟"

DARIJA HAS TWO FORMS — detect and match exactly:
1. DARIJA ARABIC (default): Customer writes Algerian dialect using Arabic script.
   Examples: "واش عندكم؟", "نقدر نطلب؟", "بغيت نعرف الثمن"
   → Reply ONLY in Arabic script Darija.
   → Example replies: "إيه نقدر نعاونك، واش تحب تطلب؟"

2. DARIJA LATIN: Customer writes Algerian dialect using Latin letters and numbers.
   Examples: "salam a5i", "na9der ndir order?", "wach 3andkom?"
   Numbers used as letters: 3=ع, 9=ق, 7=ح, 5=خ, 2=ء, 8=غ
   → Reply ONLY in Darija Latin. NEVER reply in French or English.
   → Example replies: "wah labas, ayy size trid?", "wakha, 3tini smetek"

3. MIXED: Customer mixes Darija Latin with French words → reply in Darija Latin.

VOCABULARY PREFERENCES:
- Use "دوكا" or "maintenant" instead of "mtn"
- Use "زين" instead of "mzyan"
- Use "نهار زين" instead of "nhar zyn"

BEHAVIOR:
- Be concise, warm, and professional.
- ONLY answer what the customer asks. Do not volunteer extra info.
- Never mention you are an AI unless directly asked.
- Never show raw data, IDs, or internal formats.
- NEVER use markdown: no **bold**, no *italic*, no # headers, no ~~strikethrough~~.{flow_note}

ALGERIAN CULTURAL CONTEXT:
- "ls hommes" or "les hommes" = compliment meaning "real man/bro" — NOT a product request
- "sahbi" / "صاحبي" = friend/buddy
- "a5i" / "أخي" = brother
- "cbn" / "تم" = okay/done
- "dcr" / "اوك" = okay/alright
- "pointeur" = size
- "ch7al dir" / "شحال دير" = how much does it cost
- "kifash ndir order?" / "كيفاش ندير أوردر؟" = how to place an order
- "wach 3andkom?" / "واش عندكم؟" = do you have
- "n9dar n3awd n9ra lmenu?" / "نقدر نعاود نقرا المينو؟" = can you repeat the menu?
- "nheb ncommande" / "نحب نكوموندي" = I want to order
- "n9dar ncanceli?" / "نقدر نلغي الكومند؟" = can I cancel the order?


CAPABILITIES:
1. Answer product questions (price, variants, stock, description).
2. When a customer asks about a specific product image, share its image URL if available.
3. Collect order details and confirm orders.
4. Cancel or check status of existing orders.

RESPONSE FORMAT:
- Maximum 2-3 sentences per reply unless showing order summary or product list.
- ZERO markdown: no **bold**, no *italic*, no bullet dashes for products.
- NEVER put multiple products on one line.

PRODUCT LIST FORMAT — use EXACTLY this structure:

1. [اسم المنتج] — [السعر] DZD
   Variants: [variant1], [variant2]

2. [اسم المنتج] — [السعر] DZD

ORDER COLLECTION — Required fields (collect one by one  ALL before confirming if customer provide missing one ask only for missing field):
1. الاسم الكامل (Full name)
2. الهاتف (Phone number)
3. الولاية (Wilaya)
4. البلدية (Baladiya/Commune)
5. العنوان (Delivery address)
6. خيار الشحن (Shipping option: الى البيت or من الفرع)
7. المنتج والكمية (Product + quantity + variant)

- Ask for 1-2 missing fields at a time, naturally.
- Only show confirmation summary when ALL fields are collected.
- Include shipping price in the total when showing summary.

ORDER CONFIRMATION FORMAT — use EXACTLY this structure:

[Confirmation phrase]:
- المنتج: [name + variant] × [qty]
- الاسم: [customer name]
- الهاتف: [phone]
- الولاية: [wilaya]
- البلدية: [baladiya]
- العنوان: [address]
- الشحن: [shipping option] — [shipping price] DZD
- المجموع: [product price + shipping] DZD

[هل كل شيء صحيح؟ / Is everything correct?]

CANCEL FLOW:
- Ask for phone number if not known.
- Confirm cancellation in one short message.
{shipping_section}

STORE PRODUCTS:
{product_catalog}

RECENT ORDERS (for status checks and cancellations):
{orders_context}

DARIJA ARABIC EXAMPLES (default language):
Customer: "السلام" → Reply: "وعليكم السلام، كيفاش نقدر نعاونك؟"
Customer: "واش راك" → Reply: "لاباس الحمد لله، كيفاش نعاونك؟"
Customer: "واش عندكم منتجات؟" → Reply: "إيه عندنا، شنو  المنتج اللي تحب؟"
Customer: "شحال هذا المنتج؟" → Reply: "هذا المنتج ب [price] DZD"
Customer: "كيفاش نطلب؟" → Reply: "ساهل، عطيني: المنتج + اسمك + تيليفون + ولاية + بلدية + عنوان"
Customer: "التوصيل لولايتي؟" → Reply: "إيه نوصلو لكل الولايات"
Customer: "الدفع عند الاستلام؟" → Reply: "وي ، تدفع كي يوصلك الكولي"
Customer: "نقدر نلغي الطلب؟" → Reply: "وي، قولي رقم الهاتف"
Customer: "شكرا بزاف" → Reply: "يعطيك الصحة ، نهار زين"

DARIJA LATIN EXAMPLES:
Customer: "salam" → Reply: "wa3lik salam, kifach nqdar n3awnk?"
Customer: "wach 3andkom produits?" → Reply: "eyh 3andna, chno hab t3raf?"
Customer: "nheb ncommande" → Reply: "parfait, 9oli chno produit hab?"
Customer: "livraison l wilaya ta3i?" → Reply: "eyh nwaslo lkoulli wilayat"
Customer: "n9dar ncanceli?" → Reply: "eyh, 9oli numéro telephone"
Customer: "merci bzaf" → Reply: "avec plaisir, nhar zin"

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
    "baladiya": string | null,
    "address": string | null,
    "shippingOption": "home_delivery" | "pickup" | null,
    "items": [{"productName": string, "price": number, "quantity": number, "variant": string | null}]
  } | null,
  "cancelPhone": string | null
}

Rules:
- canAutoCreate = true ONLY when these fields are present: customerName, customerPhone, wilaya, items AND customer has confirmed (yes/oui/wah/wakha/ايه/نعم/correct/c'est correct/bon/cbon)
- baladiya and address are optional — do not block canAutoCreate if missing
- shippingOption: default to "home_delivery" if customer says "à domicile", "livraison", "chez moi", "dar", "البيت". Use "pickup" only if explicitly stated
- Extract product name AND variant (color + size) from conversation context
- cancelPhone: extract phone number when intent is cancel_order
- For product_inquiry and other intents, orderData and cancelPhone can be null
- "cbon", "c bon", "correct", "c'est bon", "parfait" all count as confirmation"""


async def extract_order(history: list, products: list) -> dict:
    last_customer = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    skip_patterns = [
        r"\b(merci|thanks|thank you|shukran|شكرا|yishkrek|barak)\b",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, last_customer.lower()):
            return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

    messages = [{"role": "system", "content": EXTRACTION_PROMPT}]
    for m in history[-10:]:
        role = "user" if m.role == "customer" else "assistant"
        messages.append({"role": role, "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
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
    history = request.history
    last_customer_msg = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    chosen_language = detect_language_choice(last_customer_msg)
    if chosen_language:
        language = chosen_language
    else:
        language = detect_language(last_customer_msg, None)
        word_count = len(last_customer_msg.strip().split())
        if request.detectedLanguage and not chosen_language:
            # Lock to detected language more aggressively — prevent flip-flopping
            if word_count <= 5 or language == request.detectedLanguage:
                language = request.detectedLanguage
            elif request.detectedLanguage == "fr" and language != "ar":
                # If conversation is in French, keep French unless clear Arabic signal
                language = request.detectedLanguage
            elif request.detectedLanguage == "fr" and language != "ar":
                language = request.detectedLanguage

    prior_turns = [m for m in history[:-1] if m.role in ("customer", "agent", "bot")]
    is_first_turn = len(prior_turns) == 0

    if is_first_turn and is_language_uncertain(last_customer_msg) and not chosen_language:
        language = "ar"

    if chosen_language and len(prior_turns) <= 2:
        language = chosen_language

    shipping_options = getattr(request, 'shippingOptions', None)

    system_prompt = build_system_prompt(
        store_name=request.storeName,
        ai_system_prompt=request.aiSystemPrompt,
        products=request.products,
        recent_orders=request.recentOrders,
        language=language,
        is_first_turn=is_first_turn and not chosen_language,
        ai_flow_state=request.aiFlowState,
        shipping_options=shipping_options,
    )

    openai_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-20:]:
        if m.role == "customer":
            openai_messages.append({"role": "user", "content": m.content})
        elif m.role in ("agent", "bot"):
            openai_messages.append({"role": "assistant", "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=600,
        temperature=0.4,
        messages=openai_messages,
    )
    reply = response.choices[0].message.content.strip()

    extraction = await extract_order(history, request.products)

    action = {"type": "none"}

    if extraction.get("canAutoCreate") and extraction.get("orderData"):
        od = extraction["orderData"]

        # ── Fix item prices from product catalog ──────────────────────────────
        fixed_items = []
        for item in (od.get("items") or []):
            item_price = item.get("price")
            if not item_price:
                item_name = (item.get("productName") or "").lower()
                for p in request.products:
                    if item_name in p.name.lower() or p.name.lower() in item_name:
                        item_price = float(p.price)
                        break
            fixed_items.append({
                "productName": item.get("productName"),
                "price": float(item_price) if item_price else 0.0,
                "quantity": item.get("quantity") or 1,
                "variant": item.get("variant"),
            })

        action = {
            "type": "create_order",
            "customerName": od.get("customerName"),
            "customerPhone": od.get("customerPhone"),
            "wilaya": od.get("wilaya"),
            "address": f"{od.get('baladiya', '') or ''} - {od.get('address', '') or ''}".strip(" -") or None,
            "shippingOption": od.get("shippingOption") or "home_delivery",
            "items": fixed_items,
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