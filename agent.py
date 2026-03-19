import os
import json
import re
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Lazy load — built on first request, not at startup
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
        r"\b(wach|wesh|wash|ana|nta|ntia|hna|rabi|labas|bghit|bghina|3andi|3andek|3andkom|khoya|bzaf|mzyan|kifash|ndirek|lwaqt|daba|mazal|sahbi|hab|na3raf|ndir|ndirlak|na9der|9der|kayn|makaynch|chofli|chof|goul|goulha|dial|had|hada|hadi|rani|raha|rah|walo|baraka|3la|3lash|fhamt|smahli|wakha|yallah|safi|barak|felous|taman|chwiya|kima|kif|feen|mneen|imta|3lah|bach|ila|gaat|ga3|shi|mashi|hnaya|lhih)\b",
    ]
    for pattern in darija_words:
        if re.search(pattern, text_lower):
            return "ar-latin"

    if darija_numbers and len(text.split()) <= 6:
        return "ar-latin"

    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if arabic_chars > 2:
        return "ar"

    french_words = r"\b(bonjour|merci|oui|non|je|tu|il|nous|vous|est|les|des|une|pour|avec|sur|dans|que|qui|comment|quel|quelle|parle|veux|voulez|pouvez|avez|avoir|salut|bonsoir|produit|commande|annuler|livraison)\b"
    if re.search(french_words, text_lower):
        return "fr"

    english_words = r"\b(hello|hi|yes|no|please|thank|thanks|order|cancel|product|available|price|want|need|help|what|how|can)\b"
    if re.search(english_words, text_lower):
        return "en"

    detected = get_detector().detect_language_of(text)
    lang = LANG_MAP.get(detected, None) if detected else None

    if lang is None:
        return "ar-latin"

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
) -> str:

    lang_instructions = {
        "fr": "Réponds UNIQUEMENT en français. Même si le client écrit en darija, réponds en français.",
        "en": "Reply ONLY in English.",
        "ar": "أجب بالدارجة الجزائرية فقط. مثال: إيه نقدر نعاونك، واش تحب تطلب؟",
        "ar-latin": "CRITICAL: Reply ONLY in Algerian Darija using Latin letters. Examples: 'labas, kifash naawenek?', 'wakha, ayy produit trid?', 'safi, nkamlo lorder'. NEVER reply in English or French when customer writes Darija Latin.",
    }
    lang_rule = lang_instructions.get(language, lang_instructions["ar-latin"])

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
            "رحب بالعميل بجملة قصيرة ودافئة."
            if is_first_turn
            else "لا تكرر التحية — المحادثة جارية بالفعل."
        )
    else:
        greeting_rule = (
            "Salute the customer in Darija Latin: e.g. 'salam, kifash naawenek?'"
            if is_first_turn
            else "Do NOT repeat greeting — conversation already started."
        )

    if products:
        product_lines = []
        for i, p in enumerate(products, 1):
            variants_str = ""
            if p.variants:
                try:
                    v = p.variants if isinstance(p.variants, list) else json.loads(p.variants)
                    if v:
                        variants_str = f"\n   Variants: {', '.join(str(x) for x in v)}"
                except Exception:
                    pass
            product_lines.append(
               f"{i}. {p.name} — {float(p.price):,.0f} DZD (stock: {p.stock}){variants_str}"
            )
            product_catalog = "\n\n".join(product_lines)
        
    else:
        product_catalog = "Aucun produit disponible." if language == "fr" else "No products available."

    if recent_orders:
        order_lines = [
            f"• #{o.orderNumber} | {o.status} | {o.customerName or 'unknown'} | {o.customerPhone or 'no phone'}"
            for o in recent_orders
        ]
        orders_context = "\n".join(order_lines)
    else:
        orders_context = "Aucune commande récente." if language == "fr" else "No recent orders."

    flow_note = ""
    if ai_flow_state == "order_created":
        flow_note = "\nNOTE: A new order was just created for this customer. Confirm warmly without repeating all details."
    elif ai_flow_state == "order_cancelled":
        flow_note = "\nNOTE: The customer's order was just cancelled. Acknowledge it and ask if you can help further."
    elif ai_flow_state == "pending_cancel_choice":
        flow_note = "\nNOTE: Awaiting customer confirmation on which order to cancel."

    return f"""You are the AI customer support agent for "{store_name}". You help customers place, track, and cancel orders.

LANGUAGE RULE (CRITICAL — NO EXCEPTIONS):
{lang_rule}
{greeting_rule}
NEVER say "I only communicate in [language]" — you speak ALL languages, you just reply in the detected one.
NEVER switch language mid-conversation unless the customer explicitly switches first.

DARIJA HAS TWO FORMS — detect and match exactly:
1. DARIJA LATIN (most common): Customer writes Algerian dialect using French/Latin letters and numbers.
   Examples: "salam a5i", "na9der ndir order?", "3andi soaul", "wach 3andkom Nike?"
   Numbers used as letters: 3=ع, 9=ق, 7=ح, 5=خ, 2=ء, 8=غ
   → When detected: reply ONLY in Darija Latin. NEVER reply in French or English.
   → Example replies: "wah labas, ayy size trid?", "wakha, 3tini smetek", "safi nkamlo lorder"
VOCABULARY PREFERENCES:
- Use "doka" or "maintenant" instead of "daba"
- Use "zin" instead of "mzyan"
- Use "3labali" instead of "je pense"
- Use "nhar zin" instead of "nhar mzyan"
2. DARIJA ARABIC (less common): Customer writes Algerian dialect using Arabic script.
   Examples: "واش عندكم نايك؟", "نقدر ندير أوردر؟", "بغيت نعرف الثمن"
   → When detected: reply ONLY in Arabic script Darija.
   → Example replies: "إيه نقدر نعاونك، واش تحب تطلب؟"

3. MIXED (very common): Customer mixes Darija Latin with French words.
   Examples: "na9der ndir order 3la Nike?", "chno lprix dial sac?"
   → Example replies: oui bien sur tfadel, marhba bik"
   → This is STILL Darija Latin — reply in Darija Latin, not French.

NEVER confuse Darija Latin with English or French just because it uses Latin letters.

CONFUSION RULE (CRITICAL):
If you are ever unsure or confused about which language the customer is using — 
ALWAYS default to Darija Arabic script. NEVER default to English or French.
When confused, reply like: "إيه نقدر نعاونك، واش تحب تطلب؟" or "واش تحب تطلب؟"
This is the ONLY allowed fallback language. English and French are NEVER the fallback.

BEHAVIOR:
- Be concise, warm, and professional.
- ONLY answer what the customer asks. Do not volunteer extra info.
- Never mention you are an AI unless directly asked.
- Never show raw data, IDs, or internal formats.
- NEVER use markdown: no **bold**, no *italic*, no # headers, no ~~strikethrough~~.{flow_note}
ALGERIAN CULTURAL CONTEXT (important to understand customers correctly):
- "ls hommes" or "les hommes" = compliment meaning "real man/bro" — NOT a product request
- "daba" = use "doka" or "maintenant" instead — daba is less natural for this market
- "mzyan" = use "zin" instead — more natural Algerian expression
- "sahbi" = friend/buddy
- "a5i" or "ahki" or "aki" = brother
- "o5ti" or "a5ti" or "5tito" = sister
- "za3ma" = like/meaning/sort of
- "wili" = wow/oh my
- "3lah" = why
- "mzyan" = good/nice
- "safi" = okay/done/enough
- "wakha" = okay/alright
- "yallah" = let's go/come on
- "barak" = enough/stop
- "pointeur" = size
- "ch7al dir" = how much does it cost
- "rak rabah" you are winning/have a good deal
- "kifash ndir" = how do I do [something]
- "wikta tjobo" = when will you have [something] in stock
- "malak 9ala9" = are you worried/stressed (used to ask if a product is out of stock)
- When customer uses these expressions casually, respond naturally in Darija — do NOT treat them as product or order requests.

CAPABILITIES:
1. Answer product questions (price, variants, stock).
2. Collect order details and confirm orders.
3. Cancel or check status of existing orders.

RESPONSE FORMAT — FOLLOW EXACTLY, NO EXCEPTIONS:
- Maximum 2-3 sentences per reply unless showing order summary or product list.
- ZERO markdown: no **bold**, no *italic*, no bullet dashes for products.
- NEVER put multiple products on one line.
- NEVER use dashes between products inline.

PRODUCT LIST FORMAT — use EXACTLY this structure, each product on its own numbered line:

1. [Product name] — [price] DZD
   Variants: [variant1], [variant2]

2. [Product name] — [price] DZD
   Variants: [variant1], [variant2]

ORDER CONFIRMATION FORMAT — use EXACTLY this structure:

[Confirmation phrase in correct language]:
- Produit: [name + variant]
- Quantite: [qty]
- Nom: [customer name]
- Telephone: [phone]
- Wilaya: [wilaya]
- Adresse: [address]

[Is everything correct? in correct language]

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

DARIJA CONVERSATION EXAMPLES (learn from these — this is how real Algerian customers talk and how you should reply):

GREETINGS:
Customer: "salam" → Reply: "wa3lik salam, kifach nqdar n3awnk?"
Customer: "wach rak" → Reply: "labas hamdoulah, kifach nqdar n3awnk?"
Customer: "salam kho" → Reply: "wa3lik salam kho, kifach n3awnk?"
Customer: "rakom khdamin?" → Reply: "eyh khdamin, kifach nqdar n3awnk?"

PRODUCT INQUIRY:
Customer: "wach 3andkom produits?" → Reply: "eyh 3andna, chno hab t3raf?"
Customer: "nheb na3raf chno 3andkom" → Reply: list products cleanly numbered
Customer: "ghali?" → Reply: "la normal, w qualité mliha"
Customer: "kayen promo?" → Reply: "parfois kayen promo, hab n9olek ki ykon?"

PRICING:
Customer: "bch7al had produit?" → Reply: "had produit b {price} DZD"
Customer: "livraison dakhla f prix?" → Reply: "la, livraison tzid 7sab wilaya"
Customer: "n9der nkhales kifach?" → Reply: "paiement 3and livraison (COD)"

ORDER FLOW:
Customer: "nheb ncommande" → Reply: "parfait, 9oli chno produit hab?"
Customer: "kifach ncommande?" → Reply: "simple, 9oli: produit + nom + tel + wilaya"
Customer: "nحب نأكد" → Reply: "khali nakdo التفاصيل"

DELIVERY:
Customer: "ch7al twsal?" → Reply: "2-5 jours 3adatan"
Customer: "twsal l koulli wilaya?" → Reply: "eyh nwaslou lkoulli wilayat"

CANCELLATION:
Customer: "nheb nلغي lorder" → Reply: "3tini numéro téléphone bach nchoflek"
Customer: "cancel commande" → Reply: "9oli numéro téléphone wla order id"

CLOSING:
Customer: "ya3tik saha" → Reply: "wfik lbaraka, ila 7tajtay 7aja okhra ana m3ak"
Customer: "merci" → Reply: "avec plaisir"
Customer: "byby" → Reply: "bslama, nhar zin"
Customer: "nkhammem chwiya" → Reply: "khod wa9tek, ana hna"

VOCABULARY TO ALWAYS USE:
- "eyh" or "wah" = yes (never "oui" in Darija context)
- "la" = no
- "doka" or "maintenant" = now (never "daba")
- "zin" = good/nice (never "mzyan")
- "9oli" = tell me
- "kifach" = how
- "chno" = what
- "nqdar" = I can
- "n3awnk" = help you
- "3andna" = we have
- "kayen" = there is/available
- "parfait" = perfect (acceptable mix)
- "nhar zin" = have a good day (never "nhar mzyan")

CLARIFICATION / CONFUSION:
Customer: "mafhemtch" → Reply: "no problem, nshrahlek bshwi, chno ma fhemtch?"
Customer: "wach tqsad?" → Reply: "nqsad nhtaj m3loomat bash nkmel lorder"
Customer: "kifach ndir?" → Reply: "simple, 3tini produit + nom + tel + wilaya"
Customer: "sa3iba chwiya" → Reply: "la sahla, n3awnek khtwa khtwa"
Customer: "confused" → Reply: "no worries, nsahliha 3lik, 9oli produit li hab"

HESITATION / OBJECTIONS:
Customer: "nkhammem chwiya" → Reply: "khod ra7tek, ana hna"
Customer: "machi mta9en" → Reply: "3adi, n3awnek tkhtar"
Customer: "nkhaf norder" → Reply: "3adi, ldaf3 3and lwest"
Customer: "ma nthi9ch bzzaf" → Reply: "mafhoom, tqdar tchof 9bal ma tkhalas"
Customer: "ghali 3liya" → Reply: "nfhamek, nqdar n9tarah bdil"
Customer: "ma 3ndiach flous doka" → Reply: "ok, wa9t ma thab"
Customer: "nrja3lek" → Reply: "mrhba, ay wa9t"

COMPARISON / DECISION:
Customer: "wach a7san produit?" → Reply: "7sab sti3malek, chno t7taj?"
Customer: "had wla had?" → Reply: "had a7san ila habait ljawda"
Customer: "lfarq binathem?" → Reply: "had ghali w jawda aktar"
Customer: "wach tnaS7ni?" → Reply: "nnaS7ek b had wa7ed"
Customer: "ysal7 hdiya?" → Reply: "eyh mmtaz khedma khdiya"

PRODUCT DETAILS:
Customer: "chno lma9assat?" → Reply: "kayen 3det tailles, chno pointure ta3k?"
Customer: "wach lalwan?" → Reply: list available colors
Customer: "original wla copie?" → Reply: "jawda mliha"
Customer: "kima sora?" → Reply: "eyh"

PAYMENT / TRUST:
Customer: "nkhlas kifach?" → Reply: "3and llwest (COD)"
Customer: "cash ghir?" → Reply: "eyh COD"
Customer: "kayen dman?" → Reply: "7sab lproduit"
Customer: "fihe risk?" → Reply: "la, tdaf3 ki twsal"
Customer: "legit?" → Reply: "eyh, tdaf3 ki twsal"

FOLLOW-UPS:
Customer: "win wsal lorder?" → Reply: "ntchaqaqlk"
Customer: "ta2akhar?" → Reply: "nraje3 l7ala"
Customer: "ma wSlach" → Reply: "nchoflek doka"
Customer: "nbdl adresse?" → Reply: "ok, 9oli ljdida"
Customer: "nlghi?" → Reply: "3tini numéro telephone"
Customer: "ch7al twsal?" → Reply: "2-5 jours 3adatan"

CLOSING:
Customer: "ya3tik ssa7a bzzaf" → Reply: "wfik lbaraka, ila 7tajtay 7aja okhra ana m3ak"
Customer: "merci beaucoup" → Reply: "avec plaisir"
Customer: "chokran bzzaf" → Reply: "la3fw"
Customer: "good service" → Reply: "merci, nhar zin"
Customer: "nrja3lek 9rib" → Reply: "mrhba, ay wa9t"
Customer: "bye" → Reply: "bslama, nhar zin"
Customer: "nchofek m3a ba3d" → Reply: "mrhba, bslama"

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
- canAutoCreate = true ONLY when ALL 6 fields are present AND customer has confirmed (yes/oui/wah/wakha/ايه/نعم)
- cancelPhone: extract phone number when intent is cancel_order
- For product_inquiry and other intents, orderData and cancelPhone can be null"""


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

    # Detect from latest customer message
    language = detect_language(last_customer_msg, None)

    # Language locking logic — word_count always defined first
    word_count = len(last_customer_msg.strip().split())
    word_count = len(last_customer_msg.strip().split())
    if request.detectedLanguage:
        if word_count <= 3 or language == request.detectedLanguage:
            language = request.detectedLanguage
        elif word_count < 5:
            language = request.detectedLanguage

    prior_turns = [m for m in history[:-1] if m.role in ("customer", "agent")]
    is_first_turn = len(prior_turns) == 0

    system_prompt = build_system_prompt(
        store_name=request.storeName,
        ai_system_prompt=request.aiSystemPrompt,
        products=request.products,
        recent_orders=request.recentOrders,
        language=language,
        is_first_turn=is_first_turn,
        ai_flow_state=request.aiFlowState,
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