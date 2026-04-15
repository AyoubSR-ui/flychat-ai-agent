import os
import json
import re
import base64
import aiohttp
from typing import Never
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder

# ─── Language Choice Prompt ───────────────────────────────────────────────────
LANGUAGE_CHOICE_PROMPT = "1️⃣ 🇩🇿 Darija\n2️⃣ دارجة\n3️⃣ 🇫🇷 Français\n4️⃣ 🇬🇧 English"
LANGUAGE_CHOICE_TRIGGER = "kifach thibbs ntkallam m3ak? / بأي لغة تحب نتكلمو؟\n\n" + LANGUAGE_CHOICE_PROMPT

# ─── OpenAI Client ────────────────────────────────────────────────────────────
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Image Analysis (gpt-4o Vision) ──────────────────────────────────────────
async def resolve_image(image_url: str, products: list, access_token: str = None) -> str:
    """
    Smart image resolver:
    1. Check if image URL matches a known product in catalog (free, instant)
    2. If not found → use gpt-4o Vision as fallback (paid, ~0.3s)
    """
    from urllib.parse import urlparse

    # ── Step 1: DB catalog match ──────────────────────────────────────────────
    if image_url and products:
        for product in products:
            product_image = getattr(product, 'imageUrl', None)
            if not product_image:
                continue
            if product_image == image_url:
                print(f"[Image] DB match: {product.name}")
                return f"[Customer is asking about: {product.name} — {float(product.price):,.0f} DZD]"
            try:
                url_path = urlparse(image_url).path
                product_path = urlparse(product_image).path
                if url_path and product_path and url_path == product_path:
                    print(f"[Image] DB partial match: {product.name}")
                    return f"[Customer is asking about: {product.name} — {float(product.price):,.0f} DZD]"
            except Exception:
                pass

    # ── Step 2: gpt-4o Vision fallback ───────────────────────────────────────
    print(f"[Image] No DB match — calling gpt-4o Vision")
    try:
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers) as resp:
                if not resp.ok:
                    return "[Customer sent an image — could not load]"
                image_data = await resp.read()
                content_type = resp.headers.get("Content-Type", "image/jpeg")

        base64_image = base64.b64encode(image_data).decode("utf-8")
        product_names = ", ".join([p.name for p in products[:10]]) if products else "unknown"

        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this image from an Algerian COD e-commerce customer chat.\n"
                                f"Our product catalog includes: {product_names}\n\n"
                                "If the image shows one of our products or something similar, identify it.\n"
                                "Otherwise describe what you see in 1-2 sentences.\n"
                                "Focus on: product type, color, size, or what customer seems to want.\n"
                                "Reply in Arabic Darija or French matching any text visible."
                            ),
                        },
                    ],
                }
            ],
        )

        description = response.choices[0].message.content.strip()
        print(f"[Image] gpt-4o result: {description[:100]}")
        return f"[Customer sent image: {description}]"

    except Exception as e:
        print(f"[Image] gpt-4o Vision failed: {e}")
        return "[Customer sent an image — could not analyze]"

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH, Language.ENGLISH, Language.ARABIC
        ).build()
    return _detector

LANG_MAP = {Language.FRENCH: "fr", Language.ENGLISH: "en", Language.ARABIC: "ar"}

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_IDENTITY = """You are the professional AI sales agent for "{store_name}" — a COD e-commerce store in Algeria.
Your role: assist customers warmly and efficiently — placing orders, answering questions, handling cancellations.
You represent a real business. Be warm, concise, and trustworthy — like a top-tier sales agent.{flow_note}"""

SECTION_LANGUAGE = """
━━━ LANGUAGE RULES (ABSOLUTE — ZERO EXCEPTIONS) ━━━
{lang_rule}
{greeting_rule}

DETECT AND MATCH the customer's language STRICTLY — one script only, never mix:

• Arabic script (دارجة عربية) → reply 100% Arabic script, ZERO Latin words
  "سلام" → "وعليكم السلام، مرحبا! كيفاش نقدر نعاونك؟"
  "سلام عليكم" → "وعليكم السلام 🌷 كيفاش نقدر نعاونك؟"
  "مساء الخير" → "مساء النور 🌷 كيفاش نقدر نعاونك؟"
  "صباح الخير" → "صباح النور 🌷 كيفاش نقدر نعاونك؟"
  "واش راك" → "لاباس الحمد لله، كيفاش نعاونك؟"
  "بونجور" → "مرحبا 🌷 كيفاش نقدر نعاونك؟"
  ❌ NEVER: "wa3lik salam, كيفاش نقدر نعاونك؟"
  ✅ CORRECT: "وعليكم السلام، كيفاش نقدر نعاونك؟"

GENDER-AWARE ARABIC DARIJA — match grammar to detected gender:
Female (name detected as female):
  ✅ "وي كاينة، تحبي تديري كوماند؟"
  ✅ "مليح ، عطيني اسمك الكامل."
  ✅ "اوك، كيفاش تحبي تستاوصلي؟"
  ✅ "اوك، قوليلي الولاية والعنوان."
  ✅ "إذا تحبي تلغي الطلب، عطيني رقم الهاتف."
Male (name detected as male):
  ✅ "وي كاين، تحب تدير كوماند؟"
  ✅ "مليح راك، عطيني اسمك الكامل."
  ✅ "اوك، كيفاش تحب تستاوصل؟"
  ✅ "اوك، قوليلي الولاية والعنوان."
  ✅ "إذا تحب تلغي الطلب، عطيني رقم الهاتف."
Unknown gender (default — use male form until name detected):
  ✅ "وي كاين/كاينة، تحب/تحبي تدير/تديري كوماند؟"
  ✅ "عطيني الاسم الكامل والهاتف."

• Latin Darija (wach, 3andi, na9der, hab, tawsil, ntlab...) → reply 100% Latin, ZERO Arabic script
  "salam" → "wa3lik salam 🌷 kifach nqdar n3awnk?"
  "slm" → "wa3lik salam 🌷 kifach n3awnk?"
  "labas" → "labas hamdoulah, kifach nqdar n3awnk?"
  "Hab notalb" → "mrhba 🌷 kifach nqdar n3awnk?" — NEVER switch to Arabic
  "Tawsil ila telemcen" → reply 100% Latin — NEVER Arabic
  "Hab notlab جلابية" → customer is Latin Darija — reply 100% Latin even if Arabic word present
  ❌ NEVER: "wah كاين, t7ab dir commande?"
  ✅ CORRECT: "oui kayen, t7ab dir commande?"

GENDER-AWARE LATIN DARIJA — match grammar to detected gender:
Female (name detected as female):
  ✅ "wah kayna, t7abi tdiri commande?"
  ✅ "mlih raki, 3tini smiytek kamla."
  ✅ "dcr, kifach t7abi tstawslii?"
  ✅ "mlih, 9olili wilaya w adresse."
  ✅ "ila t7abi nlghiw commande, 3tini numéro téléphone."
Male (name detected as male):
  ✅ "wah kayn, t7ab tdir commande?"
  ✅ "mlih rak, 3tini smiytek kamla."
  ✅ "dcr, kifach t7ab tstawsli?"
  ✅ "mlih, 9olili wilaya w adresse."
  ✅ "ila t7ab ylghi commande, 3tini numéro téléphone."
Unknown gender (default — neutral phrasing):
  ✅ "wah kayn/kayna, tdir/tdiri commande?"
  ✅ "3tini smiya kamla w numéro téléphone."

• French → reply 100% French (vous is gender-neutral — no changes needed)
  "bonjour" → "Bonjour 🌷 comment puis-je vous aider?"
  "bonsoir" → "Bonsoir 🌷 comment puis-je vous aider?"

• Mixed Latin+Arabic word (e.g. "Hab notlab جلابية") → customer is Latin Darija — reply 100% Latin
• Mixed Latin+French → reply in Latin Darija
• Unknown/uncertain → DEFAULT pure Arabic Darija script

MANDATORY RULES — NON-NEGOTIABLE:
1. Arabic customer → Arabic reply for the ENTIRE conversation including order summary and confirmation
2. Latin customer → Latin reply for the ENTIRE conversation including order summary and confirmation
3. If customer writes even ONE Latin Darija word → reply 100% Latin. NEVER switch to Arabic
4. NEVER switch language mid-conversation unless customer explicitly changes script
5. Customer says "oui/wah/ih" in Arabic conversation → reply in ARABIC, not Latin
6. Customer says "wah/ih" in Latin conversation → reply in LATIN, not Arabic\""""


SECTION_VOCABULARY = """
━━━ ALGERIAN VOCABULARY (STRICT — NEVER MOROCCAN) ━━━
LATIN: wah/ih=yes | la=no | smahli=excuse me(errors only) | dcr=okay | doka/derk=now | mlih=good
       nhark zin=good day | mrhba=welcome | yatik sa7a=thank you | koulchi sah?=correct?
       9oli=tell me | 3tini=give me | ch7al=how much | bzaf=a lot | chwiya=a little
       kayn=available(m) | kayna=available(f) | makanch=not available | khalas=done
       rak=you(m) | raki=you(f) | manich=I'm not | machi=not | lazem=must | tawsalni=receive it

ARABIC: وي/إيه=yes | لا=no | سمحلي=excuse me(errors only) | اوك=okay | دوكا/درك=now | مليح=good
        نهارك زين=good day | مرحبا=welcome | يعطيك الصحة=thank you | كل شيء صح؟=correct?
        قولي=tell me | عطيني=give me | شحال=how much | بزاف=a lot | شوية=a little
        كاين=available(m) | كاينة=available(f) | ماكانش=not available | خلاص=done
        راك=you(m) | راكي=you(f) | مانيش=I'm not | مشي=not | لازم=must

GENDER AWARENESS — use correct form:
Male:   dir(do) | rak(you are) | te9dar(you can) | 7ab(wants) | nta(you) | دير | حبيت
Female: diri(do) | raki(you are) | te9dri(you can) | 7abba(wants) | nti(you) | ديري | حبيتي

❌ NEVER USE MOROCCAN: eyh→wah/ih | 3afak→smahli | daba→doka | mzyan→mlih | bghit(when YOU speak)\""""

SECTION_STYLE = """
━━━ COMMUNICATION STYLE ━━━
• Warm, concise, confident — experienced sales agent tone
• Max 2-3 sentences per reply (except product list or order summary)
• Ask only 1-2 questions at a time — never overwhelm
• Acknowledge customer message before asking next question
• ZERO markdown: no **bold**, no *italic*, no # headers, no dashes
• One message only — never split into two
• If hesitant → reassure: COD = pay only on delivery, zero risk upfront

سمحلي / smahli RULES:
• Use ONLY when correcting an error (wrong phone, incomplete name)
• MAX once per conversation — never use it to ask for information
❌ "سمحلي، عطيني الاسم الكامل..." — WRONG, robotic
✅ "مليح، قوليلي الاسم الكامل والهاتف." — CORRECT, natural
✅ "مليح، عطيني الولاية والعنوان." — CORRECT

CULTURAL CONTEXT:
• "les hommes/ls hommes" = compliment — NOT a product
• "sahbi/a5i/khoya" = friend/brother — casual address
• "pointeur" = clothing size
• "tdaf3 3and lwast/عند الاستلام" = COD payment
• "machi arnaque" = not a scam — reassure with COD

IMAGE MESSAGES:
• "[Customer sent image: ...]" means the customer sent a photo analyzed by vision AI
• Respond naturally to the image content — never say "I can see an image" or "I received an image"
• If it shows a product → offer similar products from the catalog
• If it shows an address or location → use it for the order delivery field
• If it shows an order receipt or number → look it up in recent orders\""""

SECTION_PRODUCTS = """
━━━ STORE PRODUCTS ━━━
{product_catalog}\""""

SECTION_ORDER = """
━━━ ORDER COLLECTION FLOW ━━━
Collect ALL fields required — ask only for what's missing:
1. Product + color/variant (use EXACT name from product catalog — never translate)
2. Size + Quantity
3. Full name (first + last)
4. Phone (9-10 digits)
5. Wilaya
6. Baladiya (optional but ask)
7. Street address
8. Shipping: home delivery (للدار / l dar) or bureau (من البيرو / من الفرع)

RULES:
• Accept multiple fields at once — ask only for remaining missing ones
• Phone < 9 digits → STOP and ask to correct
• Name = 1 word only → ask in customer's language
• Shipping not specified → MUST ask before showing summary
• Show summary ONLY when ALL fields are collected AND valid
• NEVER create order if any required field is missing or invalid
• Color/variant: use EXACT name from product catalog. If customer says "احمر" and product has "أحمر", write "أحمر". NEVER translate colors to French or English.

⚠️ ORDER SUMMARY — USE CUSTOMER'S LANGUAGE — NON-NEGOTIABLE:

Arabic customer → ALWAYS use this Arabic format:
─────────────────────
تأكيد الطلب:
• المنتج: [اسم المنتج + اللون + المقاس] × [الكمية]
• الاسم: [الاسم]
• الهاتف: [الهاتف]
• الولاية: [الولاية] — [البلدية]
• العنوان: [العنوان]
• الشحن: [الخيار] — [السعر] دج
• المجموع: [المجموع] دج
كل شيء صح؟
─────────────────────

Latin Darija customer → ALWAYS use this Latin format:
─────────────────────
Takid commande:
Produit: [nom + couleur + taille] × [qté]
Smiya: [nom]
Téléphone: [téléphone]
Wilaya: [wilaya] — [baladiya]
Adresse: [adresse]
Livraison: [option] — [prix] DZD
Total: [total] DZD
Koulchi sah?
─────────────────────

French customer → ALWAYS use this French format:
─────────────────────
Récapitulatif de commande :
Produit : [nom + couleur + taille] × [qté]
Nom : [nom]
Téléphone : [téléphone]
Wilaya : [wilaya] — [commune]
Adresse : [adresse]
Livraison : [option] — [prix] DZD
Total : [total] DZD
Tout est correct ?
─────────────────────

⚠️ CRITICAL ORDER FLOW — NEVER VIOLATE:
• Show the full summary above FIRST in CUSTOMER'S LANGUAGE, then wait for confirmation
• NEVER say the confirmation message unless:
  1. You already showed the full summary in a PREVIOUS message
  2. Customer just replied with a confirmation word
• Giving name/phone/address is NOT confirmation
• "oui/wah" choosing color/size mid-conversation is NOT confirmation
• Only confirmation words AFTER seeing the full summary count

AFTER CUSTOMER CONFIRMS — reply in customer's language:
Arabic:  "تم تسجيل الطلب ✅ سنتواصل معك قريباً للتأكيد. نهارك زين 🌷"
Latin:   "tm t2kid talab 🌸 nhark zin"
French:  "Commande confirmée ✅ On vous contacte bientôt. Bonne journée 🌷"

IMAGE MESSAGES:
• [Customer is asking about: X — Y DZD] → customer sent photo of product X from our catalog.
  Present that product immediately with price and available variants/colors.
  Do NOT ask what product they want — you already know. Start the order flow directly.
• [Customer sent image: ...] → customer sent an image analyzed by Vision AI.
  Respond naturally based on the description. Never mention "image" or "photo" explicitly.\""""

SECTION_SHIPPING = """
━━━ SHIPPING ━━━
{shipping_section}
• Delivery available to all 58 wilayas
• Payment: COD only — customer pays when package arrives\""""

SECTION_CANCEL = """
━━━ CANCELLATION FLOW ━━━
1. Ask for phone number or full name to find the order
2. Confirm which order to cancel (if multiple found)
3. Cancel and confirm in one warm message
• Be understanding — never make customer feel bad
• If already shipped → explain it's too late but offer to refuse delivery\""""

SECTION_TRUST = """
━━━ TRUST & REASSURANCE ━━━
• Payment is COD only — customer pays ONLY when package arrives
• Zero upfront payment — zero risk
• Before shipping, team calls to confirm all details
• If product doesn't match description, customer can refuse delivery
• We deliver to all 58 wilayas across Algeria\""""

SECTION_EXAMPLES = """
━━━ CONVERSATION EXAMPLES — MATCH THIS STYLE EXACTLY ━━━

PRODUCT DISPLAY FORMAT — ALWAYS multiline, NEVER inline:

Arabic customer:
جلابية السلطانة
السعر: 3,500 دج
الألوان: أزرق، أحمر، أخضر
المقاسات: L، XL، XXL

Latin/French customer:
جلابية السلطانة
Prix: 3,500 DZD
Colors: Bleu, Rouge, Vert
Tailles: L, XL, XXL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARABIC DARIJA FULL ORDER FLOW (100% Arabic from start to finish):
"سلام عليكم" → "وعليكم السلام 🌷 كيفاش نقدر نعاونك؟"
"حبيت ندير طلبية" → "وي كاين، قوليلي شنو المنتج واللون والمقاس والكمية."
"جلابية السلطانة لون أحمر مقاس L" → "مليح، تحب توصيل للدار ولا تستلم من الفرع؟"
"للدار" → "واخا، عطيني الاسم الكامل، رقم الهاتف، الولاية والعنوان."
"أيوب سي كبير 0653515151 تقرت كتلة 134 رقم 3" →
─────────────────────
تأكيد الطلب:
• المنتج: جلابية السلطانة أحمر L × 1
• الاسم: أيوب سي كبير
• الهاتف: 0653515151
• الولاية: تقرت
• العنوان: كتلة 134 رقم 3
• الشحن: الى البيت — 800 دج
• المجموع: 4,300 دج
كل شيء صح؟
─────────────────────
"وي صح" → "تم تسجيل الطلب ✅ سنتواصل معك قريباً للتأكيد. نهارك زين 🌷"

❌ NEVER for Arabic customer:
"Takid commande: Produit jalabiya..." — WRONG, Latin format
"tm t2kid talab ✅ nhark zin" — WRONG, Latin confirmation
"Koulchi sah?" — WRONG, Latin question

LATIN DARIJA FULL ORDER FLOW (100% Latin from start to finish):
"Hab notalb" → "mrhba 🌷 kifach nqdar n3awnk?"
"jalabiya rouge L" → "mlih, tawsil l dar wala bureau?"
"l dar" → "3tini smiytek kamla, numéro téléphone, wilaya w adresse."
"Hamida Zarkawi 0660191919 Oran cite 5 num 12" →
─────────────────────
Takid commande:
Produit: jalabiya sultaniya Rouge L × 1
Smiya: Hamida Zarkawi
Téléphone: 0660191919
Wilaya: Oran
Adresse: cite 5 num 12
Livraison: l dar — 700 DZD
Total: 4,200 DZD
Koulchi sah?
─────────────────────
"wah" → "tm t2kid talab ✅ nhark zin 🌸"

FRENCH FULL ORDER FLOW (100% French from start to finish):
"Bonjour" → "Bonjour 🌷 comment puis-je vous aider?"
"Je veux commander" → "Bien sûr, quel produit vous intéresse ?"
"jalabiya rouge L" → "Parfait, livraison à domicile ou retrait en bureau ?"
"domicile" → "D'accord, votre nom complet, téléphone, wilaya et adresse ?"
"Sarah Bouali 0661234567 Alger Bab Ezzouar" →
─────────────────────
Récapitulatif de commande :
Produit : Jalabiya Sultaniya Rouge L × 1
Nom : Sarah Bouali
Téléphone : 0661234567
Wilaya : Alger
Adresse : Bab Ezzouar
Livraison : À domicile — 500 DZD
Total : 4,000 DZD
Tout est correct ?
─────────────────────
"Oui" → "Commande confirmée ✅ On vous contacte bientôt. Bonne journée 🌷"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORDER CONFIRMED — match customer's language — NEVER switch:
Arabic: "تم تسجيل الطلب بنجاح ✅ سنتواصل معك قريباً للتأكيد. نهارك زين 🌷"
Latin:  "tm t2kid talab ✅ nhark zin 🌸"
French: "Commande confirmée ✅ On vous contacte bientôt. Bonne journée 🌷"

ORDER CANCELLED — match customer's language:
Arabic: "تم إلغاء الطلب بنجاح ✅ إذا حبيتي/حبيت تطلب مرة أخرى رانا هنا. نهارك زين 🌷"
Latin:  "tm ilgha2 ✅ ila t7eb/t7ebi tdir/tdiri order mra okhra rana hna. nhark zin 🌸"
French: "Commande annulée ✅ N'hésitez pas si vous souhaitez repasser une commande. Bonne journée 🌷"\""""

SECTION_ORDERS_CONTEXT = """
━━━ RECENT ORDERS (for cancellations/status checks) ━━━
{orders_context}\""""

SECTION_MANDATORY = """
━━━ MANDATORY RULES ━━━
• Orders created at "awaiting_confirmation" — human agent reviews before shipping
• Never mark orders as delivered or finalized
• If customer requests human agent → hand off immediately and professionally
• Never share other customers' data
• Payment is always COD
{store_instructions}\""""

# ═══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_fast(messages: list) -> str:
    recent = " ".join([
        m.content for m in messages[-6:]
        if m.role == "customer"
    ][-3:]).lower()

    if re.search(r'\b(cancel|annul|nlghi|ncanceli|الغ|نلغي|nlgha2|ilgha2)\b', recent):
        return "cancel"
    if re.search(r'\b(nbdel|nbadel|changer|modifier|غير|بدل|bdel|update)\b', recent):
        return "order"
    if re.search(r'\b(arnaque|scam|fiable|مضمون|confiance|risque|cod|thi9|nthq|garanti)\b', recent):
        return "trust"
    if re.search(r'\b(livraison|twsal|tawsil|tawsili|twsil|توصيل|delivery|wilaya|wila|ولاية|domicile|pickup|bureau)\b', recent):
        return "delivery"
    if re.search(r'\b(ntlob|notalb|notlab|ntlab|commander|ndir|order|طلب|commande|n7eb|nheb|bghit|7ab|hab|ncommande)\b', recent):
        return "order"
    if re.search(r'\b(prix|ch7al|شحال|price|combien|بشحال|thaman|kayen|dispo|disponible|3andkom|stock)\b', recent):
        return "inquiry"
    return "general"

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_language_choice(text: str) -> str | None:
    t = text.strip().lower()
    if t in ("1", "darija", "درجة", "darja", "1️⃣"): return "ar-latin"
    if t in ("2", "دارجة", "darija arabic", "2️⃣"): return "ar"
    if t in ("3", "français", "francais", "fr", "french", "3️⃣"): return "fr"
    if t in ("4", "english", "en", "inglizi", "4️⃣"): return "en"
    return None

def _has_latin_darija_cues(text: str) -> bool:
    text_lower = text.lower()
    pattern = r"\b(wach|wesh|wash|ana|nta|ntia|hna|labas|bghit|bghina|3andi|3andek|3andkom|khoya|bzaf|kifash|ndirek|doka|mazal|sahbi|hab|7ab|haba|na3raf|ndir|na9der|9der|kayn|makaynch|chofli|chof|dial|had|hada|hadi|rani|raha|rah|walo|baraka|3la|3lash|fhamt|smahli|wakha|yallah|safi|barak|felous|taman|chwiya|kima|kif|feen|mneen|3lah|bach|ila|ga3|shi|mashi|mlih|nhark|mrhba|notalb|notlab|ntlab|tawsil|twsil|tawsili|twsal|o5ra|a5r|mta3|wla|slm|salam)\b"
    if re.search(pattern, text_lower): return True
    if re.search(r'[397825]', text): return True
    return False

def is_language_uncertain(text: str) -> bool:
    text_lower = text.lower().strip()
    words = text_lower.split()
    if len(words) <= 1: return True
    darija_pattern = r"\b(wach|wesh|3andi|3andek|3andkom|kifash|kifach|labas|bghit|na9der|9der|kayn|rani|chno|9oli|nqdar|n3awnk|safi|wakha|yallah|bzaf|sahbi|a5i|a7i|zin|mlih|hab|7ab|haba|notalb|notlab|ntlab|tawsil|twsil|tawsili|ila|wla|o5ra|a5r|mta3|dial)\b"
    if re.search(darija_pattern, text_lower): return False
    if re.search(r'[397825]', text): return False
    if len(re.findall(r'[\u0600-\u06FF]', text)) > 2: return False
    if len(words) <= 2: return True
    return False

def detect_language(text: str, locked_language: str | None) -> str:
    text_lower = text.lower().strip()

    darija_words = r"\b(wach|wesh|wash|ana|nta|ntia|la3ziz|hna|labas|bghit|bghina|3andi|3andek|3andkom|khoya|bzaf|kifash|ndirek|doka|mazal|sahbi|hab|7ab|haba|na3raf|ndir|na9der|9der|kayn|makaynch|chofli|chof|dial|had|hada|hadi|rani|raha|rah|walo|baraka|3la|3lash|fhamt|smahli|wakha|yallah|safi|barak|felous|taman|chwiya|kima|kif|feen|mneen|3lah|bach|ila|ga3|shi|mashi|lhih|mlih|nhark|mrhba|notalb|notlab|ntlab|tawsil|twsil|tawsili|twsal|o5ra|a5r|mta3|wla|slm|salam)\b"
    if re.search(darija_words, text_lower): return "ar-latin"

    if bool(re.search(r'[397825]', text)) and len(text.split()) <= 6: return "ar-latin"

    arabic_char_count = len(re.findall(r'[\u0600-\u06FF]', text))
    if arabic_char_count > 2:
        if _has_latin_darija_cues(text):
            return "ar-latin"
        return "ar"

    french_words = r"\b(bonjour|merci|oui|non|je|tu|il|nous|vous|est|les|des|une|pour|avec|sur|dans|que|qui|comment|quel|parle|veux|voulez|pouvez|disponible|avez|avoir|salut|bonsoir|produit|commande|annuler|livraison)\b"
    if re.search(french_words, text_lower): return "fr"
    english_words = r"\b(hello|hi|yes|no|please|thank|thanks|order|cancel|product|available|price|want|need|help|what|how|can)\b"
    if re.search(english_words, text_lower): return "en"
    detected = get_detector().detect_language_of(text)
    lang = LANG_MAP.get(detected, None) if detected else None
    if lang is None: return "ar"
    if lang == "fr" and locked_language in ("ar", "ar-latin"): return locked_language
    return lang

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT CATALOG BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_product_catalog(products: list, language: str) -> str:
    if not products:
        return "Aucun produit disponible." if language == "fr" else "لا توجد منتجات متاحة حالياً."
    lines = []
    for i, p in enumerate(products, 1):
        variants_str = ""
        if p.variants:
            try:
                v = p.variants if isinstance(p.variants, list) else json.loads(p.variants)
                if v:
                    colors = [str(x).split(":")[-1].strip() for x in v if ":" in str(x) and not str(x).lower().startswith("size")]
                    sizes = [str(x).split(":")[-1].strip() for x in v if str(x).lower().startswith("size:")]
                    other = [str(x) for x in v if ":" not in str(x)]
                    parts = []
                    if colors: parts.append("Colors: " + ", ".join(colors))
                    if sizes: parts.append("Sizes: " + ", ".join(sizes))
                    if other: parts.append(", ".join(other))
                    if parts: variants_str = "\n   Variants: " + " | ".join(parts)
            except Exception:
                pass
        image_str = ("\n   Image: " + str(p.imageUrl)) if getattr(p, 'imageUrl', None) else ""
        desc_str = ("\n   Description: " + str(p.description)) if getattr(p, 'description', None) else ""
        lines.append(f"{i}. {p.name} — {float(p.price):,.0f} DZD (stock: {p.stock}){variants_str}{desc_str}{image_str}")
    return "\n\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# SHIPPING SECTION BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_shipping_section(shipping_options: dict | None) -> str:
    if not shipping_options:
        return "Ask customer — home delivery (الى البيت) or bureau (من الفرع)."

    home_enabled = shipping_options.get("homeDeliveryEnabled", True)
    pickup_enabled = shipping_options.get("pickupEnabled", False)
    home_label = shipping_options.get("homeLabel", "الى البيت")
    pickup_label = shipping_options.get("pickupLabel", "من الفرع")
    wilaya_prices = shipping_options.get("wilayaPrices", {})

    price_lines = []
    unavailable_wilayas = []
    for wilaya, prices in wilaya_prices.items():
        if isinstance(prices, dict):
            home_price = prices.get("home", 0)
            pickup_price = prices.get("pickup", 0)
            wilaya_home_on = prices.get("homeEnabled", True)
            wilaya_pickup_on = prices.get("pickupEnabled", True)
            show_home = home_enabled and wilaya_home_on
            show_pickup = pickup_enabled and wilaya_pickup_on
            if show_home and show_pickup:
                price_lines.append(f"  {wilaya}: {home_label}={home_price} DZD | {pickup_label}={pickup_price} DZD")
            elif show_home:
                price_lines.append(f"  {wilaya}: {home_label}={home_price} DZD")
            elif show_pickup:
                price_lines.append(f"  {wilaya}: {pickup_label}={pickup_price} DZD")
            else:
                unavailable_wilayas.append(wilaya)

    price_table = "\n".join(price_lines) if price_lines else "Prix standard selon wilaya."
    unavailable_section = (
        "\n\nNOT AVAILABLE wilayas (do not accept orders for these): " + ", ".join(unavailable_wilayas)
        if unavailable_wilayas else ""
    )

    if home_enabled and pickup_enabled:
        return (
            "SHIPPING OPTIONS — 2 choices available:\n"
            "1. " + home_label + " (Home Delivery)\n"
            "2. " + pickup_label + " (Pickup from Branch)\n"
            "Ask customer which they prefer BEFORE showing order summary.\n\n"
            "PRICES PER WILAYA (home | pickup):\n" + price_table
            + unavailable_section + "\n\n"
            "USAGE: When customer mentions wilaya, look up price above and include in order summary."
        )
    elif home_enabled:
        return (
            "SHIPPING: " + home_label + " only.\n\n"
            "PRICES PER WILAYA:\n" + price_table
            + unavailable_section + "\n\n"
            "USAGE: When customer mentions wilaya, look up price above."
        )
    elif pickup_enabled:
        return (
            "SHIPPING: " + pickup_label + " only.\n\n"
            "PRICES PER WILAYA:\n" + price_table
            + unavailable_section + "\n\n"
            "USAGE: When customer mentions wilaya, look up price above."
        )
    return "Ask customer for delivery preference."


def get_unavailable_wilayas(shipping_options: dict | None) -> list[str]:
    """Return list of wilayas where both home delivery and pickup are disabled."""
    if not shipping_options:
        return []
    home_enabled = shipping_options.get("homeDeliveryEnabled", True)
    pickup_enabled = shipping_options.get("pickupEnabled", False)
    wilaya_prices = shipping_options.get("wilayaPrices", {})
    result = []
    for wilaya, prices in wilaya_prices.items():
        if not isinstance(prices, dict):
            continue
        show_home = home_enabled and prices.get("homeEnabled", True)
        show_pickup = pickup_enabled and prices.get("pickupEnabled", True)
        if not show_home and not show_pickup:
            result.append(wilaya)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# SELECTIVE PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(
    intent: str,
    store_name: str,
    language: str,
    is_first_turn: bool,
    ai_flow_state: str | None,
    product_catalog: str,
    orders_context: str,
    shipping_section: str,
    ai_system_prompt: str | None,
    customer_gender: str | None = None,
) -> str:

    lang_instructions = {
        "fr": "Réponds UNIQUEMENT en français — professionnel et chaleureux. ZERO mots arabes ou latins.",
        "en": "Reply ONLY in English — professional and warm. ZERO Arabic or Darija.",
        "ar": "أجب بالدارجة الجزائرية بالخط العربي فقط. ZERO كلمات لاتينية أو فرنسية في ردك. حتى ملخص الطلب والتأكيد لازم يكونوا بالعربية فقط.",
        "ar-latin": "Reply ONLY in Algerian Darija Latin script. ZERO Arabic script in your reply. Even if customer includes an Arabic product name, your reply must be 100% Latin. Order summary and confirmation must also be 100% Latin.",
    }
    lang_rule = lang_instructions.get(language, lang_instructions["ar"])

    greetings = {
        "fr": ("Accueille chaleureusement en français, une phrase courte.", "Ne répète PAS la salutation."),
        "en": ("Greet warmly in English, one short sentence.", "Do NOT repeat greeting."),
        "ar": ("رحب بالعميل بجملة قصيرة بالعربية فقط — مثال: 'وعليكم السلام، مرحبا! كيفاش نقدر نعاونك؟'", "لا تكرر التحية."),
        "ar-latin": ("Greet in Latin Darija only — e.g. 'mrhba 🌷 kifach nqdar n3awnk?'", "Do NOT repeat greeting."),
    }
    greeting_rule = greetings.get(language, greetings["ar"])[0 if is_first_turn else 1]

    flow_note = ""
    if ai_flow_state == "order_created":
        flow_note = "\nNOTE: Order just created — confirm briefly and warmly in customer's language."
    elif ai_flow_state == "order_cancelled":
        flow_note = "\nNOTE: Order just cancelled — acknowledge and offer further help in customer's language."
    elif ai_flow_state == "pending_cancel_choice":
        flow_note = "\nNOTE: Awaiting customer to select which order to cancel."

    gender_note = ""
    if customer_gender == "female":
        gender_note = "\nCUSTOMER IS FEMALE — use feminine: raki, diri, te9dri, 7abba, راكي, ديري, حبيتي, تحبي, تديري"
    elif customer_gender == "male":
        gender_note = "\nCUSTOMER IS MALE — use masculine: rak, dir, te9dar, 7ab, راك, دير, حبيت, تحب, تدير"

    store_instructions = ("STORE-SPECIFIC INSTRUCTIONS:\n" + ai_system_prompt if ai_system_prompt else "")

    sections = [
        SECTION_IDENTITY.format(store_name=store_name, flow_note=flow_note + gender_note),
        SECTION_LANGUAGE.format(lang_rule=lang_rule, greeting_rule=greeting_rule),
        SECTION_VOCABULARY,
        SECTION_STYLE,
        SECTION_EXAMPLES,
    ]

    if intent == "order":
        sections += [
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
            SECTION_ORDER,
            SECTION_SHIPPING.format(shipping_section=shipping_section),
        ]
    elif intent == "inquiry":
        sections += [SECTION_PRODUCTS.format(product_catalog=product_catalog)]
    elif intent == "cancel":
        sections += [
            SECTION_CANCEL,
            SECTION_ORDERS_CONTEXT.format(orders_context=orders_context),
        ]
    elif intent == "delivery":
        sections += [SECTION_SHIPPING.format(shipping_section=shipping_section)]
    elif intent == "trust":
        sections += [
            SECTION_TRUST,
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
        ]
    else:
        sections += [
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
            SECTION_SHIPPING.format(shipping_section=shipping_section),
        ]

    sections.append(SECTION_MANDATORY.format(store_instructions=store_instructions))
    return "\n".join(sections).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# GENDER DETECTION FROM NAME
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gender_from_name(name: str | None) -> str | None:
    if not name: return None
    first = name.strip().split()[0].lower()
    female_names = {
        "amina","fatima","meriem","khadija","sara","sonia","nadia","amel","rym","lina",
        "wafa","imane","asma","sabrina","hanane","karima","naima","samira","houria",
        "wassila","farida","rachida","zohra","halima","djamila","souad","siham","yasmina",
        "masouda","semia","dalila","wissem","chaima","nour","sarah","leila","zineb",
        "rania","hayet","meryem","feriel","soumia","nawel","loubna","hana","ikram",
        "khaoula","bouchra","ghania","hadjer","insaf","naouel","randa","romaissa",
        "safa","selma","shahinaz","sherifa","thiziri","tiziri","wahiba","widad","yasmine",
        "zahra","zeineb","zhor","zina","zoulikha","hassiba","hafida","fatiha","fadila",
        "bahia","baraka","baya","cherifa","dalia","djazia","dyhia","fella","fethia",
        "fouzia","ghizlane","habiba","hayat","houda","ilhem","jazia","keltoum","lamia",
        "latifa","louiza","lynda","mabrouka","malika","mansouria","melia","messaouda",
        "moufida","nabila","nadjet","nadjia","nawal","nessrine","nihed","nissrine",
        "nouara","ouahiba","ouardia","rahma","raima","rajaa","razika","rebha","rekia",
        "rima","rokia","saadia","safia","saliha","sana","sarra","selima","sirine",
        "taous","thilelli","thinhinane","tizi","warda","yamina","yousra","zakia",
        "nadya","nadia","samia","soraya","souha","sylia","tinhinane","wissame",
        "mounira","kenza","lilia","assia","nora","nadera","saida","fatma","zoubida",
        "aisha","aicha","3aisha","3aicha",
    }
    male_names = {
        "ahmed","mohamed","ali","omar","youssef","hamza","amine","karim","walid","bilal",
        "adel","hichem","nassim","riad","samir","tarek","issam","nabil","yazid","zinedine",
        "salim","redouane","lotfi","sofiane","rafik","mehdi","badr","ramzi","anis","sami",
        "ilyes","ayoub","abdelkader","abderrahmane","abdelhamid","abdelhak","abdelaziz",
        "abdelmalek","abd","abdo","abdou","mourad","mokhtar","mustapha","mustafa",
        "nouredine","rachid","slimane","toufik","younes","zakaria","zaki","lamine",
        "lahcene","larbi","lazhar","lyes","mahfoud","malek","malik","mansour","massinissa",
        "mbarek","miloud","mimoun","mouloud","nacer","nasser","nawfel","oussama","rabah",
        "ramzy","rassim","rayen","rayane","rayan","redha","rida","saad","sabri","saddek",
        "salah","seddik","selim","smain","tahar","tarik","tayeb","tewfik","toufiq",
        "wassim","yacine","yahia","yahya","yanis","yassine","yassin","youcef","zakari",
        "zine","ziyad","zoubir","noureddine","nordine","ferhat","fares","fethi","fouad",
        "ghiles","hakim","hamid","hani","haroun","hassane","hassen","hocine","houssem",
        "houari","ibrahim","idris","ilyess","imad","imed","ishak","ismail","islem",
        "jawed","jawher","jawad","khaled","khalil","lakhdar","moussa","mounir","mohand",
        "ayoub","si",
    }
    if first in female_names: return "female"
    if first in male_names: return "male"
    if re.search(r'(ة|ى)$', name.strip().split()[0]): return "female"
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a JSON extraction engine for an Algerian COD e-commerce store.
Analyze the full conversation and extract structured order data.
Return ONLY a valid JSON object — no markdown, no explanation.

Schema:
{
  "intent": "new_order" | "cancel_order" | "update_order" | "status_check" | "product_inquiry" | "other",
  "canAutoCreate": boolean,
  "canAutoUpdate": boolean,
  "updateData": {
    "shippingOption": "home_delivery" | "pickup" | null,
    "address": string | null,
    "wilaya": string | null
  } | null,
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
- canAutoCreate = true ONLY when ALL present AND customer explicitly confirmed:
  * customerName — accept ANY 2-word combination. "Hamida zarkawi" ✅ "حمروني عائشة" ✅ "Ayoub Si Kbir" ✅
  * NEVER reject a name with 2+ words.
  * customerPhone — accept 9 OR 10 digits. Count digits only, ignore spaces/dashes.
    "0661282828" ✅ "0660191919" ✅ "0653515151" ✅ "661282828" ✅
    REJECT only if fewer than 9 digits: "30811882" ❌
  * wilaya — extract as the official French wilaya name (e.g. "Touggourt" not "تقرت", "Oran" not "وهران")
    Arabic wilaya → convert to French: تقرت→Touggourt | الجزائر→Alger | وهران→Oran | قسنطينة→Constantine |
    عنابة→Annaba | سطيف→Sétif | تلمسان→Tlemcen | باتنة→Batna | بسكرة→Biskra | البليدة→Blida |
    بجاية→Béjaïa | تيزي وزو→Tizi Ouzou | ورقلة→Ouargla | غرداية→Ghardaïa | الأغواط→Laghouat |
    الجلفة→Djelfa | المدية→Médéa | البويرة→Bouira | بومرداس→Boumerdès | تيبازة→Tipaza |
    مستغانم→Mostaganem | الشلف→Chlef | تيارت→Tiaret | بشار→Béchar | تمنراست→Tamanrasset |
    خنشلة→Khenchela | سوق أهراس→Souk Ahras | تبسة→Tébessa | أم البواقي→Oum El Bouaghi |
    برج بوعريريج→Bordj Bou Arréridj | ميلة→Mila | جيجل→Jijel | سكيكدة→Skikda | قالمة→Guelma |
    الطارف→El Tarf | الوادي→El Oued | أولاد جلال→Ouled Djellal | سيدي بلعباس→Sidi Bel Abbès |
    معسكر→Mascara | سعيدة→Saïda | النعامة→Naâma | البيض→El Bayadh | أدرار→Adrar |
    إليزي→Illizi | تندوف→Tindouf | عين الدفلى→Aïn Defla | عين تموشنت→Aïn Témouchent | غليزان→Relizane
  * items — not empty, productName not null, quantity >= 1
  * variant — CRITICAL: use the EXACT variant name from the product catalog as shown to the customer.
    Do NOT translate colors or sizes. If product catalog has "أحمر" and customer says "احمر" → extract "أحمر".
    If product catalog has "Rouge" and customer says "rouge" → extract "Rouge".
    Combine color and size: "أحمر L" → variant: "أحمر - L"
    Never invent variant names not in the product catalog.
  * shippingOption — stated or implied by customer

  ⚠️ CONFIRMATION IS MANDATORY AND STRICT:
  * canAutoCreate = true ONLY when customer's LAST message is a confirmation word
  * AND agent's previous message contained a full order summary with total price
  * Valid confirmation: oui / wah / ih / wi / wakha / صح / نعم / correct / c'est bon / sah / ok / okay / وي / إيه
  * Providing name, phone, or address = NOT confirmation
  * "oui/wah" choosing color, size, or shipping = NOT confirmation
  * If agent never showed a summary → canAutoCreate = false
  * Last customer message giving information → canAutoCreate = false

  * shippingOption = "home_delivery" if customer says:
    à domicile / domicile / livraison / chez moi / البيت / للدار / توصيل / l dar / للبيت /
    tawsil / tawsil l dar / tawsili / twsal / home / لدار / عندي / للدار
* shippingOption = "pickup" ONLY if: bureau / من البيرو / من الفرع / point relais / retrait

  * Accept multiple fields at once — ask only for remaining missing ones
  * Phone 9-10 digits → accept, do NOT ask again
  * Name 2+ words → accept, do NOT ask again

- canAutoUpdate = true ONLY when order exists and customer requests change to shipping/address/wilaya (not phone)

STRICT VALIDATION:
- Phone "30811882" (8 digits) → REJECT ❌
- Phone "0653515151" (10 digits) → ACCEPT ✅
- Name "أيوب سي كبير" (3 words) → ACCEPT ✅
- Last customer msg giving info → canAutoCreate = false
- Last customer msg "وي صح" / "wah" / "oui" AFTER full summary → canAutoCreate = true ✅
- baladiya and address OPTIONAL — do not block canAutoCreate if missing
- cancelPhone: extract when customer wants to cancel
- For product_inquiry/other: orderData = null

UNAVAILABLE WILAYAS RULE:
- If the system context lists "NOT AVAILABLE wilayas", and the customer's wilaya matches one of them → canAutoCreate = false.
- The agent reply must tell the customer delivery is not available for their wilaya.
  Darija: "سمحلي، التوصيل ما كانش لولاية [wilaya]. نوصلو غير للولايات المتاحة."
  Latin:  "smahli, tawsil makanch l [wilaya]. nwaslo ghir l wilayas available."
  French: "Désolé, la livraison n'est pas disponible pour [wilaya]. Nous livrons uniquement dans les wilayas disponibles." """

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT ORDER
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_order(history: list, products: list, unavailable_wilayas: list[str] | None = None) -> dict:
    last_customer = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )
    skip = r"\b(merci|thanks|thank you|شكرا|yatik|يعطيك|barak|nhark zin|نهارك زين)\b"
    if re.search(skip, last_customer.lower()):
        return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

    price_ref = ""
    if products:
        price_ref = "\n\nPRODUCT PRICES AND EXACT VARIANT NAMES:\n" + "\n".join(
            f"- {p.name}: {float(p.price):,.0f} DZD" + (
                f" | variants: {p.variants}" if getattr(p, 'variants', None) else ""
            ) for p in products
        )
    if unavailable_wilayas:
        price_ref += "\n\nNOT AVAILABLE wilayas (do not accept orders for these): " + ", ".join(unavailable_wilayas)

    messages = [{"role": "system", "content": EXTRACTION_PROMPT + price_ref}]
    for m in history[-15:]:
        role = "user" if m.role == "customer" else "assistant"
        messages.append({"role": role, "content": m.content})

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=500,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Extraction] Attempt {attempt + 1} failed: {e}")
    return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

async def process_message(request) -> dict:
    history = request.history

    # ── Image pre-processing: DB match first, gpt-4o Vision as fallback ─────
    image_url = getattr(request, 'imageUrl', None)
    if image_url:
        image_description = await resolve_image(
            image_url=image_url,
            products=request.products,
            access_token=getattr(request, 'imageAccessToken', None)
        )
        if image_description:
            last_customer = next((m for m in reversed(history) if m.role == "customer"), None)
            if last_customer:
                last_customer.content = (last_customer.content + "\n" + image_description).strip()
            else:
                from types import SimpleNamespace
                history.append(SimpleNamespace(role="customer", content=image_description))
            print(f"[Agent] Image resolved: {image_description[:80]}")

    last_customer_msg = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    # ── Language detection ────────────────────────────────────────────────────
    chosen_language = detect_language_choice(last_customer_msg)
    if chosen_language:
        language = chosen_language
    else:
        locked = request.detectedLanguage if request.detectedLanguage else None
        language = detect_language(last_customer_msg, locked)
        word_count = len(last_customer_msg.strip().split())

        # FIX: if last message is uncertain, check full conversation history
        if is_language_uncertain(last_customer_msg) and not locked:
            all_customer_text = " ".join(
                m.content for m in history if m.role == "customer"
            )
            language = detect_language(all_customer_text, locked)

        if locked and not chosen_language:
            if locked == "ar-latin" and _has_latin_darija_cues(last_customer_msg):
                language = "ar-latin"
            elif word_count <= 5 or language == locked:
                language = locked
            elif locked == "fr" and language != "ar":
                language = locked

    prior_turns = [m for m in history[:-1] if m.role in ("customer", "agent", "bot")]
    is_first_turn = len(prior_turns) == 0

    if is_first_turn and is_language_uncertain(last_customer_msg) and not chosen_language:
        language = "ar"
    if chosen_language and len(prior_turns) <= 2:
        language = chosen_language

    # ── Resume detection ──────────────────────────────────────────────────────
    resume_context = ""
    if len(history) >= 2:
        last_msg = history[-1]
        prev_bot_msgs = [m for m in history if m.role == "bot"]
        if last_msg.role == "customer" and prev_bot_msgs:
            last_bot = prev_bot_msgs[-1]
            unanswered = []
            for m in reversed(history):
                if m.role == "customer":
                    unanswered.append(m.content)
                elif m.role == "bot":
                    break
            if len(unanswered) > 1:
                resume_context = (
                    "\n\nRESUME CONTEXT: The AI was temporarily unavailable. "
                    "The customer sent these messages without receiving a response: "
                    + " | ".join(reversed(unanswered)) +
                    "\nThe last question/request you asked the customer was: " +
                    last_bot.content[:200] +
                    "\nIMPORTANT: Resume naturally from where the conversation stopped. "
                    "Do NOT greet again. Do NOT ask what you already asked. "
                    "Process the customer's answers and continue the flow."
                )

    # ── Intent classification ─────────────────────────────────────────────────
    intent = classify_intent_fast(history)

    # ── Gender detection ──────────────────────────────────────────────────────
    customer_gender = None
    for m in history:
        if m.role == "customer" and len(m.content) > 3:
            content = m.content
            if re.search(r'\b(حابة|بغيت\s+نطلبي|راني\s+حابة|نحبي|7abba|raki|diri|te9dri)\b', content):
                customer_gender = "female"
                break
            if re.search(r'\b(حاب\b|راني\s+حاب|7ab\b|rak\b|dir\b|te9dar\b)\b', content):
                customer_gender = "male"
                break

    if not customer_gender:
        for m in history:
            if m.role in ("agent", "bot") and re.search(r'(اسم|smiya|nom)', m.content.lower()):
                idx = history.index(m)
                if idx + 1 < len(history) and history[idx + 1].role == "customer":
                    name_candidate = history[idx + 1].content.strip()
                    if 2 < len(name_candidate) < 40:
                        customer_gender = detect_gender_from_name(name_candidate)
                        if customer_gender:
                            break

    # ── Build context ─────────────────────────────────────────────────────────
    product_catalog = build_product_catalog(request.products, language)
    shipping_opts = getattr(request, 'shippingOptions', None)
    shipping_section = build_shipping_section(shipping_opts)
    unavailable_wilayas = get_unavailable_wilayas(shipping_opts)

    if recent_orders := getattr(request, 'recentOrders', []):
        orders_context = "\n".join([
            f"• #{o.orderNumber} | {o.status} | {o.customerName or 'unknown'} | {o.customerPhone or 'no phone'}"
            for o in recent_orders
        ])
    else:
        orders_context = "لا توجد طلبات حديثة."

    combined_system_prompt = request.aiSystemPrompt or ""
    if resume_context:
        combined_system_prompt = (combined_system_prompt + resume_context).strip()

    system_prompt = build_prompt(
        intent=intent,
        store_name=request.storeName,
        language=language,
        is_first_turn=is_first_turn and not chosen_language,
        ai_flow_state=request.aiFlowState,
        product_catalog=product_catalog,
        orders_context=orders_context,
        shipping_section=shipping_section,
        ai_system_prompt=combined_system_prompt or None,
        customer_gender=customer_gender,
    )

    # ── AI reply with auto-retry ──────────────────────────────────────────────
    openai_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-20:]:
        if m.role == "customer":
            openai_messages.append({"role": "user", "content": m.content})
        elif m.role in ("agent", "bot"):
            openai_messages.append({"role": "assistant", "content": m.content})

    reply = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=700,
                temperature=0.3,
                messages=openai_messages,
            )
            reply = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"[Agent] AI call attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                reply = "سمحلي، ممكن تعاود وش قتلي "

    # ── Order extraction ──────────────────────────────────────────────────────
    extraction = await extract_order(history, request.products, unavailable_wilayas)
    action = {"type": "none"}

    if extraction.get("canAutoCreate") and extraction.get("orderData"):
        od = extraction["orderData"]
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

        baladiya = od.get("baladiya") or ""
        address = od.get("address") or ""
        full_address = " - ".join(filter(None, [baladiya, address])) or None

        action = {
            "type": "create_order",
            "customerName": od.get("customerName"),
            "customerPhone": od.get("customerPhone"),
            "wilaya": od.get("wilaya"),
            "address": full_address,
            "shippingOption": od.get("shippingOption") or "home_delivery",
            "items": fixed_items,
        }
    elif extraction.get("intent") == "cancel_order" and extraction.get("cancelPhone"):
        action = {
            "type": "cancel_order",
            "customerPhone": extraction["cancelPhone"],
        }
    elif extraction.get("canAutoUpdate") and extraction.get("updateData"):
        action = {
            "type": "update_order",
            "customerPhone": (extraction.get("orderData") or {}).get("customerPhone"),
            "updateData": extraction["updateData"],
        }

    return {
        "reply": reply,
        "detectedLanguage": language,
        "action": action,
    }