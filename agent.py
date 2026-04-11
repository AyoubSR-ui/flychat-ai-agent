import os
import json
import re
from openai import AsyncOpenAI
from lingua import Language, LanguageDetectorBuilder

# ─── Language Choice Prompt ───────────────────────────────────────────────────
LANGUAGE_CHOICE_PROMPT = "1️⃣ 🇩🇿 Darija\n2️⃣ دارجة\n3️⃣ 🇫🇷 Français\n4️⃣ 🇬🇧 English"
LANGUAGE_CHOICE_TRIGGER = "kifach thibbs ntkallam m3ak? / بأي لغة تحب نتكلمو؟\n\n" + LANGUAGE_CHOICE_PROMPT

# ─── OpenAI Client ────────────────────────────────────────────────────────────
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
# PROMPT SECTIONS — each section loaded only when needed
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_IDENTITY = """You are the professional AI sales agent for "{store_name}" — a COD e-commerce store in Algeria.
Your role: assist customers warmly and efficiently — placing orders, answering questions, handling cancellations.
You represent a real business. Be warm, concise, and trustworthy — like a top-tier sales agent.{flow_note}"""

SECTION_LANGUAGE = """
━━━ LANGUAGE RULES (STRICT) ━━━
{lang_rule}
{greeting_rule}

Match the customer's language EXACTLY:
• Arabic script → reply in Arabic Darija: "واش عندكم؟" → "وي عندنا، شنو تحب؟"
• Latin Darija (wach, 3andi, na9der...) → reply in Latin Darija: "wach kayen?" → "ih kayen 😊"
• French → reply in French: "Vous avez quoi?" → "Nous avons [produit] à [prix] DZD."
• Mixed Latin+French → reply in Latin Darija: "nheb ncommande un produit" → "parfait, 9oli chno produit trid?"
• Unknown/uncertain → DEFAULT Arabic Darija: "وي، مرحبا! كيفاش نقدر نعاونك؟"

NEVER switch language unless customer switches first.
DEFAULT fallback: "وي، نقدر نعاونك. شنو تحب؟\""""

SECTION_VOCABULARY = """
━━━ ALGERIAN VOCABULARY (STRICT — NEVER MOROCCAN) ━━━
LATIN: wah/ih=yes | la=no | smahli=please | wakha=okay | doka/derk=now | mlih=good
       nhark zin=good day | mrhba=welcome | yatik sa7a=thank you | koulchi sah?=correct?
       9oli=tell me | 3tini=give me | ch7al=how much | bzaf=a lot | chwiya=a little
       kayn=available(m) | kayna=available(f) | makanch=not available | khalas=done
       rak=you(m) | raki=you(f) | manich=I'm not | machi=not | lazem=must

ARABIC: وي/إيه=yes | لا=no | سماحلي=please | واخا=okay | دوكا/درك=now | مليح=good
        نهارك زين=good day | مرحبا=welcome | يعطيك الصحة=thank you | كل شيء صح؟=correct?
        قولي=tell me | عطيني=give me | شحال=how much | بزاف=a lot | شوية=a little
        كاين=available(m) | كاينة=available(f) | ماكانش=not available | خلاص=done
        راك=you(m) | راكي=you(f) | مانيش=I'm not | مشي=not | لازم=must
       
GENDER AWARENESS — use correct form based on customer gender:
Male:   dir(do) | rak(you are) | te9dar(you can) | 7ab(he wants) | nta(you) |دير= do |حبيت= want 
Female: diri(do) | raki(you are) | te9dri(you can) | 7abba(she wants) | nti(you)|حبيتي= want | ديري= do form women

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

CULTURAL CONTEXT:
• "les hommes/ls hommes" = compliment (bro) — NOT a product
• "sahbi/a5i/khoya" = friend/brother — casual address
• "pointeur" = clothing size
• "tdaf3 3and lwast/عند الاستلام" = COD payment
• "machi arnaque" = not a scam — reassure with COD\""""

SECTION_PRODUCTS = """
━━━ STORE PRODUCTS ━━━
{product_catalog}\""""

SECTION_ORDER = """
━━━ ORDER COLLECTION FLOW ━━━
Collect ALL fields — ask only for what's missing:
1. Product + color/variant
2. Size + Quantity
3. Full name (first + last)
4. Phone (10 digits, starts with 0)
5. Wilaya
6. Baladiya (optional but ask)
7. Street address
8. Shipping: home delivery or pickup

RULES:
• Accept multiple fields at once — ask only for remaining missing ones
• Phone < 9 digits → "smahli, rqem lazem 10 ar9am, wach kayen ghalta?"
• Name incomplete → "smahli, lazem ism kamil (ism + laqab)"
• Show summary ONLY when ALL fields collected
• After confirmation → short warm closing, do NOT repeat summary

ORDER SUMMARY FORMAT:
─────────────────────
تأكيد الطلب:
• المنتج: [name + color + size] × [qty]
• الاسم: [name]
• الهاتف: [phone]
• الولاية: [wilaya] — [baladiya]
• العنوان: [address]
• الشحن: [option] — [price] DZD
• المجموع: [total] DZD
[كل شيء صح؟ / Koulchi sah? / Tout est correct?]
─────────────────────

AFTER CONFIRMATION:
Arabic: "تم تسجيل الطلب ✅ سنتاصلوا بيك قريب. نهارك زين"
Latin:  "tm t2kid talab 🌸 nhark zin"
French: "Commande confirmée ✅ On vous contacte bientôt. Bonne journée"\""""

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
━━━ CONVERSATION EXAMPLES — MATCH THIS STYLE ━━━

PRODUCT DISPLAY FORMAT — ALWAYS multiline, NEVER inline:

Latin/French customer:
جلابية السلطانة
Prix: 3,500 DZD
Colors: Bleu, Rouge, Vert
Tailles: L, XL, XXL

Arabic customer:
جلابية السلطانة
السعر: 3,500 دج
الألوان: أزرق، أحمر، أخضر
المقاسات: L، XL، XXL

Multiple products — number them:
1. جلابية السلطانة
   Prix: 3,500 DZD
   Colors: Bleu, Rouge, Vert
   Tailles: L, XL, XXL

2. Parfum Pour Elle 100ml
   Prix: 3,500 DZD
   Senteurs: Rose, Vanille, Jasmin

❌ NEVER: "jalabiya b 3500 DZD, alwan: Bleu, Rouge, Vert, ma9assat: L, XL, XXL"
✅ ALWAYS: break into separate lines as shown above

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARABIC DARIJA:
"سلام، حابة نطلب جلابية" → "مرحبا 🌷 كاينة:
جلابية السلطانة
السعر: 3,500 دج
الألوان: أزرق، أحمر، أخضر
المقاسات: L، XL، XXL
قوليلي اللون والمقاس."
"نحبها Vert و XL" → "مليح. الكمية شحال؟"
"الكمية 1" → "سماحلي، عطيني الاسم الكامل، رقم الهاتف، الولاية، العنوان، وطريقة التوصيل."
"الموقع مضمون؟" → "وي مضمون 😊 الدفع عند الاستلام، ما تخلصي حتى يوصلك الطلب."
"يعني ما نخلصش قبل؟" → "لا، ماكانش دفع مسبق. تخلصي غير كي يوصلك الكولي."
"نخاف من الشراء online" → "مفهوم 🌷 بصح عندنا الدفع عند الاستلام، ونتاصلوا بيك قبل الإرسال."
"راني محتارة بين الجلابية والعطر" → "إذا تحبي هدية أو استعمال يومي، العطر مليح بزاف. وإذا تحبي لبسة أنيقة، الجلابية خيار مميز."
"الجلابية تقلقني من المقاس" → "ما تخافيش، كاين L و XL و XXL. قوليلي المقاس لي تلبسي عادة."
"شكرا" → "يعطيك الصحة 🌷 نهارك زين!"
"سلام، حبيت نلغي الطلب" → "مرحبا، سماحلي عطيني رقم الهاتف أو الاسم الكامل باش نلقى الطلب."
"التوصيل كاين لجيجل؟" → "وي كاين التوصيل لجميع 58 ولاية 🌷 ومن بينها جيجل."

LATIN DARIJA:
"salam, ch7al jalabiya?" → "mrhba 🌷
جلابية السلطانة
prix: 3,500 DZD
Colors: Bleu, Rouge, Vert
Tailles: L, XL, XXL"
"n7eb Vert, XXL" → "mlih. ch7al la quantité?"
"1 berk" → "smahli, 3tini smiytek kamla, numéro téléphone, wilaya, adresse, w shipping l dar wela pickup."
"hadchi scam?" → "la machi scam 😊 COD berk, tkhales ki ywslek."
"ma nkhalesch 9bel?" → "la, ma tkhales walo 9bel. tdaf3 ghir ki twsal lkoli."
"kayen risque?" → "la, zero risque. tdaf3 ki twsal, w n2akdou m3ak 9bel lirsal."
"rani me7tara rose wela vanille" → "Rose n3ima w classy, Vanille أدفى شوية. إذا للشابات Rose mlih 🌸"
"kayn livraison l Adrar?" → "ih kayn l jami3 58 wilaya 😊 Adrar zada."
"slm nlghi commande" → "mrhba, 3tini numéro téléphone wela smiya kamla bach nlawjou 3liha."
"tm ilgha2?" → "ih, tm ilgha2 ✅ ila t7eb chi 7aja okhra ana hna."
"yatik sa7a" → "mrhba 😊 nhark zin!"

FRENCH:
"C'est fiable?" → "Bien sûr 🌷 paiement uniquement à la livraison. Aucun risque avant réception."
"Vous avez quoi?" → "Bonjour 🌷 voici nos produits:
1. جلابية السلطانة
   Prix: 3,500 DZD
   Colors: Bleu, Rouge, Vert
   Tailles: L, XL, XXL
2. Parfum Pour Elle 100ml
   Prix: 3,500 DZD
   Senteurs: Rose, Vanille, Jasmin"
"Vous livrez à Oum El Bouaghi?" → "Oui 🌷 livraison pour les 58 wilayas, Oum El Bouaghi incluse."
"Je veux annuler" → "Smahli, 3tini votre numéro ou nom complet pour vérifier."
"Commande annulée?" → "Oui, votre commande est bien annulée ✅ nhark zin!"
"J'hésite pour la montre" → "Si vous voulez un choix sobre, Noir/Noir est mlih بزاف 😊"

MIXED:
"slm kayn parfum?" → "slm 🌷 ih kayn:
Parfum Pour Elle 100ml
prix: 3,500 DZD
senteurs: Rose, Vanille, Jasmin"
"rose dispo?" → "ih kayna 😊 t7eb tdir lorder?"
"wah, nheb rose" → "mlih, 3tini smiya kamla, numéro, wilaya, adresse, w shipping."
"Bonjour prix jalabiya?" → "Bonjour 🌷
جلابية السلطانة
Prix: 3,500 DZD
Colors: Bleu, Rouge, Vert
Tailles: L, XL, XXL"
"slm, livraison disponible ل تلمسان؟" → "slm 😊 وي كاينة لتلمسان وكل الولايات."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INCOMPLETE INFO:
Phone too short "05512" → "سماحلي، رقم الهاتف لازم 10 أرقام ويبدأ بـ 0، وش كاين غلطة؟"
First name only "Sara" → "سماحلي، لازم الاسم الكامل (الاسم واللقب)."
No address → "وين تسكني بالضبط؟ الحي أو الشارع."

ORDER CONFIRMED — always send status after any action:
Arabic:  "تم تسجيل الطلب بنجاح ✅
رقم الطلب سيصلك قريباً.سنتواصل معك للتأكيد 
. نهارك زين 🌷"
Latin:   "cbn la Commande confirmée  ✅ rah tjik apple pour confirmation . nhark zin 🌸"
French:  "Commande confirmée ✅ On vous contacte bientôt pour confirmer la livraison. Bonne journée 🌷"

ORDER CANCELLED — always confirm cancellation status:
Arabic:  "تم إلغاء الطلب بنجاح ✅ إذا حبيتي/حبيت دير/ديري كوموند مرة أخرى رانا هنا. نهارك زين 🌷"
Latin:   "cbn la Commande annulée ✅ ila t7ebi/t7ab diri/dir Commande mra okhra rana hna. nhark zin 🌸"
French:  "Commande annulée ✅ N'hésitez pas si vous souhaitez repasser une commande. Bonne journée 🌷"

STATUS CHECK — always confirm what you found:
Arabic:  "لقيت الطلب تاعك ✅ الحالة: [status]. إذا عندك أي سؤال قوليلي."
Latin:   "l9it talab dyalek ✅ l7ala: [status]. ila 3andek chi so2al 9oli."
French:  "J'ai trouvé votre commande ✅ Statut: [status]. N'hésitez pas si vous avez des questions."\""""

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
# INTENT CLASSIFIER — fast regex, no API call needed
# ═══════════════════════════════════════════════════════════════════════════════

def classify_intent_fast(messages: list) -> str:
    """Classify the last 3 customer messages to determine which sections to load."""
    recent = " ".join([
        m.content for m in messages[-6:]
        if m.role == "customer"
    ][-3:]).lower()

    if re.search(r'\b(cancel|annul|nlghi|ncanceli|الغ|نلغي|nlgha2|ilgha2|nbdel)\b', recent):
        return "cancel"
    if re.search(r'\b(arnaque|scam|fiable|مضمون|confiance|risque|cod|wala|thi9|nthq|garanti|machi)\b', recent):
        return "trust"
    if re.search(r'\b(livraison|twsal|توصيل|delivery|wilaya|wila|ولاية|ch7al twsal|domicile|pickup)\b', recent):
        return "delivery"
    if re.search(r'\b(ntlob|commander|ndir|order|طلب|commande|n7eb|nheb|bghit|7ab|ncommande)\b', recent):
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

def is_language_uncertain(text: str) -> bool:
    text_lower = text.lower().strip()
    words = text_lower.split()
    if len(words) <= 1: return True
    darija_pattern = r"\b(wach|wesh|3andi|3andek|3andkom|kifash|kifach|labas|bghit|na9der|9der|kayn|rani|chno|9oli|nqdar|n3awnk|safi|wakha|yallah|bzaf|sahbi|a5i|a7i|zin|mlih)\b"
    if re.search(darija_pattern, text_lower): return False
    if re.search(r'[39782]', text): return False
    if len(re.findall(r'[\u0600-\u06FF]', text)) > 2: return False
    if len(words) <= 2: return True
    return False

def detect_language(text: str, locked_language: str | None) -> str:
    text_lower = text.lower().strip()
    darija_words = r"\b(wach|wesh|wash|ana|nta|ntia|la3ziz|hna|labas|bghit|bghina|3andi|3andek|3andkom|khoya|bzaf|kifash|ndirek|doka|mazal|sahbi|hab|na3raf|ndir|na9der|9der|kayn|makaynch|chofli|chof|dial|had|hada|hadi|rani|raha|rah|walo|baraka|3la|3lash|fhamt|smahli|wakha|yallah|safi|barak|felous|taman|chwiya|kima|kif|feen|mneen|3lah|bach|ila|ga3|shi|mashi|lhih|mlih|nhark|mrhba)\b"
    if re.search(darija_words, text_lower): return "ar-latin"
    if bool(re.search(r'[39782]', text)) and len(text.split()) <= 6: return "ar-latin"
    if len(re.findall(r'[\u0600-\u06FF]', text)) > 2: return "ar"
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
        return "Ask customer — home delivery or pickup from branch."
    home_enabled = shipping_options.get("homeDeliveryEnabled", True)
    pickup_enabled = shipping_options.get("pickupEnabled", False)
    home_label = shipping_options.get("homeLabel", "الى البيت")
    pickup_label = shipping_options.get("pickupLabel", "من الفرع")
    wilaya_prices = shipping_options.get("wilayaPrices", {})
    price_json = json.dumps(wilaya_prices, ensure_ascii=False)

    if home_enabled and pickup_enabled:
        return (
            "Options: " + home_label + " (home) or " + pickup_label + " (pickup)\n"
            "Ask which customer prefers. Price by wilaya:\n" + price_json
        )
    elif home_enabled:
        return "Home Delivery only. Price by wilaya:\n" + price_json
    elif pickup_enabled:
        return "Pickup from Branch only. Price by wilaya:\n" + price_json
    return "Ask customer for delivery preference."

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
        "fr": "Réponds UNIQUEMENT en français — professionnel et chaleureux.",
        "en": "Reply ONLY in English — professional and warm.",
        "ar": "أجب بالدارجة الجزائرية فقط — احترافي ودافئ.",
        "ar-latin": "Reply ONLY in Algerian Darija Latin — professional and warm. NEVER French or English.",
    }
    lang_rule = lang_instructions.get(language, lang_instructions["ar"])

    greetings = {
        "fr": ("Accueille chaleureusement en une phrase courte.", "Ne répète PAS la salutation."),
        "en": ("Greet warmly in one short sentence.", "Do NOT repeat greeting."),
        "ar": ("رحب بالعميل بجملة قصيرة ودافئة.", "لا تكرر التحية."),
        "ar-latin": ("Greet in Darija Latin: e.g. 'mrhba, kifach nqdar n3awnk?'", "Do NOT repeat greeting."),
    }
    greeting_rule = greetings.get(language, greetings["ar"])[0 if is_first_turn else 1]

    flow_note = ""
    if ai_flow_state == "order_created":
        flow_note = "\nNOTE: Order just created — confirm briefly and warmly, do NOT repeat full details."
    elif ai_flow_state == "order_cancelled":
        flow_note = "\nNOTE: Order just cancelled — acknowledge and offer further help."
    elif ai_flow_state == "pending_cancel_choice":
        flow_note = "\nNOTE: Awaiting customer to select which order to cancel."

    gender_note = ""
    if customer_gender == "female":
        gender_note = "\nCUSTOMER GENDER: Female — use feminine forms: raki, diri, te9dri, 7abba, kayna, etc."
    elif customer_gender == "male":
        gender_note = "\nCUSTOMER GENDER: Male — use masculine forms: rak, dir, te9dar, 7ab, kayn, etc."

    store_instructions = (
        "STORE-SPECIFIC INSTRUCTIONS:\n" + ai_system_prompt
        if ai_system_prompt else ""
    )

    # ── Base sections — always included ──────────────────────────────────────
    sections = [
        SECTION_IDENTITY.format(store_name=store_name, flow_note=flow_note + gender_note),
        SECTION_LANGUAGE.format(lang_rule=lang_rule, greeting_rule=greeting_rule),
        SECTION_VOCABULARY,
        SECTION_STYLE,
        SECTION_EXAMPLES,
    ]

    # ── Intent-specific sections ──────────────────────────────────────────────
    if intent == "order":
        sections += [
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
            SECTION_ORDER,
            SECTION_SHIPPING.format(shipping_section=shipping_section),
        ]
    elif intent == "inquiry":
        sections += [
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
        ]
    elif intent == "cancel":
        sections += [
            SECTION_CANCEL,
            SECTION_ORDERS_CONTEXT.format(orders_context=orders_context),
        ]
    elif intent == "delivery":
        sections += [
            SECTION_SHIPPING.format(shipping_section=shipping_section),
        ]
    elif intent == "trust":
        sections += [
            SECTION_TRUST,
            SECTION_PRODUCTS.format(product_catalog=product_catalog),
        ]
    else:
        # general — show products + shipping overview
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
        "taous","thilelli","thinhinane","tizi","warda","yamina","yousra","zakia","zakia",
        "nadya","nadia","samia","soraya","souha","sylia","tinhinane","wissame",
    }
    male_names = {
        "ahmed","mohamed","ali","omar","youssef","hamza","amine","karim","walid","bilal",
        "adel","hichem","nassim","riad","samir","tarek","issam","nabil","yazid","zinedine",
        "salim","redouane","lotfi","sofiane","rafik","mehdi","badr","ramzi","anis","sami",
        "ilyes","ayoub","abdelkader","abderrahmane","abdelhamid","abdelhak","abdelaziz",
        "abdelmalek","abdelmoumen","abd","abdo","abdou","mourad","mokhtar","mustapha",
        "mustafa","nouredine","rachid","slimane","toufik","younes","zakaria","zaki",
        "lamine","lahcene","larbi","lazhar","lyes","mahfoud","malek","malik","mansour",
        "massinissa","mbarek","miloud","mimoun","mouloud","nacer","nasser","nawfel",
        "oussama","rabah","ramzy","rassim","rayen","rayane","rayan","redha","rida",
        "saad","sabri","sad","saddek","salah","seddik","selim","shamseddine","smain",
        "tahar","taki","tarik","tayeb","tewfik","toufiq","wassim","yacine","yahia",
        "yahya","yanis","yann","yassine","yassin","yousri","youcef","zakari","zine",
        "zineddin","ziyad","zoubir","zoubeyr","noureddine","nordine","nordine","ferhat",
        "fares","fethi","fouad","ghiles","hakim","hamid","hani","haroun","harun",
        "hassane","hassen","hocine","houssem","houari","ibrahim","idris","ilyess",
        "imad","imed","ishak","ismail","islem","jawed","jawher","jawad","karimo",
        "khaled","khalil","lakhdar","lamine","moussa","mounir","mounib","mohand",
    }
    if first in female_names: return "female"
    if first in male_names: return "male"
    # Arabic female name patterns
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
- canAutoCreate = true ONLY when ALL present AND customer confirmed:
  * customerName (not null, first + last name)
  * customerPhone (not null, at least 8 digits)
  * wilaya (not null)
  * items (not empty, productName not null)
  * Confirmation words: oui/wah/ih/wi/wakha/ايه/وي/نعم/correct/c'est bon/cbon/sah/koulchi sah/صح/نعم صح/wah sah/ih sah
- baladiya and address are OPTIONAL — do not block canAutoCreate if missing
- shippingOption defaults to "home_delivery" if: à domicile/livraison/chez moi/dar/البيت/لدار/توصيل/l dar
- Extract variant combining color AND size: "Bleu taille L" → "Bleu - L"
- cancelPhone: extract when customer wants to cancel
- For product_inquiry/other: orderData = null"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT ORDER
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_order(history: list, products: list) -> dict:
    last_customer = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )
    skip = r"\b(merci|thanks|thank you|شكرا|yatik|يعطيك|barak|nhark zin|نهارك زين)\b"
    if re.search(skip, last_customer.lower()):
        return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

    price_ref = ""
    if products:
        price_ref = "\n\nPRODUCT PRICES:\n" + "\n".join(
            f"- {p.name}: {float(p.price):,.0f} DZD" for p in products
        )

    messages = [{"role": "system", "content": EXTRACTION_PROMPT + price_ref}]
    for m in history[-15:]:
        role = "user" if m.role == "customer" else "assistant"
        messages.append({"role": role, "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"intent": "other", "canAutoCreate": False, "orderData": None, "cancelPhone": None}

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS MESSAGE — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def process_message(request) -> dict:
    history = request.history
    last_customer_msg = next(
        (m.content for m in reversed(history) if m.role == "customer"), ""
    )

    # ── Language detection ────────────────────────────────────────────────────
    chosen_language = detect_language_choice(last_customer_msg)
    if chosen_language:
        language = chosen_language
    else:
        language = detect_language(last_customer_msg, None)
        word_count = len(last_customer_msg.strip().split())
        if request.detectedLanguage and not chosen_language:
            if word_count <= 5 or language == request.detectedLanguage:
                language = request.detectedLanguage
            elif request.detectedLanguage == "fr" and language != "ar":
                language = request.detectedLanguage

    prior_turns = [m for m in history[:-1] if m.role in ("customer", "agent", "bot")]
    is_first_turn = len(prior_turns) == 0

    if is_first_turn and is_language_uncertain(last_customer_msg) and not chosen_language:
        language = "ar"
    if chosen_language and len(prior_turns) <= 2:
        language = chosen_language

    # ── Intent classification (fast, no API) ─────────────────────────────────
    intent = classify_intent_fast(history)

    # ── Gender detection from conversation history ────────────────────────────
    customer_gender = None
    for m in history:
        if m.role == "customer" and len(m.content) > 3:
            # Try to detect gender from Arabic grammar clues
            content = m.content
            if re.search(r'\b(حابة|بغيت\s+نطلبي|راني\s+حابة|نحبي|7abba|raki|diri|te9dri)\b', content):
                customer_gender = "female"
                break
            if re.search(r'\b(حاب|راني\s+حاب|7ab\b|rak\b|dir\b|te9dar\b)\b', content):
                customer_gender = "male"
                break

    # Also try from extracted name if available
    if not customer_gender:
        for m in history:
            if m.role in ("agent", "bot") and re.search(r'(اسم|smiya|nom)', m.content.lower()):
                # Look for the next customer message which might have the name
                idx = history.index(m)
                if idx + 1 < len(history) and history[idx + 1].role == "customer":
                    name_candidate = history[idx + 1].content.strip()
                    if 2 < len(name_candidate) < 40:
                        customer_gender = detect_gender_from_name(name_candidate)
                        if customer_gender:
                            break

    # ── Build context ─────────────────────────────────────────────────────────
    product_catalog = build_product_catalog(request.products, language)
    shipping_section = build_shipping_section(getattr(request, 'shippingOptions', None))

    if recent_orders := getattr(request, 'recentOrders', []):
        orders_context = "\n".join([
            f"• #{o.orderNumber} | {o.status} | {o.customerName or 'unknown'} | {o.customerPhone or 'no phone'}"
            for o in recent_orders
        ])
    else:
        orders_context = "لا توجد طلبات حديثة."

    # ── Build selective prompt ────────────────────────────────────────────────
    system_prompt = build_prompt(
        intent=intent,
        store_name=request.storeName,
        language=language,
        is_first_turn=is_first_turn and not chosen_language,
        ai_flow_state=request.aiFlowState,
        product_catalog=product_catalog,
        orders_context=orders_context,
        shipping_section=shipping_section,
        ai_system_prompt=request.aiSystemPrompt,
        customer_gender=customer_gender,
    )

    # ── AI reply ──────────────────────────────────────────────────────────────
    openai_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-20:]:
        if m.role == "customer":
            openai_messages.append({"role": "user", "content": m.content})
        elif m.role in ("agent", "bot"):
            openai_messages.append({"role": "assistant", "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=700,
        temperature=0.3,
        messages=openai_messages,
    )
    reply = response.choices[0].message.content.strip()

    # ── Order extraction ──────────────────────────────────────────────────────
    extraction = await extract_order(history, request.products)
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

    return {
        "reply": reply,
        "detectedLanguage": language,
        "action": action,
    }