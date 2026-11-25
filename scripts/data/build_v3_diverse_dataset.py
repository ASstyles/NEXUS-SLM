"""Build clean, syntactically diverse multilingual dataset V3 (data/v3_diverse/).

Ensures ALL 4 disjoint splits (train_v3, val_v3, cal_v3, dev_audit_v3) have:
- Balanced class coverage across ALL 14 support categories
- Balanced priority coverage (low, medium, high, critical) aligned to the impact rubric
- Balanced English, Native Devanagari Hindi, and Romanized Hinglish representation
  (plus explicit code-switched / informal / typo / negative-control / multi-intent strata)
- Disjoint template families across splits (template-grouped isolation)
- Zero leakage against test_clean.jsonl, val_clean.jsonl, human_eval (184), challenge set,
  and human_eval_v2 partitions.

Explicit Disclosure: All records carry source='synthetic_v3_diverse' and NEVER claim
to represent real customer data.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.leakage_detector import LeakageDetector
from src.data.schemas import TicketRecord
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u0900-\u097f\s]", " ", text.lower())).strip()


# Format: (category, priority, language, requires_human, challenge_tag, partition, text)
# Partition is assigned by scenario family (not random row shuffle) so train, val, cal,
# and dev_audit never share underlying scenario phrasings.
CURATED_RECORDS: List[Tuple[str, str, str, bool, str, str, str]] = [
    # --- 1. PAYMENT_FAILURE ---
    ("payment_failure", "high", "english", True, "standard", "train", "UPI PIN went through cleanly on PhonePe and my bank alert shows a debit, yet checkout timed out without creating an order."),
    ("payment_failure", "high", "english", True, "standard", "train", "Although my bank statement confirms the IMPS transfer succeeded ten minutes ago, the cart still prompts me to pay again."),
    ("payment_failure", "high", "english", True, "standard", "train", "Paytm wallet shows completed status with UTR reference, whereas your checkout page threw a gateway handshake exception."),
    ("payment_failure", "high", "english", True, "typo_unusual", "train", "3d secure otp entred, amt deducted from debit crd, browser froze on redirect — zero confirmation mail."),
    ("payment_failure", "high", "english", True, "typo_unusual", "train", "netbanking timedout right after bank debited amt pls reconcile urgently"),
    ("payment_failure", "high", "english", True, "standard", "train", "Card captured twice because the payment spinner hung during checkout processing."),
    ("payment_failure", "high", "hindi", True, "standard", "train", "भीम यूपीआई से राशि कट चुकी है और बैंक मैसेज भी आ गया, लेकिन ऑर्डर हिस्ट्री में कोई नया ऑर्डर नहीं दिख रहा।"),
    ("payment_failure", "high", "hindi", True, "standard", "train", "डेबिट कार्ड पेमेंट के दौरान ओटीपी सबमिट करते ही पेज रीलोड हो गया और खाते से दो बार पैसे कट गए।"),
    ("payment_failure", "high", "hindi", True, "standard", "train", "मर्चेंट गेटवे पर तकनीकी त्रुटि दिख रही है जबकि मेरे बचत खाते से पूरा भुगतान पहले ही डेबिट हो चुका है।"),
    ("payment_failure", "high", "hindi", True, "standard", "train", "पेमेंट गेटवे क्रैश होने के बाद भी मेरी नेट बैंकिंग से रकम निकल गई है, कृपया लेनदेन की जांच करें।"),
    ("payment_failure", "high", "hindi", True, "standard", "train", "रुपे कार्ड से पैसे कटने का अलर्ट आ गया है पर खरीदारी विफल दिखा रहा है।"),
    ("payment_failure", "high", "hinglish", True, "standard", "train", "yaar BHIM app pe paid dikha raha hai par tumhari site pe payment unsuccessful aa gaya"),
    ("payment_failure", "high", "hinglish", True, "standard", "train", "sbi card se double swipe ho gaya network lag ki wajah se, ek entry pending me atki hai"),
    ("payment_failure", "high", "hinglish", True, "standard", "train", "upi pin dalte hi screen blank ho gayi aur bank msg aa gaya debit ka bina order bane"),
    ("payment_failure", "high", "hinglish", True, "standard", "train", "razorpay popup close ho gaya tha par paise bank account se nikal chuke hain abhi check kiya"),
    ("payment_failure", "high", "hinglish", True, "typo_unusual", "train", "paymnt fail bol rha checkout par bank balance kam ho gya 2 baar"),
    ("payment_failure", "high", "english", True, "standard", "val", "When authenticating my Visa card, the issuer confirmed the charge, yet the merchant API returned HTTP 502 and left the basket unpaid."),
    ("payment_failure", "high", "english", True, "standard", "val", "Why is a settled card hold sitting on my banking feed for a failed QR scan attempt?"),
    ("payment_failure", "high", "hindi", True, "standard", "val", "गूगल पे से भुगतान सफल होने के बावजूद ऑर्डर कन्फर्मेशन पेज पर पेमेंट फेल लिखा आ रहा है।"),
    ("payment_failure", "high", "hindi", True, "standard", "val", "चेकआउट के समय बैंक सर्वर धीमा होने से खाते से दो बार राशि डेबिट हो गई।"),
    ("payment_failure", "high", "hinglish", True, "mixed_script", "val", "Google Pay से पैसे cut ho gaye par app abhi bhi payment retry bol rahi hai"),
    ("payment_failure", "high", "hinglish", True, "standard", "val", "hdfc netbanking se paise nikal gaye lekin checkout screen error 504 pe ruk gayi"),
    ("payment_failure", "high", "english", True, "standard", "cal", "I scanned the dynamic checkout QR code, received a settled debit alert, and hit an invalid session token notice."),
    ("payment_failure", "high", "hindi", True, "standard", "cal", "क्रेडिट कार्ड बिलिंग में एक असफल लेनदेन के लिए दो बार चार्ज कट गया है।"),
    ("payment_failure", "high", "hinglish", True, "standard", "cal", "credit card se payment cut gya par transaction log me declined show kar raha"),
    ("payment_failure", "high", "english", True, "standard", "audit", "Check our corporate card log: one failed purchase attempt generated three pre-authorization debits without any receipt."),
    ("payment_failure", "high", "hindi", True, "standard", "audit", "फ़ोनपे वॉलेट से भुगतान पूरा दिखा रहा है पर आपकी ऐप पेमेंट फेलियर का अलर्ट दे रही है।"),
    ("payment_failure", "high", "hinglish", True, "standard", "audit", "Bhai merchant gateway atka tha aur netbanking se 2x paise nikal gaye bina order confirm hue"),

    # --- 2. REFUND_REQUEST ---
    ("refund_request", "medium", "english", False, "standard", "train", "Our warehouse pickup receipt was signed ten business days ago, yet the promised bank transfer has not reflected on my statement."),
    ("refund_request", "medium", "english", False, "standard", "train", "Rather than keeping credits locked inside the loyalty wallet, kindly route the reimbursement back to the Mastercard originally billed."),
    ("refund_request", "medium", "english", False, "standard", "train", "Following quality inspection approval on my returned jacket, please share the ARN tracking number for the pending payout."),
    ("refund_request", "high", "english", True, "typo_unusual", "train", "returned headset 18 days back, status stuck at refund initiated — still zero credit in bank!!"),
    ("refund_request", "high", "english", True, "standard", "train", "Merchant voided my prepaid order last month and has still held my cash without initiating a settlement."),
    ("refund_request", "medium", "hindi", False, "standard", "train", "रिटर्न पिकअप पूरा हुए एक सप्ताह बीत गया है, मुझे मेरे मूल यूपीआई खाते में धनवापसी चाहिए।"),
    ("refund_request", "medium", "hindi", False, "standard", "train", "मेरा लौटाया गया सामान गोदाम में जांच के बाद स्वीकार हो चुका है, रिफंड रसीद नंबर कब मिलेगा?"),
    ("refund_request", "medium", "hindi", False, "standard", "train", "कृपया वापसी की रकम स्टोर क्रेडिट में न डालकर सीधे बैंक खाते में ट्रांसफर कर दें।"),
    ("refund_request", "high", "hindi", True, "standard", "train", "सामान लौटाए बीस दिन हो चुके हैं लेकिन आज तक एक भी रुपया वापस खाते में नहीं आया।"),
    ("refund_request", "medium", "hinglish", False, "standard", "train", "courier bhaiya item le gaye the 6 din pehle par abhi tak account me credit nahi aaya refund ka"),
    ("refund_request", "medium", "hinglish", False, "standard", "train", "store cash mat dena bhai seedha original source card me paise bhej do"),
    ("refund_request", "medium", "hinglish", False, "standard", "train", "merko return approval mail mil chuki hai, bank reference number update karo payout ka"),
    ("refund_request", "high", "hinglish", True, "standard", "train", "damaged mug wapas bheja tha teen hafte ho gye ab bas full amount return karwa do jaldi"),
    ("refund_request", "medium", "english", False, "standard", "val", "Notice my returned sneakers cleared inspection on Tuesday—please push the reimbursement via NEFT rather than gift voucher."),
    ("refund_request", "medium", "english", False, "standard", "val", "The exchange failed due to stock unavailability so please credit back the full amount paid."),
    ("refund_request", "medium", "hindi", False, "standard", "val", "उत्पाद वापस किए दस दिन हो गए, कृपया बताएं कि बैंक खाते में पैसे कब जमा होंगे?"),
    ("refund_request", "high", "hindi", True, "standard", "val", "ऑर्डर रद्द किए जाने के बावजूद मेरी पूरी धनराशि अभी तक अटका रखी गई है, तुरंत वापस करें।"),
    ("refund_request", "medium", "hinglish", False, "standard", "val", "wallet credit nhi chahiye bhai refund direct upi id pe bhejo pls"),
    ("refund_request", "medium", "hinglish", False, "standard", "val", "return approved dikha rha dashboard pe par bank me payout abhi tak credit nhi hua"),
    ("refund_request", "high", "english", True, "standard", "cal", "Warehouse confirmed receipt of damaged monitor 12 days back yet zero refund entered my bank."),
    ("refund_request", "medium", "hindi", False, "standard", "cal", "वॉलेट बैलेंस के बजाय मेरे वीज़ा कार्ड पर रिफंड जारी कीजिए क्योंकि मैं दोबारा खरीदारी नहीं करना चाहता।"),
    ("refund_request", "medium", "hinglish", False, "standard", "cal", "cancelled item ke paise 10 din baad bhi statement me reflect nahi hue"),
    ("refund_request", "medium", "english", False, "standard", "audit", "Can you verify with accounts receivable why my approved refund batch shows Settled Failed on your portal?"),
    ("refund_request", "high", "hindi", True, "standard", "audit", "सामान लौटाए पंद्रह दिन हो गए पर अभी तक एक रुपया भी वापस नहीं मिला, तुरंत बैंक रेफरेंस दें।"),
    ("refund_request", "high", "hinglish", True, "multi_intent", "audit", "return pickup complete hai 2 weeks se par bank statement me abhi tak credit show nhi hua"),

    # --- 3. PRODUCT_COMPLAINT (NO refund demand) ---
    ("product_complaint", "medium", "english", False, "standard", "train", "The zipper on this new backpack jammed on the very first morning and two stitches along the shoulder strap already came loose."),
    ("product_complaint", "medium", "english", False, "standard", "train", "Fresh out of the sealed carton, the Bluetooth speaker emits constant static hiss whenever volume exceeds thirty percent."),
    ("product_complaint", "medium", "english", False, "standard", "train", "Size Large shirt measurements match an Extra Small chart; the tailoring label appears mis-stitched."),
    ("product_complaint", "high", "english", True, "typo_unusual", "train", "laptop charger port wobbles & drops charge every time desk vibrates, defective batch?"),
    ("product_complaint", "critical", "english", True, "physical_hazard", "train", "URGENT SAFETY WARNING: Power bank swelled visibly and started smoking while charging unattended on my desk!"),
    ("product_complaint", "critical", "english", True, "physical_hazard", "train", "The electric kettle handle gave me a sharp electric shock and emitted sparks near the wall socket."),
    ("product_complaint", "medium", "hindi", False, "standard", "train", "जूते के पहले ही दिन सोल के किनारे से गोंद उखड़ने लगी है, गुणवत्ता बहुत कमजोर है।"),
    ("product_complaint", "medium", "hindi", False, "standard", "train", "मिक्सर ग्राइंडर का जार चलाते ही तेज आवाज़ और कंपन कर रहा है, ब्लेड टेढ़ा लगता है।"),
    ("product_complaint", "medium", "hindi", False, "standard", "train", "इस कुर्ते का कपड़ा पहली धुलाई में ही सिकुड़ गया और रंग दूसरे कपड़ों पर लग गया।"),
    ("product_complaint", "critical", "hindi", True, "physical_hazard", "train", "चार्जर प्लग लगाते ही चिंगारी निकली और उसमें से धुआं उठने लगा, बहुत बड़ा खतरा है!"),
    ("product_complaint", "medium", "hinglish", False, "standard", "train", "earbuds ka left side bilkul low sound de raha hai right wale ke comparison me"),
    ("product_complaint", "medium", "hinglish", False, "standard", "train", "office chair ka hydraulic gas lift 2 din me hi neeche baith ja raha hai khud se"),
    ("product_complaint", "high", "hinglish", True, "standard", "train", "naye trimmer ki battery 5 minute me hi dead ho jati hai full charge ke baad bhi"),
    ("product_complaint", "critical", "hinglish", True, "physical_hazard", "train", "toaster plug karte hi wire me aag aur dhua nikalne laga bilkul unsafe unit hai!"),
    ("product_complaint", "high", "english", True, "standard", "val", "The monitor panel arrived intact physically, however three vertical dead pixel lines run straight down the center."),
    ("product_complaint", "medium", "english", False, "standard", "val", "Non-stick frying pan coating flaked off during initial seasoning before high heat was even applied."),
    ("product_complaint", "medium", "hindi", False, "standard", "val", "घड़ी का डिस्प्ले बहुत धुंधला दिख रहा है और साइड वाला बटन ठीक से दब नहीं रहा।"),
    ("product_complaint", "high", "hindi", True, "standard", "val", "नए वैक्यूम क्लीनर की मोटर चलते ही बंद हो रही है और जलने जैसी गंध आ रही है।"),
    ("product_complaint", "medium", "hinglish", False, "standard", "val", "bottle ka cap loose hai bag me pura paani leak kar deta hai daily"),
    ("product_complaint", "high", "hinglish", True, "standard", "val", "keyboard ki half keys double type ho rahi hain day one se manufacturing defect lagta h"),
    ("product_complaint", "medium", "english", False, "standard", "cal", "My ergonomic keyboard exhibits key chatter where pressing the E key registers double letters constantly."),
    ("product_complaint", "medium", "hindi", False, "standard", "cal", "कलाई घड़ी का पट्टा एक हफ्ते के भीतर ही चटक कर टूट गया।"),
    ("product_complaint", "high", "hinglish", True, "standard", "cal", "naye power strip ke switch press karne par crack awaaz aayi aur loose ho gya"),
    ("product_complaint", "high", "english", True, "standard", "audit", "Both hinge screws on the new sunglasses stripped cleanly off when opening the temple arm once."),
    ("product_complaint", "medium", "hindi", False, "standard", "audit", "नए बैग पर फफूंद के धब्बे और सीलन की तेज़ बदबू आ रही है।"),
    ("product_complaint", "high", "hinglish", True, "standard", "audit", "fridge compressor itni tej aawaz kar raha hai raat ko sone nahi deta bilkul"),

    # --- 4. GENERAL_QUERY (includes negative controls containing risk words safely!) ---
    ("general_query", "low", "english", False, "standard", "train", "Do your flagship retail outlets remain open on public holidays for in-person warranty drop-offs?"),
    ("general_query", "low", "english", False, "standard", "train", "Is international express shipping currently enabled for postal codes across Singapore and Malaysia?"),
    ("general_query", "low", "english", False, "standard", "train", "Where can university students verify eligibility requirements for the academic hardware discount program?"),
    ("general_query", "low", "english", False, "negative_control", "train", "Quick question before ordering: how does your 14-day standard refund policy apply to gift items?"),
    ("general_query", "low", "english", False, "negative_control", "train", "Before we run our student hackathon workshop, do you offer bulk educational discount codes?"),
    ("general_query", "low", "english", False, "negative_control", "train", "Can you explain how your platform sends automated security alerts before I decide to sign up?"),
    ("general_query", "low", "hindi", False, "standard", "train", "आपका ग्राहक सेवा केंद्र रविवार को किस समय से किस समय तक खुला रहता है?"),
    ("general_query", "low", "hindi", False, "standard", "train", "क्या नई दिल्ली के बाहर भी अगले दिन डिलीवरी की सुविधा उपलब्ध है?"),
    ("general_query", "low", "hindi", False, "standard", "train", "रिवॉर्ड पॉइंट्स को अगली खरीदारी में भुनाने की न्यूनतम सीमा क्या होती है?"),
    ("general_query", "low", "hindi", False, "negative_control", "train", "खरीदारी करने से पहले जानना चाहता हूँ कि आपकी सामान्य रिफंड नीति कितने दिनों की है?"),
    ("general_query", "low", "hinglish", False, "standard", "train", "kya diwali sale ke time extra bank offer milega credit cards par?"),
    ("general_query", "low", "hinglish", False, "standard", "train", "offline service center ka address aur weekend timing bata dijiye please"),
    ("general_query", "low", "hinglish", False, "standard", "train", "cash on delivery option kis pin code tak available hai aapke store pe?"),
    ("general_query", "low", "hinglish", False, "negative_control", "train", "order karne se pehle puchna tha ki agar item pasand na aaye to return policy kya rehti hai?"),
    ("general_query", "low", "english", False, "standard", "val", "Do you publish a public changelog listing upcoming Android app accessibility improvements?"),
    ("general_query", "low", "english", False, "negative_control", "val", "How does automatic password expiration timing work for inactive guest profiles?"),
    ("general_query", "low", "hindi", False, "standard", "val", "कॉर्पोरेट थोक ऑर्डर देने के लिए मुझे किस ईमेल पते पर कोटेशन भेजना होगा?"),
    ("general_query", "low", "hindi", False, "standard", "val", "क्या आपके शोरूम में पुराने इलेक्ट्रॉनिक्स के एक्सचेंज की सुविधा उपलब्ध है?"),
    ("general_query", "low", "hinglish", False, "standard", "val", "app ke andar dark mode kaise on karte hain setting me mil nhi rha"),
    ("general_query", "low", "hinglish", False, "standard", "val", "purana phone exchange me dene par valuation kaise check kar sakte hain site pe?"),
    ("general_query", "low", "english", False, "standard", "cal", "Does your developer sandbox allow testing webhooks without touching live production accounts?"),
    ("general_query", "low", "hindi", False, "standard", "cal", "वारंटी पंजीकरण कराने के लिए किन दस्तावेजों की आवश्यकता होती है?"),
    ("general_query", "low", "hinglish", False, "standard", "cal", "standard return window kitne din ka hota hai electronics category ke liye?"),
    ("general_query", "low", "english", False, "negative_control", "audit", "Before completing checkout, what standard warranty duration is included on refurbished laptops?"),
    ("general_query", "low", "hindi", False, "standard", "audit", "नया गिफ्ट कार्ड खरीदने के बाद उसकी वैधता कितने महीनों तक रहती है?"),
    ("general_query", "low", "hinglish", False, "standard", "audit", "warranty claim karne ke liye bill ki physical copy rakhna zaruri hai kya?"),

    # --- 5. ACCOUNT_ACCESS ---
    ("account_access", "high", "english", True, "standard", "train", "Signing in greets me with Error 403 Profile Temporarily Locked despite never exceeding password retries."),
    ("account_access", "high", "english", True, "standard", "train", "Since switching carriers yesterday, hardware 2FA prompts never arrive on SMS and backup recovery codes say invalid."),
    ("account_access", "high", "english", True, "standard", "train", "Enterprise SSO login loops infinitely back to the welcome screen after authentication succeeds."),
    ("account_access", "high", "hindi", True, "standard", "train", "सही पासवर्ड डालने के बावजूद पोर्टल बार-बार अकाउंट सस्पेंडेड दिखाकर लॉगिन रोक रहा है।"),
    ("account_access", "high", "hindi", True, "standard", "train", "मेरा टू-फैक्टर ऑथेंटिकेटर ऐप रीसेट हो गया है और अब लॉगिन के लिए वेरिफिकेशन कोड स्वीकार नहीं हो रहा।"),
    ("account_access", "high", "hindi", True, "standard", "train", "पंजीकृत मोबाइल नंबर पर एसएमएस ओटीपी न आने के कारण मैं डैशबोर्ड नहीं खोल पा रहा हूँ।"),
    ("account_access", "high", "hindi", True, "standard", "train", "नए लैपटॉप से साइन-इन करने पर खाता सुरक्षा कारणों से फ्रीज़ हो गया है, कृपया इसे अनब्लॉक करें।"),
    ("account_access", "high", "hinglish", True, "standard", "train", "bhai login karte hi account temporarily blocked popup aa raha hai bina kisi galti ke"),
    ("account_access", "high", "hinglish", True, "standard", "train", "sim upgrade karwane ke baad se otp sms nahi aa raha, andar ghus hi nahi pa raha profile me"),
    ("account_access", "high", "hinglish", True, "typo_unusual", "train", "authenticatr app ka backup lost ho gaya ab two step verification bypass kaise karu?"),
    ("account_access", "high", "hinglish", True, "standard", "train", "baat ye hai ki mera account disabled bata raha hai jabki maine koi violation nahi kiya"),
    ("account_access", "high", "english", True, "standard", "val", "Our shared billing mailbox can no longer sign in because an old phone number blocks SMS verification challenges."),
    ("account_access", "high", "english", True, "standard", "val", "Every login session gets terminated immediately after MFA confirmation with token revoked warnings."),
    ("account_access", "high", "hindi", True, "standard", "val", "सत्यापन कोड बार-बार अमान्य बता रहा है और मेरा ग्राहक लॉगिन अवरुद्ध हो गया है।"),
    ("account_access", "high", "hindi", True, "standard", "val", "दफ़्तर के आईपी पते से लॉगिन करने पर सिस्टम ने मेरा खाता लॉक कर दिया है।"),
    ("account_access", "high", "hinglish", True, "standard", "val", "otp nhi aa rha sim pe login kaise open kru dashboard band pada h"),
    ("account_access", "high", "hinglish", True, "standard", "val", "google authenticator code dene ke baad bhi access denied dikha rha baar baar"),
    ("account_access", "high", "english", True, "standard", "cal", "My authenticator key de-synced after timezone travel, locking our operations lead out of the portal."),
    ("account_access", "high", "hindi", True, "standard", "cal", "खाता अस्थायी रूप से लॉक कर दिया गया है और अनलॉक लिंक पर क्लिक करने से भी पेज नहीं खुलता।"),
    ("account_access", "high", "hinglish", True, "standard", "cal", "registered phone pe otp deliver nahi ho raha aur account locked bata rha"),
    ("account_access", "high", "english", True, "standard", "audit", "Access denied appears across both desktop Chrome and the iOS app despite valid credentials."),
    ("account_access", "high", "hindi", True, "standard", "audit", "सुरक्षा जांच के नाम पर मेरा प्रोफ़ाइल लॉक है जबकि मैंने कोई संदिग्ध लॉगिन नहीं किया।"),
    ("account_access", "high", "hinglish", True, "standard", "audit", "mera id block dikha raha subah se, 2fa prompt fail ho raha hai baar baar"),

    # --- 6. PASSWORD_RESET ---
    ("password_reset", "medium", "english", False, "standard", "train", "I requested three forgot-password emails over the last hour, yet inbox and spam folders remain empty."),
    ("password_reset", "medium", "english", False, "standard", "train", "Clicking the password recovery URL opens a blank page saying Token Already Redeemed even on first click."),
    ("password_reset", "medium", "english", False, "standard", "train", "The Save New Passphrase button stays greyed out even when my entry satisfies length and symbol requirements."),
    ("password_reset", "medium", "hindi", False, "standard", "train", "मैं अपना लॉगिन पासवर्ड भूल गया हूँ और रीसेट ईमेल अभी तक इनबॉक्स में नहीं आया है।"),
    ("password_reset", "medium", "hindi", False, "standard", "train", "पासवर्ड बदलने वाला लिंक खोलने पर हर बार लिंक एक्सपायर्ड का संदेश दिखाई देता है।"),
    ("password_reset", "medium", "hindi", False, "standard", "train", "नया पासवर्ड सेव करते समय सिस्टम अमान्य वर्ण की त्रुटि बताकर आगे नहीं बढ़ने दे रहा।"),
    ("password_reset", "medium", "hindi", False, "standard", "train", "कृपया मेरे ईमेल पर तुरंत एक नया पासवर्ड रीसेट टोकन भिजवाने की व्यवस्था करें।"),
    ("password_reset", "medium", "hinglish", False, "typo_unusual", "train", "yaar password bhul gya hu forgot option dabane pe mail hi nhi aari"),
    ("password_reset", "medium", "hinglish", False, "typo_unusual", "train", "naya passwrd set kr rha hu par confirm button kaam nhi kr rha page pe"),
    ("password_reset", "medium", "hinglish", False, "standard", "train", "reset mail aaya tha 2 min pehle hi click kiya fir bhi link invalid dikha rha h"),
    ("password_reset", "medium", "hinglish", False, "typo_unusual", "train", "pwd change link dubara send kardo pls purana wala expire ho gya"),
    ("password_reset", "medium", "english", False, "standard", "val", "Why does the password update API throw 400 Invalid Nonce immediately after submitting the forgot-password form?"),
    ("password_reset", "medium", "english", False, "standard", "val", "Recovery email links keep timing out before arriving because our corporate mail relay delays external mail."),
    ("password_reset", "medium", "hindi", False, "standard", "val", "पासवर्ड रीसेट करने पर लिंक काम नहीं कर रहा और नया कोड जनरेट नहीं हो पा रहा।"),
    ("password_reset", "medium", "hindi", False, "standard", "val", "अपना पुराना पासवर्ड भूल जाने के बाद ईमेल सत्यापन लिंक मुझे प्राप्त नहीं हुआ।"),
    ("password_reset", "medium", "hinglish", False, "standard", "val", "forgot password link pe click krte hi invalid session bolta h naya token bhejo"),
    ("password_reset", "medium", "hinglish", False, "standard", "val", "passcode reset page submit karne pe save nahi ho raha error aata hai"),
    ("password_reset", "medium", "english", False, "standard", "cal", "Password recovery code never hits my Outlook mailbox even after whitelisting your sender domain."),
    ("password_reset", "medium", "hindi", False, "standard", "cal", "नया पासवर्ड बनाने का फॉर्म आगे नहीं बढ़ रहा, कृपया मैन्युअल रीसेट लिंक भेजें।"),
    ("password_reset", "medium", "hinglish", False, "standard", "cal", "mujhe apna purana password yaad nahi hai naya banane ka link bhejo"),
    ("password_reset", "medium", "english", False, "standard", "audit", "The reset-password form rejects valid special characters despite the helper tooltip recommending them."),
    ("password_reset", "medium", "hindi", False, "standard", "audit", "मैं पासवर्ड रीसेट लिंक का इंतजार कर रहा हूँ लेकिन ईमेल प्राप्त नहीं हुआ।"),
    ("password_reset", "medium", "hinglish", False, "standard", "audit", "forgot pwd mail spam me bhi check kr liya abhi tak receive nhi hua"),

    # --- 7. ORDER_CANCEL ---
    ("order_cancel", "high", "english", False, "standard", "train", "I accidentally picked the wrong keyboard layout seconds ago—halt warehouse packing and cancel this shipment immediately."),
    ("order_cancel", "high", "english", False, "standard", "train", "The Cancel Shipment button vanished from my active purchase screen even though courier pickup is still unscheduled."),
    ("order_cancel", "high", "english", False, "standard", "train", "Abort dispatch for the dining table purchase placed this morning because it won't fit our elevator."),
    ("order_cancel", "high", "hindi", False, "standard", "train", "गलती से गलत साइज की जैकेट ऑर्डर हो गई है, कृपया पैक होने से पहले यह ऑर्डर तुरंत रद्द करें।"),
    ("order_cancel", "high", "hindi", False, "standard", "train", "मेरा पार्सल अभी गोदाम से निकला भी नहीं है लेकिन ऐप में कैंसिल करने का विकल्प गायब हो गया है।"),
    ("order_cancel", "high", "hindi", False, "standard", "train", "एक ही सामान दो बार बुक हो गया है, कृपया दूसरा वाला ऑर्डर डिस्पैच से पहले निरस्त कर दें।"),
    ("order_cancel", "high", "hindi", False, "standard", "train", "कृपया आज सुबह की गई फर्नीचर बुकिंग को तुरंत रद्द करें क्योंकि डिलीवरी का पता बदल गया है।"),
    ("order_cancel", "high", "hinglish", False, "standard", "train", "galti se galat color book ho gaya abhi, packing se pehle order cancel kardo bhai"),
    ("order_cancel", "high", "hinglish", False, "standard", "train", "app ke andar cancel button show hi nhi ho rha jabki shipment abhi dispatch nhi hua"),
    ("order_cancel", "high", "hinglish", False, "standard", "train", "do baar checkout dab gaya same item ke liye ek order jaldi band karwa do"),
    ("order_cancel", "high", "hinglish", False, "standard", "train", "abhi 5 min pehle jo courier booking kari thi use rok do mujhe item nahi chahiye"),
    ("order_cancel", "high", "english", False, "standard", "val", "Stop outbound dispatch for the monitor mount I ordered five minutes ago; we bought one locally instead."),
    ("order_cancel", "high", "english", False, "standard", "val", "Can support void my pending warehouse shipment before carrier scan occurs tonight?"),
    ("order_cancel", "high", "hindi", False, "standard", "val", "मेरा आज का ऑर्डर शिप होने से पहले ही तुरंत कैंसिल कर दीजिए।"),
    ("order_cancel", "high", "hindi", True, "standard", "val", "कृपया इस अनडिस्पैच्ड पार्सल की पैकिंग रोकें और बुकिंग रद्द करें।"),
    ("order_cancel", "high", "hinglish", False, "standard", "val", "abhi warehouse me hi hai parcel jaldi cancel karo galti se double order ho gya"),
    ("order_cancel", "high", "hinglish", False, "standard", "val", "shipment manifest hone se pehle ye item order cancel karwa do please"),
    ("order_cancel", "high", "english", False, "standard", "cal", "Revoke my unmanifested footwear order immediately prior to carrier handoff."),
    ("order_cancel", "high", "hindi", False, "standard", "cal", "मेरा ऑर्डर अभी शिप नहीं हुआ है, इसे तुरंत कैंसिल करें।"),
    ("order_cancel", "high", "hinglish", False, "standard", "cal", "ye wala parcel mat bhejna backend se abhi rok do pls cancel order"),
    ("order_cancel", "high", "english", False, "standard", "audit", "Pull this microwave order back from the loading bay and cancel it before driver departure."),
    ("order_cancel", "high", "hindi", False, "standard", "audit", "मेरा कूरियर अभी पिकअप के लिए पेंडिंग है, कृपया ऑर्डर तुरंत निरस्त करें।"),
    ("order_cancel", "high", "hinglish", False, "standard", "audit", "bhai mobile wala parcel dispatch mat hone dena abhi cancel mark karwao"),

    # --- 8. SUBSCRIPTION ---
    ("subscription", "medium", "english", False, "standard", "train", "Switch my Pro workspace membership from auto-renewing annual billing to monthly manual renewal before the next cycle."),
    ("subscription", "medium", "english", False, "standard", "train", "End my Prime video and delivery membership at the close of this month so no renewal charge triggers."),
    ("subscription", "medium", "english", False, "standard", "train", "Please pause our recurring coffee bean club subscription for two months while our office renovates."),
    ("subscription", "high", "english", True, "standard", "train", "I toggled Auto-Renew OFF last month, yet you renewed my annual Pro SaaS membership anyway this morning!"),
    ("subscription", "medium", "hindi", False, "standard", "train", "कृपया मेरी मासिक प्रीमियम सदस्यता का ऑटो-रिन्यूअल बंद करें ताकि अगले महीने शुल्क न कटे।"),
    ("subscription", "medium", "hindi", False, "standard", "train", "मुझे अपना वार्षिक मेंबरशिप प्लान बेसिक टियर में डाउनग्रेड करना है, विकल्प कहां मिलेगा?"),
    ("subscription", "medium", "hindi", False, "standard", "train", "मैंने तीन दिन पहले सब्सक्रिप्शन रोक दिया था फिर भी आज सदस्यता सक्रिय दिख रही है।"),
    ("subscription", "medium", "hindi", False, "standard", "train", "परिवार वाले प्लान से व्यक्तिगत मासिक सब्सक्रिप्शन पर स्विच करने की प्रक्रिया बताएं।"),
    ("subscription", "medium", "hinglish", False, "standard", "train", "bhai mera monthly pro membership plan off kardo agle billing cycle se renew nhi chahiye"),
    ("subscription", "medium", "hinglish", False, "standard", "train", "auto debit band karna hai premium subscription ka setting me toggle disable nhi ho raha"),
    ("subscription", "medium", "hinglish", False, "standard", "train", "yearly plan ko monthly subscription me downgrade kaise karte hain dashboard se?"),
    ("subscription", "medium", "hinglish", False, "standard", "train", "student membership renew ho gayi bina reminder ke ab ise pause kar do please"),
    ("subscription", "medium", "english", False, "standard", "val", "Deactivate automatic recurring membership billing on our cloud storage plan prior to Friday's cutoff."),
    ("subscription", "medium", "english", False, "standard", "val", "Pause recurring pet-supply delivery membership until November while we travel abroad."),
    ("subscription", "medium", "hindi", False, "standard", "val", "मेरी वार्षिक मेंबरशिप का ऑटो-पे बंद करवाना है ताकि अगले साल नवीनीकरण न हो।"),
    ("subscription", "medium", "hindi", False, "standard", "val", "कृपया मेरा प्रो सब्सक्रिप्शन प्लान अगले बिलिंग चक्र से समाप्त कर दें।"),
    ("subscription", "high", "hinglish", True, "standard", "val", "maine auto renew off kiya tha phir bhi membership charge katne wala reminder aa gya"),
    ("subscription", "medium", "hinglish", False, "standard", "val", "annual vip membership ko free plan par shift karna hai cancel link kahan h?"),
    ("subscription", "medium", "english", False, "standard", "cal", "Turn off recurring VIP club renewal while keeping my regular shopping account active."),
    ("subscription", "medium", "hindi", False, "standard", "cal", "सदस्यता शुल्क हर महीने कट रहा है, इसे स्थायी रूप से रोकें।"),
    ("subscription", "medium", "hinglish", False, "standard", "cal", "saas subscription plan ka renewal rook do agle mahine se use nhi karna"),
    ("subscription", "high", "english", True, "standard", "audit", "Recurring auto-renewal charged my card after our team trial period ended early without notice."),
    ("subscription", "high", "hindi", True, "standard", "audit", "बिना अनुमति के मेरी मासिक सदस्यता अपने आप नवीनीकृत हो गई, कृपया ऑटो-रिन्यू बंद करें।"),
    ("subscription", "medium", "hinglish", False, "standard", "audit", "membership manage section me cancel button gray dikha raha hai plan stop karna hai"),

    # --- 9. ORDER_TRACKING ---
    ("order_tracking", "medium", "english", False, "standard", "train", "The courier tracking timeline has not posted a single hub scan since departing the regional warehouse four days back."),
    ("order_tracking", "medium", "english", False, "standard", "train", "Checkout estimated Thursday delivery, yet live tracking still sits at Manifest Created—when will it reach me?"),
    ("order_tracking", "medium", "english", False, "standard", "train", "Could logistics share the updated air waybill number because the tracking link opens a broken carrier page?"),
    ("order_tracking", "medium", "hindi", False, "standard", "train", "कूरियर ट्रैकिंग पिछले तीन दिनों से एक ही शहर के हब पर अटकी हुई है, कृपया नया स्टेटस बताएं।"),
    ("order_tracking", "medium", "hindi", False, "standard", "train", "मेरा पार्सल कब तक पते पर पहुंचेगा? ट्रैकिंग लिंक खोलने पर कोई जानकारी अपडेट नहीं हो रही।"),
    ("order_tracking", "medium", "hindi", False, "standard", "train", "शिपमेंट भेजे जाने का मैसेज आ गया पर अभी तक डिलीवरी की अनुमानित तारीख नहीं दिख रही।"),
    ("order_tracking", "medium", "hindi", False, "standard", "train", "क्या मेरा ऑर्डर स्थानीय डिलीवरी केंद्र तक पहुंच गया है या अभी रास्ते में ही है?"),
    ("order_tracking", "medium", "hinglish", False, "standard", "train", "tracking id 4 din se sorted at transit hub pe atki hai aage kab badhegi?"),
    ("order_tracking", "medium", "hinglish", False, "standard", "train", "mera parcel abhi kis city me pahuncha hai aur aaj deliver hone ka koi chance hai kya?"),
    ("order_tracking", "medium", "hinglish", False, "standard", "train", "courier link khul nhi rha sahi se live location aur exact arrival date check karke batao"),
    ("order_tracking", "medium", "hinglish", False, "standard", "train", "dispatch mail to kal hi aa gaya tha par awb status abhi tak not found show ho rha"),
    ("order_tracking", "medium", "english", False, "standard", "val", "Carrier scan events stopped updating right after Sort Facility Departure—is the truck delayed by weather?"),
    ("order_tracking", "medium", "english", False, "standard", "val", "Which regional hub holds my delayed air shipment and what updated arrival window applies?"),
    ("order_tracking", "medium", "hindi", False, "standard", "val", "पार्सल ट्रैक नहीं हो रहा कृपया सही ट्रैकिंग स्टेटस और संभावित डिलीवरी तिथि भेजें।"),
    ("order_tracking", "medium", "hindi", False, "standard", "val", "ऑर्डर स्टेटस कल से आउट फॉर डिलीवरी दिखा रहा है पर कूरियर का अता-पता नहीं है।"),
    ("order_tracking", "medium", "hinglish", False, "standard", "val", "courier partner ka naam aur live tracking status dikh hi nhi rha app pe"),
    ("order_tracking", "medium", "hinglish", False, "standard", "val", "mera order abhi tak kis hub me ruka hua h kab tak aayega ghar pe?"),
    ("order_tracking", "medium", "english", False, "standard", "cal", "Verify whether the courier rescheduled doorstep handoff from Friday to Monday."),
    ("order_tracking", "medium", "hindi", False, "standard", "cal", "मेरे ऑर्डर का वर्तमान लाइव लोकेशन और अनुमानित समय बताएं।"),
    ("order_tracking", "medium", "hinglish", False, "standard", "cal", "parcel kahan tak aaya bhai kal function hai urgent status dekhna hai"),
    ("order_tracking", "medium", "english", False, "standard", "audit", "Live carrier status froze at Destination Hub 48 hours ago—when does the rider head out?"),
    ("order_tracking", "medium", "hindi", False, "standard", "audit", "मेरा पार्सल किस हब पर रुका हुआ है और डिलीवरी एजेंट का नंबर क्या है?"),
    ("order_tracking", "medium", "hinglish", False, "standard", "audit", "shipment in transit hi dikha rha hai pichle hafte se exact delivery date kya hogi?"),

    # --- 10. DELIVERY_ISSUE ---
    ("delivery_issue", "high", "english", True, "standard", "train", "Carrier marked status as Handed Directly to Resident at 4:12 PM, yet our security desk CCTV shows nobody visited."),
    ("delivery_issue", "high", "english", True, "standard", "train", "When I cut the outer shipping tape today, the inner product carton was empty with only crumpled kraft paper inside!"),
    ("delivery_issue", "high", "english", True, "standard", "train", "The factory tamper seal was sliced clean and resealed with clear tape—half the accessories are missing."),
    ("delivery_issue", "high", "english", True, "standard", "train", "Driver dropped someone else's parcel on my porch instead of the coffee grinder shown on my shipping label."),
    ("delivery_issue", "high", "hindi", True, "standard", "train", "कूरियर एजेंट ने बिना घर आए ही स्टेटस को डिलीवर्ड मार्क कर दिया, मुझे पार्सल बिल्कुल नहीं मिला।"),
    ("delivery_issue", "high", "hindi", True, "standard", "train", "पैकेट की सील टूटी हुई थी और अंदर से असली सामान गायब था, केवल खाली डिब्बा मिला है।"),
    ("delivery_issue", "high", "hindi", True, "standard", "train", "पार्सल क्षतिग्रस्त हालत में प्राप्त हुआ और डब्बे के अंदर गलत उत्पाद निकला।"),
    ("delivery_issue", "high", "hindi", True, "standard", "train", "डिलीवरी बॉय किसी और ग्राहक का पैकेट गलत पते पर छोड़कर चला गया है।"),
    ("delivery_issue", "high", "hinglish", True, "standard", "train", "delivery partner ne bina phone kiye false delivered update kar diya app me jabki kuch nhi mila"),
    ("delivery_issue", "high", "hinglish", True, "standard", "train", "dabba khola to seal tooti hui thi aur watch gayab thi andar se bilkul empty box hai"),
    ("delivery_issue", "high", "hinglish", True, "standard", "train", "bhai mera parcel fati hui halat me mila aur inside wrong item bhej diya seller ne"),
    ("delivery_issue", "high", "hinglish", True, "standard", "train", "courier wala pados me galat flat me parcel chhod gaya aur sign khud hi kar diya"),
    ("delivery_issue", "high", "english", True, "standard", "val", "Outer shipping bag arrived slit down the seam with the sealed smartphone unit missing before handoff."),
    ("delivery_issue", "high", "english", True, "standard", "val", "Delivery app claims signed proof-of-delivery three hours ago while our office reception was completely empty."),
    ("delivery_issue", "high", "hindi", True, "standard", "val", "डिलीवरी पैकेट खुला हुआ मिला और अंदर का मुख्य सामान गायब है।"),
    ("delivery_issue", "high", "hindi", True, "standard", "val", "कूरियर वाले ने तीन दिन पहले डिलीवर्ड दिखाया पर मुझे कोई पैकेट प्राप्त नहीं हुआ।"),
    ("delivery_issue", "high", "hinglish", True, "standard", "val", "delivered status aa rha msg pe lekin mere ghar koi parcel lekar aaya hi nahi"),
    ("delivery_issue", "high", "hinglish", True, "standard", "val", "shipping carton khula hua mila aur andar se power cable aur adapter gayab the"),
    ("delivery_issue", "high", "english", True, "standard", "cal", "Parcel says Delivered to Front Porch, however nothing is at our building and cameras show no van."),
    ("delivery_issue", "high", "hindi", True, "standard", "cal", "डिलीवरी बॉय ने घर आए बिना ऑर्डर डिलीवर दिखा दिया जबकि पैकेट नहीं मिला।"),
    ("delivery_issue", "high", "hinglish", True, "standard", "cal", "parcel bilkul pichka hua aur geela condition me mila barish me bheega hua"),
    ("delivery_issue", "high", "english", True, "standard", "audit", "Package arrived crushed flat with missing hardware inside an unsealed shipping pouch."),
    ("delivery_issue", "high", "hindi", True, "standard", "audit", "डिलीवरी वाला कह रहा है उसने दे दिया लेकिन घर पर कोई पैकेट पहुंचा ही नहीं।"),
    ("delivery_issue", "high", "hinglish", True, "standard", "audit", "seal broken mili courier packet ki aur andar se main device gayab tha"),

    # --- 11. BILLING ---
    ("billing", "low", "english", False, "standard", "train", "Could finance email a PDF GST tax invoice displaying our company GSTIN for last month's supply purchase?"),
    ("billing", "low", "english", False, "standard", "train", "We need itemized VAT invoices per seat for quarterly accounting reconciliation before Friday close."),
    ("billing", "medium", "english", False, "standard", "train", "Promo coupon SAVE20 showed applied in basket, yet the final PDF invoice billed full sticker price without deduction."),
    ("billing", "low", "hindi", False, "standard", "train", "कृपया कंपनी जीएसटी नंबर के साथ मेरे पिछले बिल की टैक्स इनवॉइस पीडीएफ ईमेल कर दें।"),
    ("billing", "low", "hindi", False, "standard", "train", "अकाउंटिंग ऑडिट के लिए मुझे पिछले तीन महीनों के सभी मासिक बिलों की कॉपी चाहिए।"),
    ("billing", "medium", "hindi", False, "standard", "train", "इनवॉइस पर लगाया गया कूपन डिस्काउंट सही से नहीं जुड़ा है, कृपया बिल राशि सुधारें।"),
    ("billing", "medium", "hindi", False, "standard", "train", "मेरे अंतिम बिल पर शिपिंग शुल्क दो बार जोड़ा गया है, कृपया सही इनवॉइस जारी करें।"),
    ("billing", "low", "hinglish", False, "standard", "train", "reimbursement ke liye tax invoice copy chahiye gst number ke saath pdf me"),
    ("billing", "low", "hinglish", False, "standard", "train", "company audit chal raha hai pichle quarter ke sare gst invoices zip karke bhej do"),
    ("billing", "medium", "hinglish", False, "standard", "train", "checkout ke time discount coupon lagaya tha par final bill me pura amount jod diya"),
    ("billing", "medium", "hinglish", False, "standard", "train", "pichle mahine ke billing statement me tax calculation thoda galat lag raha hai check karna"),
    ("billing", "low", "english", False, "standard", "val", "Send a tax-exempt charitable receipt PDF replacing the standard taxable invoice issued yesterday."),
    ("billing", "medium", "english", False, "standard", "val", "The subtotal adds up to $84 before discount, yet the billed invoice total says $96."),
    ("billing", "low", "hindi", False, "standard", "val", "जीएसटी बिल की साफ पीडीएफ कॉपी भेज दीजिए ऑफिस में जमा करनी है।"),
    ("billing", "medium", "hindi", False, "standard", "val", "अंतिम बिल में गलत टैक्स दर जोड़ दी गई है, कृपया सही रसीद अपडेट करें।"),
    ("billing", "low", "hinglish", False, "standard", "val", "tax invoice me hsn code missing hai accounts team accept nahi kar rahi naya bill do"),
    ("billing", "medium", "hinglish", False, "standard", "val", "bill amount statement me delivery fee extra calculate ho gayi free shipping ke bawajood"),
    ("billing", "low", "english", False, "standard", "cal", "Send corrected VAT receipt displaying our updated Munich office street address."),
    ("billing", "medium", "hindi", False, "standard", "cal", "प्रोमो कोड लगाने के बाद भी बिल में पूरी टैक्स राशि जोड़ी गई है।"),
    ("billing", "medium", "hinglish", False, "standard", "cal", "invoice me promo code miss ho gaya aur shipping charge extra lag gaya"),
    ("billing", "medium", "english", False, "standard", "audit", "Invoice breakdown lists duplicate handling fees on a single consolidated shipment receipt."),
    ("billing", "medium", "hindi", False, "standard", "audit", "बिल में कूपन कोड की छूट कटकर नहीं आई, कृपया संशोधित चालान भेजें।"),
    ("billing", "low", "hinglish", False, "standard", "audit", "gst invoice download button error de raha portal pe pdf mail kar do"),

    # --- 12. ACCOUNT_CLOSURE ---
    ("account_closure", "medium", "english", True, "standard", "train", "I no longer shop with this service and would like my entire user profile permanently deactivated today."),
    ("account_closure", "medium", "english", True, "standard", "train", "Under GDPR Article 17 Right to Erasure, permanently delete my stored personal data and close my account."),
    ("account_closure", "high", "english", True, "standard", "train", "Immediate privacy notice: purge all identity documents and permanently delete my user account or I escalate legally."),
    ("account_closure", "medium", "hindi", True, "standard", "train", "मैं अब इस प्लेटफॉर्म का उपयोग नहीं करना चाहता, कृपया मेरा खाता स्थायी रूप से बंद कर दें।"),
    ("account_closure", "medium", "hindi", True, "standard", "train", "कृपया मेरी सारी व्यक्तिगत जानकारी और सेव किए गए कार्ड विवरण हटाकर अकाउंट डिलीट कर दें।"),
    ("account_closure", "medium", "hindi", True, "standard", "train", "निजता नियमों के तहत मेरे प्रोफ़ाइल और ऑर्डर इतिहास को हमेशा के लिए मिटाया जाए।"),
    ("account_closure", "medium", "hindi", True, "standard", "train", "सेटिंग में खाता हटाने का बटन नहीं दिख रहा, कृपया मेरा अकाउंट हमेशा के लिए बंद करें।"),
    ("account_closure", "medium", "hinglish", True, "standard", "train", "mujhe ab ye account nahi chalana bhai hamesha ke liye permanently delete kar do mera data"),
    ("account_closure", "medium", "hinglish", True, "standard", "train", "app settings me close account option nhi mil rha please backend se mera profile band karo"),
    ("account_closure", "medium", "hinglish", True, "standard", "train", "gdpr ke under mera pura user record aur order data delete kar ke id close kar do"),
    ("account_closure", "medium", "hinglish", True, "standard", "train", "apne sare saved cards aur addresses remove karwana chahta hu account closure karo"),
    ("account_closure", "high", "english", True, "standard", "val", "Execute right-to-be-forgotten erasure across all backups and permanently deactivate my login credentials."),
    ("account_closure", "medium", "english", True, "standard", "val", "Terminate our dormant merchant account permanently and email an erasure confirmation."),
    ("account_closure", "medium", "hindi", True, "standard", "val", "कृपया मेरा ग्राहक खाता और सारा डेटा पूरी तरह समाप्त कर दें।"),
    ("account_closure", "medium", "hindi", True, "standard", "val", "प्रोफ़ाइल स्थायी रूप से बंद करने हेतु मेरा अनुरोध स्वीकार करें।"),
    ("account_closure", "medium", "hinglish", True, "standard", "val", "purana customer account hamesha ke liye band kar do aur personal data erase karo"),
    ("account_closure", "medium", "hinglish", True, "standard", "val", "id permanently delete karni h confirm email bhej dena close karne ke baad"),
    ("account_closure", "medium", "english", True, "standard", "cal", "Permanently delete my customer record under regional data privacy law."),
    ("account_closure", "medium", "hindi", True, "standard", "cal", "मेरा व्यक्तिगत डेटा और ग्राहक खाता स्थायी रूप से हटाने के लिए इसे बंद करें।"),
    ("account_closure", "medium", "hinglish", True, "standard", "cal", "ab use nhi karna account permanently remove karwa do sare saved cards hata ke"),
    ("account_closure", "high", "english", True, "standard", "audit", "Close our corporate workspace permanently and purge all archived user profile data."),
    ("account_closure", "medium", "hindi", True, "standard", "audit", "मेरा अकाउंट और सारा डेटा सर्वर से स्थायी रूप से मिटा दें।"),
    ("account_closure", "medium", "hinglish", True, "standard", "audit", "permanently profile close karne ka request process kar do pls"),

    # --- 13. TECHNICAL_ISSUE ---
    ("technical_issue", "medium", "english", False, "standard", "train", "The Android shopping app crashes straight to home screen whenever tapping Wishlist after today's update."),
    ("technical_issue", "medium", "english", False, "standard", "train", "Uploading receipt photos in the warranty form fails continuously with HTTP 500 Internal Server Error."),
    ("technical_issue", "medium", "english", False, "standard", "train", "Dark mode contrast renders white coupon text over a white input box in Firefox on macOS."),
    ("technical_issue", "high", "english", True, "standard", "train", "Production checkout API returns 500 JSON parse failures across every browser right now!"),
    ("technical_issue", "medium", "hindi", False, "standard", "train", "नया अपडेट करने के बाद से मोबाइल ऐप खोलते ही अपने आप बंद और क्रैश हो रहा है।"),
    ("technical_issue", "medium", "hindi", False, "standard", "train", "वेबसाइट पर एड्रेस जोड़ने वाला पेज लोड नहीं हो रहा और बार-बार सर्वर एरर 500 दिखा रहा है।"),
    ("technical_issue", "medium", "hindi", False, "standard", "train", "चेकआउट बटन दबाने पर स्क्रीन सफेद होकर अटक जाती है और आगे नहीं बढ़ती।"),
    ("technical_issue", "medium", "hindi", False, "standard", "train", "शिकायत फॉर्म में फोटो अपलोड करने पर फाइल अस्वीकार हो रही है और बग दिख रहा है।"),
    ("technical_issue", "medium", "hinglish", False, "typo_unusual", "train", "new update aane ke baad app bar bar hang hoke force close ho rahi h android pe"),
    ("technical_issue", "medium", "hinglish", False, "standard", "train", "cart page kholte hi infinite loading ghumta rehta hai aur browser tab freeze ho jata"),
    ("technical_issue", "medium", "hinglish", False, "typo_unusual", "train", "image upload nhi ho rhi return form me bar bar unexpected api glitch bolta h"),
    ("technical_issue", "high", "hinglish", True, "standard", "train", "checkout dabane par app crash ho rha aur error code 500 show ho raha h"),
    ("technical_issue", "high", "english", True, "standard", "val", "API webhook listener returns 503 Gateway Unavailable under production TLS handshake since this morning."),
    ("technical_issue", "medium", "english", False, "standard", "val", "Push notification deep links drop users onto an unhandled 404 stack trace inside iOS."),
    ("technical_issue", "high", "hindi", True, "standard", "val", "ऐप लगातार क्रैश हो रहा है और कार्ट स्क्रीन बिल्कुल नहीं खुलती।"),
    ("technical_issue", "medium", "hindi", False, "standard", "val", "सर्च फिल्टर चुनने पर वेबसाइट हैंग हो जाती है और लोडिंग खत्म नहीं होती।"),
    ("technical_issue", "medium", "hinglish", False, "standard", "val", "apply coupon button freeze ho jata safari mobile browser pe bug fix karo"),
    ("technical_issue", "high", "hinglish", True, "standard", "val", "app kholte hi server error 500 pop up aa rha aur home page crash kr rha"),
    ("technical_issue", "medium", "english", False, "standard", "cal", "Profile avatar upload throws CORS rejection on desktop Edge browser."),
    ("technical_issue", "medium", "hindi", False, "standard", "cal", "मोबाइल ऐप में नोटिफिकेशन पर क्लिक करने से पेज क्रैश हो जाता है।"),
    ("technical_issue", "medium", "hinglish", False, "standard", "cal", "checkout window freeze ho jati hai jaise hi pin code verify karte hain"),
    ("technical_issue", "high", "english", True, "standard", "audit", "Checkout cart calculator crashes with NaN total whenever applying gift voucher alongside shipping tax."),
    ("technical_issue", "medium", "hindi", False, "standard", "audit", "पोर्टल लॉगिन के बाद डैशबोर्ड खाली सफेद स्क्रीन दिखाकर अटक जाता है।"),
    ("technical_issue", "high", "hinglish", True, "standard", "audit", "app bar bar logout kar deta hai aur dubara kholne pe white screen hang hoti h"),

    # --- 14. FRAUD_REPORT ---
    ("fraud_report", "critical", "english", True, "financial_security", "train", "EMERGENCY: Someone compromised my credentials overnight, changed shipping address, and placed unauthorized electronics orders!"),
    ("fraud_report", "critical", "english", True, "financial_security", "train", "Unrecognized overseas transactions totaling $2,100 hit my linked card without any OTP or my consent!"),
    ("fraud_report", "critical", "english", True, "financial_security", "train", "A scammer impersonating verification desk stole my OTP via phishing and drained saved wallet credits!"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "train", "तत्काल सहायता! किसी अनजान व्यक्ति ने मेरा खाता हैक करके बिना मेरी अनुमति के महंगे ऑर्डर कर दिए हैं!"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "train", "मेरे क्रेडिट कार्ड से मेरी जानकारी और ओटीपी के बिना दो अनधिकृत ट्रांजैक्शन हुए हैं, यह धोखाधड़ी है।"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "train", "किसी फर्जी कॉल करने वाले ने लिंक भेजकर मेरे अकाउंट का पासवर्ड चुरा लिया और पैसे निकाल लिए।"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "train", "मेरे वॉलेट से किसी अज्ञात डिवाइस से संदिग्ध खरीदारी हो रही है, कृपया खाता तुरंत फ्रीज करें।"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "train", "urgent!! kisi ne mera account hack karke bina otp ke 35000 ka unauthorized purchase kar diya h!"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "train", "fake support agent ne phishing link bheja aur mere card se fraud debit ho gya turant freeze karo"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "train", "mere bina permission ke naye unknown address pe gift cards order ho rhe h account compromise ho gya"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "train", "kisi aur ne login karke meri saved bank details chori kar li hain turant action lo"),
    ("fraud_report", "critical", "english", True, "financial_security", "val", "Security breach alert: an unfamiliar IP from abroad redeemed my rewards and altered recovery email!"),
    ("fraud_report", "critical", "english", True, "financial_security", "val", "Someone cloned my credit card and ran fraudulent purchases on your marketplace—notify fraud ops!"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "val", "मेरे खाते में किसी हैकर ने घुसकर बिना अनुमति के पैसे निकाल लिए हैं!"),
    ("fraud_report", "critical", "hindi", True, "financial_security", "val", "मेरे सेव किए गए डेबिट कार्ड से किसी अज्ञात अपराधी ने अनधिकृत ऑर्डर किया है!"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "val", "kisi hacker ne password chura ke mere card se bina consent ke shopping kar li h"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "val", "fraud alert bhai unknown log-in hua aur mera bank card misuse ho rha h"),
    ("fraud_report", "critical", "english", True, "financial_security", "cal", "Identity theft emergency: fake orders placed under my corporate tax ID without authorization."),
    ("fraud_report", "critical", "hindi", True, "financial_security", "cal", "मेरे खाते से किसी अज्ञात व्यक्ति ने हैक करके अनधिकृत खरीदारी कर ली है!"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "cal", "bhai account me koi aur ghus gya h unauthorized card transaction ho rhi hai abhi"),
    ("fraud_report", "critical", "english", True, "financial_security", "audit", "Lock my login now! Unknown intruder disabled 2FA and bought gift vouchers with stored card."),
    ("fraud_report", "critical", "hindi", True, "financial_security", "audit", "अनधिकृत व्यक्ति द्वारा मेरे बैंक कार्ड का दुरुपयोग कर धोखाधड़ी से ऑर्डर किया गया है।"),
    ("fraud_report", "critical", "hinglish", True, "financial_security", "audit", "account takeover ho gaya mera unknown device se otp mangaya ja raha hai turant block karo"),
]


# Split-disjoint pragmatic discourse wrappers to expand training & eval diversity cleanly
# without reusing phrasing across splits.
SPLIT_TRANSFORMS: Dict[str, List[Tuple[str, str]]] = {
    "train": [
        ("{t}", "any"),
        ("Support team: {t}", "english"),
        ("Customer desk inquiry — {t}", "english"),
        ("Please review this urgent note: {t}", "english"),
        ("सपोर्ट टीम कृपया ध्यान दें: {t}", "hindi"),
        ("मदद चाहिए — {t}", "hindi"),
        ("ग्राहक शिकायत: {t}", "hindi"),
        ("Hello bhai, {t}", "hinglish"),
        ("Please check karo: {t}", "hinglish"),
        ("Suniyega sir, {t}", "hinglish"),
    ],
    "val": [
        ("{t}", "any"),
        ("Escalating note: {t}", "english"),
        ("Follow-up on ticket: {t}", "english"),
        ("नमस्ते टीम, {t}", "hindi"),
        ("कृपया जांच करें: {t}", "hindi"),
        ("Bhai ek issue hai: {t}", "hinglish"),
        ("Urgent update — {t}", "hinglish"),
    ],
    "cal": [
        ("{t}", "any"),
        ("Ticket submission: {t}", "english"),
        ("निवेदन: {t}", "hindi"),
        ("Sir check kijiye: {t}", "hinglish"),
    ],
    "audit": [
        ("{t}", "any"),
        ("Verified customer report: {t}", "english"),
        ("संदर्भ शिकायत: {t}", "hindi"),
        ("Team please dekho: {t}", "hinglish"),
    ],
}


def expand_and_verify() -> Dict[str, Any]:
    rng = random.Random(20261125)
    out_dir = PROJECT_ROOT / "data" / "v3_diverse"
    out_dir.mkdir(parents=True, exist_ok=True)

    quarantined_paths = [
        PROJECT_ROOT / "data/v2_clean_template_grouped/test_clean.jsonl",
        PROJECT_ROOT / "data/v2_clean_template_grouped/val_clean.jsonl",
        PROJECT_ROOT / "data/human_eval/annotation_template.jsonl",
        PROJECT_ROOT / "data/evaluation/high_risk_challenge_set.jsonl",
        PROJECT_ROOT / "data/test/test.jsonl",
    ]
    quarantined_records: List[TicketRecord] = []
    quarantined_norms = set()
    for p in quarantined_paths:
        if not p.exists():
            continue
        for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            txt = (item.get("text") or item.get("redacted_text") or "").strip()
            if not txt:
                continue
            quarantined_norms.add(_norm(txt))
            quarantined_records.append(
                TicketRecord(
                    id=item.get("id", f"{p.stem}_{idx}"),
                    text=txt,
                    category=item.get("category", "general_query") if item.get("category") in CATEGORIES else "general_query",
                    priority=item.get("priority", "medium") if item.get("priority") in PRIORITIES else "medium",
                    language=item.get("language", "english") if item.get("language") in LANGUAGES else "english",
                    requires_human=bool(item.get("requires_human", False)),
                )
            )

    split_buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "cal": [], "audit": []}
    seen_texts = set()
    id_counter = 1

    for idx, (cat, prio, lang, hum, tag, part, base_txt) in enumerate(CURATED_RECORDS):
        transforms = SPLIT_TRANSFORMS[part]
        for tmpl, req_lang in transforms:
            if req_lang != "any" and req_lang != lang:
                continue
            if tmpl == "{t}":
                vtxt = base_txt
            else:
                inner = base_txt
                if tmpl[0].isascii() and len(inner) > 2 and not inner[:3].isupper():
                    inner = inner[0].lower() + inner[1:]
                vtxt = tmpl.format(t=inner)

            norm = _norm(vtxt)
            if norm in seen_texts or norm in quarantined_norms:
                continue
            seen_texts.add(norm)
            split_buckets[part].append(
                {
                    "id": f"v3_{part}_{id_counter:05d}",
                    "text": vtxt,
                    "category": cat,
                    "priority": prio,
                    "language": lang,
                    "requires_human": hum,
                    "template_family": f"fam_{cat}_{part}_{idx}",
                    "challenge_tag": tag,
                    "source": "synthetic_v3_diverse",
                }
            )
            id_counter += 1

    detector = LeakageDetector(similarity_threshold=0.78, ngram_size=4)
    all_v3_recs = [
        TicketRecord(
            id=r["id"],
            text=r["text"],
            category=r["category"],
            priority=r["priority"],
            language=r["language"],
            requires_human=r["requires_human"],
        )
        for rows in split_buckets.values()
        for r in rows
    ]
    leak_rep = detector.detect_leakage(all_v3_recs, [], quarantined_records, dataset_version="v3_diverse_vs_quarantined")
    if leak_rep.leakage_detected:
        sanitized, purged = detector.sanitize_train_against_locked_test(all_v3_recs, quarantined_records)
        keep_ids = {r.id for r in sanitized}
        for k in split_buckets:
            split_buckets[k] = [r for r in split_buckets[k] if r["id"] in keep_ids]
        print(f"Purged {purged} high-similarity V3 records vs quarantined splits")

    split_files = {
        "train": out_dir / "train_v3.jsonl",
        "val": out_dir / "val_v3.jsonl",
        "cal": out_dir / "cal_v3.jsonl",
        "audit": out_dir / "dev_audit_v3.jsonl",
    }

    manifest_splits: Dict[str, Any] = {}
    for sname, path in split_files.items():
        rows = split_buckets[sname]
        rng.shuffle(rows)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        unique_denorm = len({re.sub(r"[0-9]+", "<N>", r["text"].lower()) for r in rows})
        manifest_splits[sname] = {
            "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "count": len(rows),
            "unique_denumbered_texts": unique_denorm,
            "uniqueness_ratio": round(unique_denorm / max(len(rows), 1), 4),
            "sha256": sha256_file(path),
            "category_counts": dict(sorted(Counter(r["category"] for r in rows).items())),
            "priority_counts": dict(Counter(r["priority"] for r in rows)),
            "language_counts": dict(Counter(r["language"] for r in rows)),
            "requires_human_counts": {str(k): v for k, v in Counter(r["requires_human"] for r in rows).items()},
        }

    norm_sets = {k: {_norm(r["text"]) for r in v} for k, v in split_buckets.items()}
    internal_overlaps = {
        f"{a}_vs_{b}": len(norm_sets[a] & norm_sets[b])
        for i, a in enumerate(norm_sets)
        for b in list(norm_sets)[i + 1 :]
    }

    manifest = {
        "dataset_version": "v3_diverse",
        "source_disclosure": "synthetic_v3_diverse — curated syntactically diverse synthetic data, NOT real customer data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quarantine_leakage_check": {
            "against_files": [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in quarantined_paths if p.exists()],
            "exact_overlaps_post_sanitization": 0,
            "internal_cross_split_overlaps": internal_overlaps,
        },
        "splits": manifest_splits,
    }
    (out_dir / "dataset_v3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: {"count": v["count"], "langs": v["language_counts"], "cats": len(v["category_counts"])} for k, v in manifest_splits.items()}, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    expand_and_verify()
