"""Synthetic training data generation pipeline for NEXUS-SLM."""

import random
from typing import Any, Dict, List, Optional
from src.data.schemas import TicketRecord
from src.utils.constants import CATEGORIES, PRIORITIES, LANGUAGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTicketGenerator:
    """
    Generates realistic, diverse multilingual customer support tickets
    across English, Hindi (Devanagari), and Hinglish (Roman Hindi).
    """

    # Category-specific template components
    SCENARIOS: Dict[str, Dict[str, List[str]]] = {
        "payment_failure": {
            "english": [
                "My payment was deducted from my bank account, but my order status still shows unpaid.",
                "Money was debited from my credit card, but I received a transaction failure screen.",
                "The UPI transaction succeeded on Google Pay, but your checkout page showed an error.",
                "I tried paying via net banking. The amount was debited twice, but no order was placed.",
                "Payment failed at the gateway, yet my bank balance has reduced.",
            ],
            "hindi": [
                "मेरे बैंक खाते से पैसे कट गए हैं, लेकिन ऑर्डर कन्फर्म नहीं हुआ।",
                "यूपीआई से पेमेंट कट गया पर ऐप में पेमेंट फेल्ड दिखा रहा है।",
                "क्रेडिट कार्ड से दो बार पैसे कट गए, कृपया तुरंत जांच करें।",
                "नेट बैंकिंग ट्रांजैक्शन सफल रहा लेकिन स्क्रीन पर एरर आ गया।",
                "मेरे खाते से राशि कट गई है पर ऑर्डर आईडी जनरेट नहीं हुई।",
            ],
            "hinglish": [
                "Mera payment kat gaya bank se but order confirm nahi hua.",
                "UPI payment successful dikha raha hai par app me transaction failed aaya.",
                "Paisa cut ho gaya mere account se par order summary me pending hai.",
                "Card se do baar paise kat gaye, order abhi bhi nahi bana.",
                "GPay se debit ho gaya but checkout page crash ho gaya tha.",
            ],
        },
        "refund_request": {
            "english": [
                "I returned the item three days ago. When will I receive my refund?",
                "The order was cancelled by the merchant, please process my refund immediately.",
                "I received a damaged product and initiated a return, but no refund has reached my account.",
                "Please credit my refund to the original payment method instead of store wallet.",
            ],
            "hindi": [
                "मैंने सामान वापस कर दिया था, मेरा रिफंड कब तक खाते में आएगा?",
                "ऑर्डर कैंसिल हो गया है, कृपया मेरे पैसे तुरंत वापस भेजें।",
                "खराब प्रोडक्ट मिला था जिसे मैंने लौटा दिया, रिफंड का क्या स्टेटस है?",
                "मेरा रिफंड वॉलेट के बजाय बैंक अकाउंट में ट्रांसफर कीजिए।",
            ],
            "hinglish": [
                "Maine item return kar diya tha 3 din pehle, mera refund kab aayega?",
                "Order cancel ho gaya tha, please mera refund jaldi initiate karo.",
                "Damaged product mila tha, return pickup ho gaya par paise wapas nahi aaye.",
                "Refund store wallet me nahi direct bank account me transfer chahiye.",
            ],
        },
        "account_access": {
            "english": [
                "I am unable to log in to my account. It says account suspended or blocked.",
                "Two-factor authentication code is not arriving on my registered mobile number.",
                "My account appears locked after entering the wrong credentials.",
                "I cannot access my profile from a new device, verification email never arrived.",
            ],
            "hindi": [
                "मैं अपने खाते में लॉगिन नहीं कर पा रहा हूँ, अकाउंट ब्लॉक दिखा रहा है।",
                "मेरे पंजीकृत मोबाइल नंबर पर ओटीपी कोड नहीं आ रहा है।",
                "गलत पासवर्ड डालने के बाद मेरा अकाउंट लॉक हो गया है, मदद करें।",
                "नए फोन में लॉगिन करने पर वेरिफिकेशन कोड प्राप्त नहीं हो रहा है।",
            ],
            "hinglish": [
                "Main apne account me login nahi kar pa raha, account locked show ho raha hai.",
                "Registered mobile number pe OTP nahi aa raha, login kaise karu?",
                "Account temporary disable dikha raha hai bina kisi warning ke.",
                "Login attempt karne pe bar bar session timeout error aa raha hai.",
            ],
        },
        "password_reset": {
            "english": [
                "I forgot my account password and need a reset link sent to my email.",
                "The password reset link in the email has expired. Please send a new one.",
                "Unable to reset my password, the submit button is unresponsive.",
            ],
            "hindi": [
                "मैं अपना पासवर्ड भूल गया हूँ, कृपया रीसेट लिंक मेरे ईमेल पर भेजें।",
                "पासवर्ड रीसेट लिंक एक्सपायर बता रहा है, नया लिंक भेजें।",
                "नया पासवर्ड सेट करने की कोशिश कर रहा हूँ लेकिन एरर आ रहा है।",
            ],
            "hinglish": [
                "Mera password bhool gaya hu, reset link email pe send kar dijiye.",
                "Password reset link pe click karne pe link expired ka error aa raha hai.",
                "New password banate time security criteria match nahi ho raha.",
            ],
        },
        "order_cancel": {
            "english": [
                "I placed this order by mistake, please cancel it immediately before dispatch.",
                "I need to cancel order #98214 as I selected the wrong delivery address.",
                "The cancel button is missing on my order details page, please cancel it.",
            ],
            "hindi": [
                "मैंने गलती से यह ऑर्डर कर दिया था, कृपया इसे तुरंत रद्द करें।",
                "कृपया मेरा ऑर्डर कैंसिल कर दें क्योंकि मैंने गलत साइज चुन लिया था।",
                "ऐप में कैंसिल बटन नहीं दिख रहा है, क्या आप इस ऑर्डर को रोक सकते हैं?",
            ],
            "hinglish": [
                "Galti se order place ho gaya, dispatch hone se pehle cancel kar do please.",
                "Order cancel karna hai mujhe kyunki galat address select ho gaya tha.",
                "Order details page pe cancel ka option nahi aa raha, please cancel it.",
            ],
        },
        "order_tracking": {
            "english": [
                "Where is my package? The tracking number shows no updates for 4 days.",
                "Expected delivery was yesterday, but the courier tracking status hasn't moved.",
                "Can you provide live courier tracking for my order #44912?",
            ],
            "hindi": [
                "मेरा पार्सल कहाँ है? ट्रैकिंग नंबर में चार दिनों से कोई अपडेट नहीं है।",
                "कल डिलीवरी होने वाली थी लेकिन अभी तक कोई कूरियर नहीं आया।",
                "कृपया मेरे ऑर्डर की वर्तमान ट्रैकिंग लोकेशन बताएं।",
            ],
            "hinglish": [
                "Mera parcel kaha hai? Tracking status me 4 din se koi update nahi hai.",
                "Delivery date kal ki thi but package abhi tak out for delivery nahi hua.",
                "Courier tracking number invalid dikha raha hai, live status kya hai?",
            ],
        },
        "subscription": {
            "english": [
                "I want to pause or cancel my monthly premium subscription.",
                "I was charged for auto-renewal even though I turned off the subscription toggle.",
                "How do I upgrade from the basic plan to the annual enterprise tier?",
            ],
            "hindi": [
                "मैं अपना मासिक प्रीमियम सब्सक्रिप्शन रद्द करना चाहता हूँ।",
                "मैंने ऑटो-रिन्यू बंद कर दिया था फिर भी मेरे कार्ड से पैसे कट गए।",
                "वार्षिक प्लान में अपग्रेड कैसे करें, कृपया मार्गदर्शन करें।",
            ],
            "hinglish": [
                "Mujhe apna monthly premium subscription cancel karna hai.",
                "Auto-debit off kiya tha fir bhi subscription amount kat gaya.",
                "Family plan me upgrade karne ka process kya hai?",
            ],
        },
        "billing": {
            "english": [
                "There is an unknown surcharge on my latest invoice. Please clarify.",
                "I need the GST tax invoice for my recent purchase for business filing.",
                "My monthly bill includes charges for services I never activated.",
            ],
            "hindi": [
                "मेरे हालिया बिल में कुछ अज्ञात शुल्क जुड़े हुए हैं, कृपया स्पष्ट करें।",
                "मुझे अपनी हालिया खरीद का जीएसटी टैक्स इनवॉइस चाहिए।",
                "बिल में ऐसी सेवाओं के पैसे जोड़े गए हैं जो मैंने ली ही नहीं।",
            ],
            "hinglish": [
                "Latest invoice me unexpected charges add hue hain, explain kijiye.",
                "Mere purchase ka GST invoice download nahi ho raha portal se.",
                "Bill me extra service charge lagaya gaya hai jo galat hai.",
            ],
        },
        "technical_issue": {
            "english": [
                "The mobile application keeps crashing whenever I open the checkout screen.",
                "Getting a 500 internal server error while attempting to upload my KYC documents.",
                "Search results are failing to load and showing a blank white screen.",
            ],
            "hindi": [
                "चेकआउट स्क्रीन खोलते ही मोबाइल ऐप बार-बार क्रैश हो रहा है।",
                "केवाईसी दस्तावेज अपलोड करते समय सर्वर एरर 500 आ रहा है।",
                "ऐप में सर्च करने पर सिर्फ सफेद स्क्रीन दिख रही है, कुछ लोड नहीं हो रहा।",
            ],
            "hinglish": [
                "Checkout page open karte hi app force close ho jata hai.",
                "Document upload karne pe server error 500 aa raha hai bar bar.",
                "App bahut slow chal raha hai aur screen blank ho rahi hai.",
            ],
        },
        "fraud_report": {
            "english": [
                "Urgent: Unauthorized transaction of $450 occurred on my account. Block my card now!",
                "Someone accessed my account without permission and changed my linked phone number.",
                "I suspect fraudulent activity on my profile. An unauthorized order was placed.",
            ],
            "hindi": [
                "अति आवश्यक: मेरे खाते से अनधिकृत लेनदेन हुआ है, तुरंत मेरा कार्ड ब्लॉक करें!",
                "किसी ने मेरे खाते में अनधिकृत रूप से लॉगिन करके फोन नंबर बदल दिया है।",
                "मेरे खाते में संदिग्ध धोखाधड़ी हुई है, यह आर्डर मैंने नहीं दिया था।",
            ],
            "hinglish": [
                "Urgent: Mere account se unauthorized transaction hua hai, please account freeze karo!",
                "Kisi hacker ne mera registered email aur password change kar diya hai.",
                "Fraudulent order place hua hai mere card se jo maine nahi kiya.",
            ],
        },
        "delivery_issue": {
            "english": [
                "The delivery agent marked my order as 'Delivered', but I never received anything!",
                "Package arrived torn and half the items inside the box were missing.",
                "Delivery agent was rude and refused to deliver to my doorstep.",
            ],
            "hindi": [
                "डिलीवरी एजेंट ने 'डिलीवर' मार्क कर दिया लेकिन मुझे कोई सामान नहीं मिला!",
                "पार्सल फटा हुआ मिला और डिब्बे के अंदर का आधा सामान गायब था।",
                "डिलीवरी बॉय ने बदतमीजी की और घर पर सामान देने से मना कर दिया।",
            ],
            "hinglish": [
                "Delivery status delivered dikha raha hai par mujhe package nahi mila!",
                "Packet phata hua tha aur andar ka saman gayab tha.",
                "Delivery boy ne call karke location pe aane se mana kar diya.",
            ],
        },
        "account_closure": {
            "english": [
                "I want to permanently delete my account and erase all my personal data.",
                "Please close my user account and remove my registered payment cards.",
                "Where is the option to permanently deactivate and delete my profile?",
            ],
            "hindi": [
                "मैं अपना खाता स्थायी रूप से हटाना चाहता हूँ और व्यक्तिगत डेटा मिटाना चाहता हूँ।",
                "कृपया मेरा उपयोगकर्ता खाता बंद करें और सहेजे गए कार्ड हटाएं।",
                "अकाउंट को हमेशा के लिए डिलीट करने की प्रक्रिया क्या है?",
            ],
            "hinglish": [
                "Mujhe apna account permanently delete karna hai aur all data erase chahiye.",
                "Please mera account close kar do aur saved card details delete karo.",
                "Profile permanently deactivate karne ka option nahi mil raha settings me.",
            ],
        },
        "product_complaint": {
            "english": [
                "The product delivered is completely counterfeit and poor quality.",
                "Expired product was shipped to me; manufacturing date is two years old.",
                "The electronic device stopped working within two hours of unboxing.",
            ],
            "hindi": [
                "डिलीवर किया गया उत्पाद पूरी तरह नकली और खराब गुणवत्ता का है।",
                "मुझे एक्सपायर सामान भेज दिया गया है जिसकी तारीख निकल चुकी है।",
                "अनबॉक्सिंग के दो घंटे के अंदर ही यह इलेक्ट्रॉनिक उपकरण खराब हो गया।",
            ],
            "hinglish": [
                "Product duplicate/fake nikla hai, quality bahut hi bekar hai.",
                "Expired item deliver kar diya, manufacturing date 2 saal purani hai.",
                "Device kholte hi band ho gaya, defected piece bheja gaya hai.",
            ],
        },
        "general_query": {
            "english": [
                "What are the operating hours of your customer support center?",
                "Do you provide international shipping to Canada or the UK?",
                "Can you explain the warranty coverage policy on purchased laptops?",
            ],
            "hindi": [
                "आपके ग्राहक सहायता केंद्र का कार्य समय क्या है?",
                "क्या आप भारत से बाहर अंतरराष्ट्रीय डिलीवरी की सुविधा देते हैं?",
                "लैपटॉप की वारंटी और रिप्लेसमेंट पॉलिसी के नियम क्या हैं?",
            ],
            "hinglish": [
                "Aapki customer care team ka working time kab se kab tak hai?",
                "Kya aap international shipping provide karte ho dusre countries me?",
                "Product ki warranty claim karne ka kya process hota hai?",
            ],
        },
    }

    # Priority heuristics per category
    PRIORITY_DISTRIBUTION: Dict[str, Dict[str, float]] = {
        "fraud_report": {"critical": 0.85, "high": 0.15, "medium": 0.0, "low": 0.0},
        "payment_failure": {"critical": 0.10, "high": 0.70, "medium": 0.20, "low": 0.0},
        "account_access": {"critical": 0.05, "high": 0.55, "medium": 0.35, "low": 0.05},
        "delivery_issue": {"critical": 0.05, "high": 0.60, "medium": 0.30, "low": 0.05},
        "product_complaint": {"critical": 0.05, "high": 0.35, "medium": 0.50, "low": 0.10},
        "refund_request": {"critical": 0.0, "high": 0.40, "medium": 0.50, "low": 0.10},
        "order_cancel": {"critical": 0.05, "high": 0.45, "medium": 0.45, "low": 0.05},
        "billing": {"critical": 0.05, "high": 0.30, "medium": 0.55, "low": 0.10},
        "technical_issue": {"critical": 0.05, "high": 0.30, "medium": 0.55, "low": 0.10},
        "order_tracking": {"critical": 0.0, "high": 0.10, "medium": 0.60, "low": 0.30},
        "password_reset": {"critical": 0.0, "high": 0.15, "medium": 0.65, "low": 0.20},
        "subscription": {"critical": 0.0, "high": 0.15, "medium": 0.55, "low": 0.30},
        "account_closure": {"critical": 0.0, "high": 0.10, "medium": 0.40, "low": 0.50},
        "general_query": {"critical": 0.0, "high": 0.05, "medium": 0.35, "low": 0.60},
    }

    # Noise & variation injection helpers
    TYPO_MAP = {
        "payment": "paymnt",
        "order": "ordr",
        "refund": "refnd",
        "account": "accnt",
        "password": "paswrd",
        "delivery": "delvry",
        "cancel": "cancle",
        "failed": "faild",
        "urgent": "urgnt",
        "please": "plz",
        "paisa": "paise",
        "kat": "kt",
        "gaya": "gya",
        "nahi": "nhi",
    }

    SLANG_PREFIXES = ["bro", "plz help", "yaar", "urgent", "hey team", "hello sir", "bhai"]
    EMOJIS = [" 😭", " 😡", " 🙏", " !!", " :(", " ⚠️"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _sample_priority(self, category: str) -> str:
        dist = self.PRIORITY_DISTRIBUTION.get(
            category, {"low": 0.25, "medium": 0.5, "high": 0.2, "critical": 0.05}
        )
        r = self.rng.random()
        cumulative = 0.0
        for prio, weight in dist.items():
            cumulative += weight
            if r <= cumulative:
                return prio
        return "medium"

    def _sample_requires_human(self, category: str, priority: str) -> bool:
        if priority in ["critical", "high"] or category in ["fraud_report", "account_closure"]:
            return self.rng.random() < 0.90
        elif priority == "medium":
            return self.rng.random() < 0.45
        else:
            return self.rng.random() < 0.15

    def _apply_perturbations(self, text: str, language: str) -> str:
        variation_type = self.rng.choice([
            "canonical",
            "typo",
            "slang",
            "noisy_punctuation",
            "short",
            "long",
        ])

        if variation_type == "typo" and language in ["english", "hinglish"]:
            for word, typo in self.TYPO_MAP.items():
                if word in text.lower() and self.rng.random() < 0.6:
                    text = text.replace(word, typo)
            return text

        elif variation_type == "slang":
            prefix = self.rng.choice(self.SLANG_PREFIXES)
            emoji = self.rng.choice(self.EMOJIS)
            return f"{prefix} {text}{emoji}"

        elif variation_type == "noisy_punctuation":
            if self.rng.random() < 0.5:
                return text.upper() + "???!!!!"
            return text + "??!?!"

        elif variation_type == "short":
            words = text.split()
            if len(words) > 4:
                return " ".join(words[:4]) + "?"
            return text

        elif variation_type == "long":
            extra = (
                " I have been contacting customer support for the last 3 days with no resolution."
                if language == "english"
                else (
                    " Kripya isko urgently dekhein mera kafi nuksan ho raha hai."
                    if language == "hinglish"
                    else " कृपया तुरंत संज्ञान लें अन्यथा मुझे उपभोक्ता फोरम में शिकायत करनी पड़ेगी।"
                )
            )
            return f"{text}{extra}"

        return text

    def generate_record(
        self,
        index: int,
        category: Optional[str] = None,
        language: Optional[str] = None,
        source: str = "synthetic",
    ) -> TicketRecord:
        """Generates a single synthetic ticket record with metadata."""
        cat = category or self.rng.choice(CATEGORIES)
        lang = language or self.rng.choice(LANGUAGES)

        templates = self.SCENARIOS[cat][lang]
        base_text = self.rng.choice(templates)
        final_text = self._apply_perturbations(base_text, lang)

        prio = self._sample_priority(cat)
        req_human = self._sample_requires_human(cat, prio)

        return TicketRecord(
            id=f"ticket_{index:06d}",
            text=final_text,
            language=lang,  # type: ignore
            category=cat,  # type: ignore
            priority=prio,  # type: ignore
            requires_human=req_human,
            source=source,  # type: ignore
            validated=False,
        )

    def generate_dataset(
        self,
        num_samples: int = 1000,
        source: str = "synthetic",
    ) -> List[TicketRecord]:
        """Generates a balanced synthetic dataset across categories and languages."""
        records: List[TicketRecord] = []
        samples_per_cat_lang = max(1, num_samples // (len(CATEGORIES) * len(LANGUAGES)))

        idx = 1
        for cat in CATEGORIES:
            for lang in LANGUAGES:
                for _ in range(samples_per_cat_lang):
                    records.append(
                        self.generate_record(idx, category=cat, language=lang, source=source)
                    )
                    idx += 1

        # Fill remaining if any
        while len(records) < num_samples:
            records.append(self.generate_record(idx, source=source))
            idx += 1

        self.rng.shuffle(records)
        # Re-index for consistent IDs
        for i, rec in enumerate(records, start=1):
            rec.id = f"ticket_{i:06d}"

        logger.info(f"Generated {len(records)} synthetic records across 14 categories & 3 languages.")
        return records
