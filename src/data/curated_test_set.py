"""Curated locked evaluation benchmark dataset for NEXUS-SLM.

These records represent independent, high-fidelity test scenarios
created separately from the synthetic training generator.
"""

from typing import List
from src.data.schemas import TicketRecord


def get_curated_locked_test_records() -> List[TicketRecord]:
    """
    Returns an independent, curated suite of test tickets covering
    all 14 categories across English, Hindi, and Hinglish.
    """
    raw_data = [
        # Payment Failure
        ("test_001", "Transaction ID TXN-9981 failed at bank gateway but money got debited.", "english", "payment_failure", "high", True),
        ("test_002", "Mera SBI account se 1200 kat gaye par order screen pe error aa gaya.", "hinglish", "payment_failure", "high", True),
        ("test_003", "पेमेंट करते समय गेटवे टाइमआउट हो गया और बैंक से पैसे कटने का एसएमएस आ गया।", "hindi", "payment_failure", "high", True),

        # Refund Request
        ("test_004", "Courier picked up returned shoes 4 days ago. Please initiate refund to bank.", "english", "refund_request", "medium", False),
        ("test_005", "Bhai return pickup to ho gaya tha par abhi tak refund account me credit nahi hua.", "hinglish", "refund_request", "medium", False),
        ("test_006", "वापस किए गए सामान का पैसा अभी तक मेरे खाते में क्यों नहीं आया?", "hindi", "refund_request", "medium", False),

        # Account Access
        ("test_007", "Unable to log in due to 'account locked for security reasons' warning.", "english", "account_access", "high", True),
        ("test_008", "OTP SMS receive nahi ho raha mere Jio number par login karte time.", "hinglish", "account_access", "high", True),
        ("test_009", "पासवर्ड सही डालने पर भी 'अकाउंट सस्पेंडेड' दिखा रहा है।", "hindi", "account_access", "high", True),

        # Password Reset
        ("test_010", "Password reset email verification link brings up a 404 expired page.", "english", "password_reset", "medium", False),
        ("test_011", "Forgot password link click karne pe email nahi aa raha inbox ya spam me.", "hinglish", "password_reset", "medium", False),
        ("test_012", "कृपया मेरे पंजीकृत मेल पर पासवर्ड रीसेट करने का लिंक पुनः भेजें।", "hindi", "password_reset", "medium", False),

        # Order Cancel
        ("test_013", "Please abort and cancel order #771239 immediately as I ordered duplicate.", "english", "order_cancel", "high", False),
        ("test_014", "Yeh order galti se place ho gaya hai, dispatch hone se pehle rok dijiye.", "hinglish", "order_cancel", "high", False),
        ("test_015", "मैंने गलत रंग का उत्पाद चुन लिया है, क्या यह आर्डर निरस्त हो सकता है?", "hindi", "order_cancel", "medium", False),

        # Order Tracking
        ("test_016", "The tracking AWB 88129033 shows package stuck in transit hub for a week.", "english", "order_tracking", "medium", False),
        ("test_017", "Shipment ka live status check karna hai, courier boy ka contact number dijiye.", "hinglish", "order_tracking", "medium", False),
        ("test_018", "मेरा पार्सल इस समय किस शहर में है और कब तक पहुंचेगा?", "hindi", "order_tracking", "medium", False),

        # Subscription
        ("test_019", "I disabled auto-renewal yet my credit card was billed $29 for annual plan.", "english", "subscription", "high", True),
        ("test_020", "Next month ka subscription plan cancel karke basic plan pe downgrade karna hai.", "hinglish", "subscription", "medium", False),
        ("test_021", "मेरी प्रीमियम सदस्यता स्वतः नवीनीकृत हो गई, मुझे इसे रद्द करना है।", "hindi", "subscription", "medium", False),

        # Billing
        ("test_022", "Discrepancy in billing: invoice reflects 18% tax on tax-exempt merchandise.", "english", "billing", "medium", True),
        ("test_023", "Bill me ₹250 convenience charge kis cheez ka laga hai details chahiye.", "hinglish", "billing", "low", False),
        ("test_024", "कृपया मेरी अंतिम खरीदारी का आधिकारिक टैक्स इनवॉइस पीडीएफ भेजें।", "hindi", "billing", "low", False),

        # Technical Issue
        ("test_025", "The payment submission endpoint throws a JSON serialization 500 error.", "english", "technical_issue", "high", True),
        ("test_026", "Android app update ke baad cart page pe white screen aa kar crash ho raha.", "hinglish", "technical_issue", "high", False),
        ("test_027", "वेबसाइट पर 'सबमिट' बटन दबाने पर कोई प्रतिक्रिया नहीं हो रही है।", "hindi", "technical_issue", "medium", False),

        # Fraud Report
        ("test_028", "CRITICAL: Multiple unauthorized charges detected from IP outside my country!", "english", "fraud_report", "critical", True),
        ("test_029", "Bhai kisi ne mera account hack karke unknown address pe order place kar diya!", "hinglish", "fraud_report", "critical", True),
        ("test_030", "अति गंभीर: मेरे क्रेडिट कार्ड से आपके पोर्टल पर धोखाधड़ीपूर्ण लेनदेन हुआ है।", "hindi", "fraud_report", "critical", True),

        # Delivery Issue
        ("test_031", "Delivery courier marked packet delivered at 11 PM but no doorbell was rung.", "english", "delivery_issue", "high", True),
        ("test_032", "Delivery boy ne packet seal tooti hui di aur andar headphones missing the.", "hinglish", "delivery_issue", "high", True),
        ("test_033", "डिलीवरी वाले ने पार्सल बाहर बारिश में फेंक दिया, उत्पाद पूरी तरह भीग गया।", "hindi", "delivery_issue", "high", True),

        # Account Closure
        ("test_034", "Under GDPR rights, I request permanent erasure of my profile and telemetry.", "english", "account_closure", "medium", True),
        ("test_035", "Mujhe apna account delete karna hai aur saare saved payment methods remove karne hain.", "hinglish", "account_closure", "low", False),
        ("test_036", "मैं अपनी प्रोफाइल और संबंधित सभी डेटा को स्थायी रूप से हटाना चाहता हूँ।", "hindi", "account_closure", "low", False),

        # Product Complaint
        ("test_037", "Received open box with severe scratches, clearly refurbished instead of new.", "english", "product_complaint", "high", True),
        ("test_038", "Saree ka fabric bilkul transparent aur torn nikla, website photo se alag hai.", "hinglish", "product_complaint", "medium", False),
        ("test_039", "भेजा गया सामान एकदम घटिया और टूटा हुआ है, तुरंत बदलें।", "hindi", "product_complaint", "high", False),

        # General Query
        ("test_040", "Do you offer enterprise B2B volume procurement discounts for schools?", "english", "general_query", "low", False),
        ("test_041", "Support center holiday pe open rehta hai kya calling ke liye?", "hinglish", "general_query", "low", False),
        ("test_042", "क्या आपकी सेवा रविवार को भी उपलब्ध रहती है?", "hindi", "general_query", "low", False),
    ]

    records: List[TicketRecord] = []
    for item in raw_data:
        tid, text, lang, cat, prio, req_h = item
        records.append(
            TicketRecord(
                id=tid,
                text=text,
                language=lang,  # type: ignore
                category=cat,  # type: ignore
                priority=prio,  # type: ignore
                requires_human=req_h,
                source="curated",
                validated=True,
            )
        )
    return records
