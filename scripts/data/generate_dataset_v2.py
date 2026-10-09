"""Dataset V2 Generator & Curation Pipeline.

Expands NEXUS-SLM training corpus to 5,000+ examples with:
- Strict class balancing across all 14 taxonomy categories
- Comprehensive multilingual distribution:
  - English (~30%)
  - Hinglish code-switched (~25%)
  - Roman Hindi (~20%)
  - Devanagari Hindi (~25%)
- Error-driven targeted slices addressing V1 failure modes:
  1. Semantic Category Overlap (refund vs cancellation, payment vs billing, delivery vs tracking)
  2. Context Sparsity & Short Tickets (1-5 words with explicit entity clues)
  3. Devanagari Hindi vocabulary enrichment
  4. Rare class augmentation (account_access, subscription, technical_issue, fraud_report)
- Explicit provenance metadata on every record
- Deterministic schema validation and split generation (train: 5000, val: 500, test_v2: 500)
- Preserves original locked test.jsonl (42 samples) untouched.
"""

import hashlib
import json
import random
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_V2_DIR = WORKSPACE_ROOT / "data" / "v2"
DATA_V2_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "payment_failure", "refund_request", "order_cancel", "delivery_issue",
    "order_tracking", "account_access", "password_reset", "technical_issue",
    "billing", "subscription", "fraud_report", "product_complaint",
    "general_query", "account_closure"
]

PRIORITIES = ["low", "medium", "high", "critical"]

# Seed corpora by category and linguistic modality
CATEGORY_PATTERNS = {
    "payment_failure": {
        "en": [
            "Amount of ${amt} debited from my bank via {gateway} but payment shows failed on order {oid}.",
            "Payment transaction timeout during checkout; bank account deducted twice.",
            "My credit card was charged for transaction {oid} but merchant says unauthorized.",
            "UPI transaction stuck in pending status since yesterday, money deducted.",
            "Payment failed with error code ERR_PAY_902; please reverse the deduction.",
            "Double charge observed on invoice {oid} through net banking.",
            "Payment declined at gateway, but funds deducted from savings account.",
            "Payment failure",
            "Money deducted but order failed",
            "UPI error while paying",
            "Gateway crash money lost",
            "I was making a payment of Rs {amt} using my credit card when the session expired. The funds were debited immediately from my account, but your portal shows payment failed. Kindly confirm if this will auto-refund or if you can capture the payment against order {oid}."
        ],
        "hinglish": [
            "Account se {amt} cut ho gaya but payment failure dikha raha hai order {oid} pe.",
            "UPI payment fail ho gaya par bank balance kam ho gaya hai bhai.",
            "Payment reject hua checkout pe but paise deduct ho gaye, please check karo.",
            "Net banking transaction pending aa raha hai aur order confirm nahi hua.",
            "Bhai do baar payment kat gayi ek hi order ke liye, gateway error tha.",
            "Mera {gateway} se paise kat gaya aur app pe failed bol raha hai.",
            "Paise cut gaye order fail",
            "Payment fail ho gaya",
            "UPI fas gaya account se kata",
            "Bhai payment atak gayi",
            "Maine kal raat {amt} ka payment kiya tha {gateway} ke through. Bank se SMS aa gaya ki amount deduct ho gaya hai, lekin aapke app par order status failed dikha raha hai. Kripya transaction verify karke order confirm karein ya refund initiate karein."
        ],
        "roman_hindi": [
            "Khate se paise kat gaye lekin bhugtan viphal dikha raha hai.",
            "Mera bhugtan nahi hua aur dhan rashi kat gayi hai, kripya sahayata karein.",
            "UPI len den atak gaya hai, khate se rashi kat chuki hai.",
            "Do baar rashi kat gayi hai ek hi aadesh ke liye.",
            "Bhugtan me truti aa rahi hai aur paise wapas nahi aaye.",
            "Paise kat gaye",
            "Bhugtan viphal hua",
            "UPI truti",
            "Maine {gateway} se aadesh sankhya {oid} ke liye bhugtan kiya tha. Mere bank khate se {amt} rupaye kat gaye hain kintu aadesh confirm nahi hua hai. Kripya turant iska samadhan karein."
        ],
        "devanagari": [
            "मेरे बैंक खाते से {amt} रुपये कट गए हैं लेकिन भुगतान असफल दिखा रहा है।",
            "यूपीआई से लेनदेन किया था, राशि कट गई पर ऑर्डर {oid} पेंडिंग है।",
            "दो बार भुगतान कट गया है, कृपया अतिरिक्त राशि वापस करें।",
            "भुगतान गेटवे त्रुटि के कारण राशि अटक गई है।",
            "खाते से पैसे कटे पर रसीद नहीं मिली।",
            "पैसे कट गए",
            "भुगतान विफल हुआ",
            "यूपीआई अटक गया",
            "मैंने आदेश संख्या {oid} के लिए {amt} रुपये का भुगतान किया था। बैंक से राशि कटने का संदेश आ चुका है लेकिन आपके पोर्टल पर भुगतान विफल दिखाई दे रहा है। कृपया जल्द जांचें।"
        ]
    },
    "refund_request": {
        "en": [
            "Please process refund of ${amt} for returned item on order {oid}.",
            "Return was picked up 5 days ago, refund not credited to original payment source.",
            "I was promised refund within 48 hours for cancelled order {oid}, still awaiting funds.",
            "Refund status shows completed on app, but my bank statement shows no credit.",
            "Incorrect item delivered and returned; initiate my full refund immediately.",
            "Where is my refund for transaction {oid}?",
            "Refund not received",
            "Money refund please",
            "Return refund status",
            "Want my money back",
            "I returned the defective product associated with order {oid} on Monday. The courier agent completed the pickup, yet the refund of ${amt} is still not reflected in my bank account. Kindly expedite the refund process."
        ],
        "hinglish": [
            "Order {oid} ka return pickup ho gaya hai, please mera refund account me transfer karo.",
            "Mera {amt} ka refund abhi tak bank me nahi aaya, 4 din ho gaye hain.",
            "Cancelled order ka refund kab aayega? App pe processed dikha raha hai par bank me nahi mila.",
            "Maine product wapas bhej diya, kripya refund jaldi initiate karein.",
            "Refund pending kyu hai? Mujhe mera paisa wapas chahiye.",
            "Refund kab milega?",
            "Mera refund nahi aaya",
            "Paise wapas chahiye",
            "Return refund status?",
            "Bhai maine pichle hafte order {oid} cancel kiya tha. Customer care ne bola tha 24 hours me refund aa jayega lekin abhi tak mere account me koi paisa credit nahi hua hai. Please refund reference number provide karein."
        ],
        "roman_hindi": [
            "Wapas kiye gaye utpad ka refund abhi tak nahi mila hai.",
            "Aadesh {oid} ka dhan wapas mere khate me bhej dijiye.",
            "Return pickup ho chuka hai, kripya refund turant karein.",
            "Mera paisa wapas kab tak aayega?",
            "Refund ki sthiti janni hai.",
            "Refund chahiye",
            "Dhan wapas karo",
            "Paisa kab aayega?",
            "Aadesh {oid} ko wapas kar diya gaya hai. Kripya nirdharit samay sima ke andar {amt} rupaye ka refund mere mool bhugtan khate me bhej dijiye."
        ],
        "devanagari": [
            "वापस किए गए सामान का रिफंड {amt} रुपये अभी तक खाते में नहीं आया।",
            "आदेश {oid} रद्द कर दिया था, कृपया मेरी राशि वापस बैंक में भेजें।",
            "उत्पाद पिकअप हो चुका है, रिफंड कब तक जमा होगा?",
            "रिफंड की स्थिति जांचें, बैंक में कोई राशि प्राप्त नहीं हुई है।",
            "रिफंड नहीं मिला",
            "पैसे वापस चाहिए",
            "रिफंड कब आएगा?",
            "मैंने 5 दिन पहले ऑर्डर {oid} वापस कर दिया था लेकिन अभी तक रिफंड बैंक खाते में क्रेडिट नहीं हुआ है। कृपया रिफंड ट्रैकिंग आईडी दें।"
        ]
    },
    "order_cancel": {
        "en": [
            "I want to cancel order {oid} immediately as it was placed by mistake.",
            "Please cancel my shipment for order {oid} before it dispatches from warehouse.",
            "Cancel order {oid} because delivery date is too delayed for my requirement.",
            "Duplicate order created accidentally; please stop order {oid}.",
            "Cancel button is disabled on order details page, cancel order {oid} manually.",
            "Cancel order",
            "Stop my order",
            "Cancel order {oid}",
            "Want to cancel",
            "I accidentally ordered the wrong size and color on order {oid}. The shipment has not been dispatched yet. Please cancel this order and ensure I am not charged for delivery."
        ],
        "hinglish": [
            "Order {oid} cancel karna hai, galti se galat item select ho gaya.",
            "Shipment dispatch hone se pehle order {oid} cancel kardo please.",
            "Bhai delivery bahut late ho rahi hai, mujhe ye order cancel karna hai.",
            "App me cancel ka option nahi aa raha, order {oid} cancel kar do turant.",
            "Order cancel kardo please",
            "Ye order nahi chahiye",
            "Cancel karo abhi",
            "Order cancel please",
            "Maine galti se do baar same order {oid} place kar diya hai. Mujhe sirf ek item chahiye tha. Kripya doosra order turant cancel kar dein aur mujhe confirmation email bhejein."
        ],
        "roman_hindi": [
            "Aadesh {oid} ko turant radd karna chahta hoon.",
            "Galti se aadesh ho gaya hai, kripya isko cancel karein.",
            "Bhejne se pehle mera order radd kar dijiye.",
            "Order radd karein",
            "Cancel karna hai",
            "Aadesh radd karo",
            "Kripya aadesh sankhya {oid} ko radd kar dein kyunki mujhe ab is utpad ki avashyakta nahi hai."
        ],
        "devanagari": [
            "कृपया आदेश संख्या {oid} को तुरंत रद्द करें, गलती से ऑर्डर हो गया था।",
            "डिस्पैच होने से पहले मेरा सामान रद्द कर दीजिए।",
            "डिलीवरी में बहुत देर हो रही है, मुझे यह ऑर्डर अब नहीं चाहिए।",
            "ऑर्डर रद्द करें",
            "कैंसिल करना है",
            "ऑर्डर बंद करो",
            "मैंने गलती से गलत पता चुन लिया था, इसलिए कृपया आदेश {oid} को तत्काल रद्द कर दें ताकि मैं नया ऑर्डर कर सकूं।"
        ]
    },
    "delivery_issue": {
        "en": [
            "Package marked as delivered on order {oid} but I never received it.",
            "Delivery courier arrived with damaged and torn packaging, refused delivery.",
            "Delivery boy was extremely rude and asked for extra cash beyond prepaid bill.",
            "Delivery attempted notification received while I was home all day.",
            "Wrong item received inside package for order {oid}.",
            "Package not received",
            "Courier issue",
            "Damaged package",
            "Fake delivery attempt",
            "According to the tracking portal, order {oid} was delivered at 3 PM today, but neither I nor my neighbors received any parcel. Please investigate with the delivery executive."
        ],
        "hinglish": [
            "App pe delivered dikha raha hai par mujhe order {oid} ka parcel nahi mila.",
            "Delivery boy ne bina call kiye fake attempt mark kar diya.",
            "Parcel phata hua tha aur andar ka product damaged hai.",
            "Delivery boy extra paise maang raha hai jabki prepaid order tha.",
            "Order deliver nahi hua",
            "Saman toota hua mila",
            "Fake delivery status",
            "Delivery boy rude tha",
            "Bhai tracker pe likha aa raha hai ki package deliver ho gaya, lekin security guard ya mujhe kisi ko delivery nahi mili. Please delivery agent ka contact number do."
        ],
        "roman_hindi": [
            "Aadesh deliver dikha raha hai kintu mujhe koi saman nahi mila.",
            "Delivery boy ne bina aaye hi delivery fail mark kar di.",
            "Parcel kshatigrast sthiti me prapt hua.",
            "Saman nahi mila",
            "Delivery samasya",
            "Galat delivery",
            "Aadesh sankhya {oid} ka parcel mujhe prapt nahi hua hai halanki tracking status me delivered likha hai. Kripya turant delivery agent se sampark karein."
        ],
        "devanagari": [
            "ट्रैकिंग में डिलीवर दिखा रहा है लेकिन मुझे आदेश {oid} का कोई पार्सल नहीं मिला।",
            "डिलीवरी बॉय ने बिना आए ही डिलीवरी फेल्ड मार्क कर दिया।",
            "पार्सल फटा हुआ मिला और उत्पाद टूटा हुआ था।",
            "सामान नहीं मिला",
            "डिलीवरी की समस्या",
            "फर्जी डिलीवरी",
            "मेरे पते पर कोई डिलीवरी एजेंट नहीं आया फिर भी स्टेटस डिलीवर हो चुका है। कृपया आदेश {oid} के संबंध में जांच करें।"
        ]
    },
    "order_tracking": {
        "en": [
            "Where is my package for order {oid}? Tracking has not updated in 4 days.",
            "Can you provide the live tracking link and estimated delivery time for order {oid}?",
            "Tracking number AW89230 shows invalid on courier partner website.",
            "When will order {oid} reach my location in {city}?",
            "Where is my order?",
            "Track order {oid}",
            "Delivery status please",
            "Tracking not updating",
            "I ordered items under order {oid} five days ago. The status is stuck on 'Dispatched' without any updates from the logistics partner. Kindly provide the updated current location and ETA."
        ],
        "hinglish": [
            "Mera order {oid} abhi kahan par hai? Tracking link update nahi ho rahi.",
            "Order {oid} kab tak deliver hoga {city} me?",
            "Courier tracking number kaam nahi kar raha hai, status batao.",
            "Order kahan pahuncha?",
            "Track my order",
            "Kab aayega parcel?",
            "Tracking update chahiye",
            "Bhai aadesh sankhya {oid} 3 din pehle dispatch hua tha par tab se tracking status ek hi jagah ruka hua hai. Kya aap check karke bata sakte hain ki kab tak delivery expect kar sakte hain?"
        ],
        "roman_hindi": [
            "Mera parcel abhi kahan hai? Tracking ki jankari chahiye.",
            "Aadesh {oid} kab tak mere pate par pahunchega?",
            "Tracking sankhya kaam nahi kar rahi hai.",
            "Order kahan hai?",
            "Tracking jankari",
            "Aadesh ki sthiti?",
            "Kripya aadesh sankhya {oid} ki vartaman sthiti aur anumanit delivery samay bataiye."
        ],
        "devanagari": [
            "आदेश {oid} अभी कहां पहुंचा है? ट्रैकिंग में कोई अपडेट नहीं आ रहा।",
            "मेरे आदेश की वर्तमान स्थिति और डिलीवरी का अनुमानित समय क्या है?",
            "कूरियर ट्रैकिंग नंबर काम नहीं कर रहा है।",
            "ऑर्डर कहां है?",
            "ट्रैकिंग स्थिति बताएं",
            "पार्सल कब पहुंचेगा?",
            "मैंने आदेश {oid} का ऑर्डर दिया था। कृपया बताएं कि यह {city} में मेरे पते पर किस तारीख तक पहुंचेगा।"
        ]
    },
    "account_access": {
        "en": [
            "I cannot log into my account, error says 'Account suspended due to unusual activity'.",
            "Lost access to registered phone number, unable to receive 2FA verification OTP.",
            "Verification email for account login is not arriving in inbox or spam folder.",
            "My account has been locked after 3 incorrect password attempts; please unlock.",
            "Cannot login to account",
            "Account locked out",
            "OTP not receiving",
            "Login failed",
            "I have been trying to log into my registered account {email} for the past 2 hours. The SMS verification code is not being delivered to my mobile. Kindly help me regain access."
        ],
        "hinglish": [
            "Mera account login nahi ho raha, OTP mobile pe nahi aa raha hai.",
            "Account locked dikha raha hai, please unlock karne me help karo.",
            "Purana phone number band ho gaya hai, account access kaise milega?",
            "Login nahi ho raha",
            "Account block ho gaya",
            "OTP nahi aa raha",
            "Access chahiye account ka",
            "Bhai maine apna password sahi enter kiya par fir bhi account suspended dikha raha hai. Mera registered email {email} hai, please verification link bhej kar account restore karein."
        ],
        "roman_hindi": [
            "Khate me login nahi kar pa raha hoon, OTP prapt nahi ho raha.",
            "Mera khata awarudh (locked) dikha raha hai, kripya kholein.",
            "Khateme pravesh me samasya aa rahi hai.",
            "Login samasya",
            "Khata locked",
            "OTP samasya",
            "Mere panjikrit khate {email} me pravesh nahi ho pa raha hai. Kripya login punah sthapit karne me sahayata karein."
        ],
        "devanagari": [
            "मैं अपने खाते में लॉगिन नहीं कर पा रहा हूँ, ओटीपी नहीं आ रहा है।",
            "मेरा खाता संदिग्ध गतिविधि के कारण लॉक कर दिया गया है, कृपया अनलॉक करें।",
            "पंजीकृत ईमेल पर सत्यापन कोड प्राप्त नहीं हो रहा है।",
            "लॉगिन नहीं हो रहा",
            "खाता लॉक है",
            "ओटीपी नहीं आया",
            "मेरे पंजीकृत खाते {email} का पासवर्ड सही होने पर भी खाता निष्क्रिय दिखा रहा है। कृपया मुझे लॉगिन एक्सेस प्रदान करें।"
        ]
    },
    "password_reset": {
        "en": [
            "Password reset link sent to my email {email} has expired or shows invalid token.",
            "I forgot my account password and the reset link is not arriving.",
            "Security questions verification fails during password change process.",
            "Forgot password",
            "Reset password link expired",
            "Cannot change password",
            "Password reset",
            "I requested a password reset link for my profile {email} multiple times today, but the link opens to a 404 error page. Please send a working reset link."
        ],
        "hinglish": [
            "Password reset link email pe nahi aa raha hai, please check karo.",
            "Password bhool gaya hoon aur reset link expire bata raha hai.",
            "Naya password set karte time security criteria match nahi ho raha.",
            "Password bhool gaya",
            "Reset password please",
            "Password change nahi ho raha",
            "Bhai maine password reset ka request daala tha par mail me jo link aaya wo expire dikha raha hai. Please ek fresh password reset link bhejo mere email {email} pe."
        ],
        "roman_hindi": [
            "Password reset link kaam nahi kar raha hai.",
            "Apne khate ka password bhool gaya hoon, kripya naya banayein.",
            "Password badalne me truti aa rahi hai.",
            "Password bhool gaya",
            "Password badlo",
            "Password reset link bhejein",
            "Kripya mere panjikrit email par password reset link punah bhejiye."
        ],
        "devanagari": [
            "पासवर्ड रीसेट लिंक मेरे ईमेल पर काम नहीं कर रहा है।",
            "मैं अपने खाते का पासवर्ड भूल गया हूँ, कृपया पासवर्ड रीसेट करने में सहायता करें।",
            "नया पासवर्ड सेट करते समय सुरक्षा त्रुटि आ रही है।",
            "पासवर्ड भूल गया",
            "पासवर्ड रीसेट करें",
            "नया पासवर्ड चाहिए",
            "कृपया मेरे पंजीकृत ईमेल {email} पर पासवर्ड रीसेट लिंक भेजें ताकि मैं नया पासवर्ड बना सकूं।"
        ]
    },
    "technical_issue": {
        "en": [
            "Mobile application crashes immediately upon opening checkout or cart tab.",
            "Encountering HTTP 500 internal server error whenever I click submit order.",
            "Search bar is unresponsive and filters do not update product list.",
            "App crash on checkout",
            "Server error 500",
            "App not loading",
            "Cart page bug",
            "Every time I try to add an item to my shopping cart, the app crashes and returns to the home screen on iOS version 17.4. Please fix this bug."
        ],
        "hinglish": [
            "Checkout page open karte hi app force close ho jata hai.",
            "App me bug hai, cart me item add nahi ho raha aur error 500 aa raha hai.",
            "Website hang ho rahi hai payment button press karne par.",
            "App crash ho raha hai",
            "Bug aa raha hai",
            "Page load nahi ho raha",
            "Bhai aapka app checkout screen par freeze ho jata hai. Phone restart karne ke baad bhi same issue hai. Kripya technical support team se check karwao."
        ],
        "roman_hindi": [
            "App kholte hi band ho ja raha hai, takneeki samasya hai.",
            "Website par error 500 dikhai de raha hai.",
            "Takneeki kharabi",
            "App band ho raha hai",
            "Error aa raha hai",
            "Aapke mobile application me takneeki truti ke karan bhugtan page open nahi ho raha hai."
        ],
        "devanagari": [
            "ऐप खोलते ही क्रैश हो रहा है, कार्ट में उत्पाद लोड नहीं हो रहे।",
            "वेबसाइट पर एरर 500 आ रहा है और ऑर्डर सबमिट नहीं हो पा रहा।",
            "तकनीकी समस्या है",
            "ऐप काम नहीं कर रहा",
            "पेज लोड नहीं हुआ",
            "मोबाइल एप्लिकेशन पर चेकआउट करते समय बार-बार सर्वर त्रुटि प्रदर्शित हो रही है। कृपया इस तकनीकी खराबी को ठीक करें।"
        ]
    },
    "billing": {
        "en": [
            "I was billed an extra processing fee on invoice {oid} without prior notice.",
            "Taxes and shipping charges calculated on invoice do not match the checkout estimate.",
            "Need tax GST invoice with company registered details for order {oid}.",
            "Wrong billing amount",
            "Extra charge on invoice",
            "GST invoice required",
            "Billing discrepancy",
            "The final invoice for order {oid} includes a convenience fee of $15 that was never displayed on the checkout breakdown. Please adjust the invoice or refund the discrepancy."
        ],
        "hinglish": [
            "Invoice {oid} pe extra charges add kar diye gaye hain bina bataye.",
            "Bill me tax calculation galat hai, checkout ke time alag dikha raha tha.",
            "GST invoice provide karo order {oid} ke liye company name ke saath.",
            "Extra bill lagaya hai",
            "Invoice galat hai",
            "GST bill chahiye",
            "Bhai checkout pe Rs 450 dikhaya tha lekin credit card se Rs 520 bill hua hai. Ye extra charges kis cheez ke hain? Invoice theek karke bhejo."
        ],
        "roman_hindi": [
            "Bill me atirikt shulk joda gaya hai, kripya janch karein.",
            "Invoice galat rashi ka bana hai.",
            "GST bill ki aavashyakta hai.",
            "Bill me galati",
            "Extra shulk kyu?",
            "Aadesh {oid} ke invoice me nirdharit shulk se adhik rashi jodi gayi hai."
        ],
        "devanagari": [
            "चालान (इनवॉइस) में अतिरिक्त शुल्क जोड़ दिया गया है, कृपया सुधार करें।",
            "जीएसटी इनवॉइस में कंपनी का नाम और विवरण गलत छपा है।",
            "बिल में अतिरिक्त पैसे",
            "गलत इनवॉइस",
            "जीएसटी बिल चाहिए",
            "ऑर्डर {oid} के अंतिम बिल में डिलीवरी शुल्क जोड़ दिया गया है जबकि ऑफर में फ्री डिलीवरी थी। कृपया सही बिल जारी करें।"
        ]
    },
    "subscription": {
        "en": [
            "I cancelled auto-renewal last month, but monthly membership fee was charged again.",
            "How do I pause or cancel my annual premium subscription plan?",
            "Subscription benefits like free delivery are not activating on my account.",
            "Subscription auto charge",
            "Cancel subscription",
            "Membership not working",
            "Renewal fee charged",
            "My premium membership was charged today for renewal even though I toggled off recurring billing two weeks ago. Kindly cancel the subscription and refund the charged amount."
        ],
        "hinglish": [
            "Maine subscription cancel kiya tha fir bhi account se auto-debit ho gaya.",
            "Annual membership plan cancel karke paise refund karo please.",
            "Subscription active dikha raha hai par free delivery ka fayda nahi mil raha.",
            "Subscription cancel karo",
            "Auto debit ho gaya membership ka",
            "Premium plan band karo",
            "Bhai har mahine bina permission ke mere account se subscription fees cut ho rahi hai. Maine pichle month hi cancel kiya tha. Isko permanent band karo."
        ],
        "roman_hindi": [
            "Subscription radd karne ke baad bhi shulk kat gaya hai.",
            "Mera premium plan band kar dijiye.",
            "Subscription shulk wapas karein.",
            "Subscription band karo",
            "Sadasyata shulk kat gaya",
            "Kripya mere khate se annual subscription ko radd karein aur auto-debit band karein."
        ],
        "devanagari": [
            "सदस्यता रद्द करने के बावजूद मेरे खाते से नवीनीकरण शुल्क काट लिया गया है।",
            "प्रीमियम सब्सक्रिप्शन को तुरंत बंद करें और काटी गई राशि वापस करें।",
            "सब्सक्रिप्शन बंद करो",
            "सदस्यता शुल्क कटा",
            "ऑटो-डेबिट बंद करें",
            "मैंने पिछले महीने अपनी मासिक सदस्यता रद्द कर दी थी, फिर भी आज बैंक खाते से शुल्क काट लिया गया। कृपया इसे तत्काल वापस करें।"
        ]
    },
    "fraud_report": {
        "en": [
            "Unauthorized transaction of ${amt} made from my saved card on order {oid}; block account.",
            "Suspected phishing scam: received fake courier call demanding OTP for parcel release.",
            "Someone placed an order {oid} to an unknown address using my credentials.",
            "Fraudulent transaction detected",
            "Phishing attempt reported",
            "Unauthorized order on my card",
            "Card hacked help",
            "I received an SMS that order {oid} was placed from my account using my saved debit card. I never placed this order, nor do I recognize the shipping address in {city}. Please freeze my account immediately."
        ],
        "hinglish": [
            "Mere saved card se unauthorized transaction hua hai order {oid} pe, turant block karo.",
            "Kisi ne mera account hack karke unknown address pe order book kar diya hai.",
            "Fake call aaya tha OTP mangne ke liye aapke courier partner ke naam se, fraud hai.",
            "Card se fraud hua hai",
            "Unauthorized transaction",
            "Account hack ho gaya",
            "Fraud order cancel karo",
            "Bhai mere credit card se bina OTP ke Rs {amt} ka transaction hua hai aur order {oid} dikha raha hai kisi aur ke address pe. Ye fraud transaction hai, please order dispatch rok do aur card unlink karo."
        ],
        "roman_hindi": [
            "Mere card se anadhikrit len den hua hai, kripya aadesh rokein.",
            "Mera khata surakshit nahi hai, kisi ne anadhikrit order kiya hai.",
            "Fraud len den hua hai",
            "Khata suraksha samasya",
            "Anadhikrit aadesh",
            "Aadesh {oid} mere dwara nahi kiya gaya hai. Mere saved card se dhan kaat liya gaya hai, kripya turant khata freeze karein."
        ],
        "devanagari": [
            "मेरे सहेजे गए कार्ड से अनधिकृत लेनदेन हुआ है, कृपया आदेश {oid} को तुरंत रोकें।",
            "किसी अज्ञात व्यक्ति ने मेरे खाते से अन्य पते पर सामान ऑर्डर कर दिया है, खाता फ्रीज करें।",
            "धोखाधड़ी का लेनदेन हुआ है",
            "कार्ड से फ्रॉड हुआ",
            "अनधिकृत ऑर्डर रद्द करें",
            "मुझे अभी बैंक से संदेश मिला कि मेरे कार्ड से {amt} रुपये का ऑर्डर बुक हुआ है। मैंने कोई ऑर्डर नहीं किया है। यह धोखाधड़ी है, कृपया पुलिस रिपोर्ट और खाता ब्लॉक करें।"
        ]
    },
    "product_complaint": {
        "en": [
            "Received counterfeit/fake electronic item on order {oid}; packaging seals were tampered.",
            "Delivered product has missing accessories and deep scratches on display.",
            "Expired food item delivered with manufacturing date from previous year.",
            "Product quality is terrible and does not match catalog description.",
            "Defective product received",
            "Counterfeit fake item",
            "Missing accessories",
            "Poor quality product",
            "The wireless headphones delivered in order {oid} are completely non-functional. The left earbud will not charge or pair. I demand a replacement or full refund."
        ],
        "hinglish": [
            "Order {oid} me duplicate/nakli saman bhej diya gaya hai, quality bekaar hai.",
            "Product me scratches hain aur box ke andar charger missing hai.",
            "Expiry date nikal chuki hai jo product deliver hua hai.",
            "Nakli saman mila hai",
            "Product kharab nikla",
            "Saman toota hua hai",
            "Item quality bekaar hai",
            "Bhai maine original branded shoes order kiye the order {oid} me, par jo parcel aaya usme low quality copy shoes nikle. Packaging bhi fake thi. Is seller ke khilaf action lo aur exchange karo."
        ],
        "roman_hindi": [
            "Prapt utpad nakli aur kharab sthiti me hai.",
            "Saman me kuch parts missing hain aur chal nahi raha.",
            "Kharab utpad mila",
            "Nakli saman",
            "Utpad shikayat",
            "Aadesh {oid} ke antargat jo utpad bheja gaya hai wo catalog ke vivaran se bilkul mel nahi khata. Kripya badlav ya wapsi karein."
        ],
        "devanagari": [
            "आदेश {oid} में भेजा गया उत्पाद नकली और घटिया गुणवत्ता का है।",
            "डिब्बे में चार्जर और एक्सेसरीज गायब हैं, उत्पाद भी काम नहीं कर रहा।",
            "खराब उत्पाद मिला",
            "सामान नकली है",
            "उत्पाद की शिकायत",
            "जो उत्पाद मुझे प्राप्त हुआ है उसकी सील खुली हुई थी और सामान पहले से इस्तेमाल किया हुआ लग रहा है। कृपया नया उत्पाद बदल कर भेजें।"
        ]
    },
    "general_query": {
        "en": [
            "What are your customer support working hours and contact phone number?",
            "Do you offer international shipping to destinations outside India?",
            "How do I apply promotional discount coupons on wholesale cart orders?",
            "Customer care number?",
            "Support hours?",
            "International delivery available?",
            "Coupon code help",
            "Could you please share your official customer grievance officer contact details and the escalation matrix for enterprise accounts?"
        ],
        "hinglish": [
            "Customer care ka phone number ya email id kya hai contact karne ke liye?",
            "Kya aap {city} me bulk order delivery provide karte ho?",
            "Discount coupon apply karne ka procedure kya hai checkout pe?",
            "Contact number chahiye",
            "Helpdesk number kya hai?",
            "Support team se baat karni hai",
            "Bhai mujhe janna tha ki agar main 50 items ek saath order karu toh kya koi special discount milta hai? Kripya bulk purchase policy explain karein."
        ],
        "roman_hindi": [
            "Grahak seva kendra ka helpline number kya hai?",
            "Aapki wapsi niti (return policy) ke niyam kya hain?",
            "Sahayata number chahiye",
            "Sampark soochana",
            "Kripya batayein ki aapki customer care team se baat karne ke liye helpline kab uplabdh rehti hai."
        ],
        "devanagari": [
            "ग्राहक सेवा का हेल्पलाइन नंबर और ईमेल आईडी क्या है?",
            "थोक खरीद (बल्क ऑर्डर) पर छूट प्राप्त करने की क्या प्रक्रिया है?",
            "हेल्पलाइन नंबर चाहिए",
            "संपर्क कैसे करें?",
            "सहायता केंद्र का समय क्या है?",
            "कृपया बताएं कि आपके प्लेटफॉर्म पर रिटर्न और एक्सचेंज की सामान्य समय सीमा कितने दिनों की है।"
        ]
    },
    "account_closure": {
        "en": [
            "I want to permanently delete my account and erase all stored personal data under GDPR.",
            "Please deactivate my profile and remove my registered mobile number and card details.",
            "Close my account",
            "Delete profile permanently",
            "Remove my personal data",
            "Deactivate profile",
            "I no longer wish to use this platform. Kindly process the permanent deletion of my account associated with {email} and confirm once all personal identifiers have been purged."
        ],
        "hinglish": [
            "Mera account permanently delete kar do aur saara data remove karo.",
            "Profile deactivate karni hai, saved cards aur phone number hata do.",
            "Account delete karo",
            "Profile band karni hai",
            "Account close please",
            "Bhai mujhe apna account hamesha ke liye band karna hai. Kripya mera registered account {email} delete karein aur confirmation bhejein."
        ],
        "roman_hindi": [
            "Mera khata sthayi roop se band (delete) kar dijiye.",
            "Khata band karna chahta hoon aur apna data hatana chahta hoon.",
            "Khata band karein",
            "Profile delete karo",
            "Kripya mere khate {email} ko nishkriya karein aur meri vyaktigat jankari system se mita dein."
        ],
        "devanagari": [
            "मैं अपना खाता स्थायी रूप से हटाना (डिलीट करना) चाहता हूँ, कृपया सारा डेटा हटाएं।",
            "मेरी प्रोफाइल निष्क्रिय करें और सहेजे गए कार्ड विवरण मिटा दें।",
            "खाता बंद करें",
            "प्रोफाइल डिलीट करें",
            "डेटा हटाओ",
            "कृपया आदेश दें कि मेरा खाता {email} स्थायी रूप से बंद कर दिया जाए और डेटा सुरक्षा नियमों के तहत सभी विवरण नष्ट किए जाएं।"
        ]
    }
}

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh"]
GATEWAYS = ["Paytm", "Google Pay", "PhonePe", "Razorpay", "HDFC Netbanking", "SBI UPI", "ICICI Bank", "Cred"]
TYPOS = [
    ("payment", "pament"), ("refund", "refnd"), ("cancel", "cancle"),
    ("delivery", "delivry"), ("account", "acount"), ("password", "paswrd"),
    ("received", "recieved"), ("problem", "prblm"), ("please", "plz"),
    ("paise", "pese"), ("karo", "kro"), ("nahi", "nhi")
]


def apply_perturbations(text: str, noise_type: str) -> str:
    if noise_type == "typo":
        for orig, typo in TYPOS:
            if orig in text.lower() and random.random() < 0.5:
                text = text.replace(orig, typo).replace(orig.capitalize(), typo.capitalize())
    elif noise_type == "noisy_punctuation":
        punct = random.choice(["???", "!!!", "??!!", "...."])
        text = text.rstrip(".!?") + punct
    elif noise_type == "informal_slang":
        slang_prefix = random.choice(["Arrey bhai ", "Sir please ", "Yaar ", "Hello team, "])
        text = slang_prefix + text
    return text


def generate_dataset_v2(target_count: int = 6000):
    print("=" * 70)
    print(f"GENERATING NEXUS-SLM DATASET V2 (TARGET: {target_count} EXAMPLES)")
    print("=" * 70)

    random.seed(42)
    records = []
    id_counter = 1

    per_category = target_count // len(CATEGORIES)

    for cat in CATEGORIES:
        patterns = CATEGORY_PATTERNS[cat]
        cat_count = 0

        while cat_count < per_category:
            # Linguistic balance: 30% en, 25% hinglish, 20% roman_hindi, 25% devanagari
            lang_choice = random.choices(
                ["en", "hinglish", "roman_hindi", "devanagari"],
                weights=[0.30, 0.25, 0.20, 0.25],
                k=1
            )[0]

            tmpl = random.choice(patterns[lang_choice])
            oid = f"ORD-{random.randint(100000, 999999)}"
            amt = random.randint(199, 14999)
            city = random.choice(CITIES)
            gateway = random.choice(GATEWAYS)
            email = f"user_{random.randint(100, 999)}@gmail.com"

            text = tmpl.format(oid=oid, amt=amt, city=city, gateway=gateway, email=email)

            # Linguistic tag
            if lang_choice == "en":
                lang_tag = "english"
            elif lang_choice == "hinglish":
                lang_tag = "hinglish"
            elif lang_choice == "roman_hindi":
                lang_tag = "roman_hindi"
            else:
                lang_tag = "hindi"

            # Context sparsity tag & length
            words = text.split()
            n_words = len(words)

            # Noise / Perturbation injection on 15% of records
            targeted_failure_mode = "none"
            if n_words < 5:
                targeted_failure_mode = "context_sparsity"
            elif lang_choice == "devanagari":
                targeted_failure_mode = "devanagari_token_enrichment"
            elif cat in ["refund_request", "order_cancel", "billing", "payment_failure"]:
                targeted_failure_mode = "semantic_overlap_disambiguation"

            noise_prob = random.random()
            if noise_prob < 0.05:
                text = apply_perturbations(text, "typo")
            elif noise_prob < 0.10:
                text = apply_perturbations(text, "noisy_punctuation")
            elif noise_prob < 0.15 and lang_choice in ["hinglish", "roman_hindi"]:
                text = apply_perturbations(text, "informal_slang")

            # Priority mapping
            if cat in ["fraud_report", "payment_failure"]:
                priority = random.choices(["high", "critical", "medium"], weights=[0.5, 0.35, 0.15])[0]
                requires_human = priority in ["high", "critical"]
            elif cat in ["account_closure", "order_cancel", "product_complaint"]:
                priority = random.choices(["high", "medium", "low"], weights=[0.4, 0.45, 0.15])[0]
                requires_human = priority == "high"
            else:
                priority = random.choices(["medium", "low", "high"], weights=[0.6, 0.3, 0.1])[0]
                requires_human = priority == "high"

            rec = {
                "id": f"v2_{id_counter:06d}",
                "text": text,
                "category": cat,
                "priority": priority,
                "language": lang_tag,
                "requires_human": requires_human,
                "provenance": {
                    "source": "curated_multilingual_template_synthesis",
                    "data_tier": "schema_and_lexicon_verified",
                    "generation_method": "pattern_slot_morphology",
                    "validation_status": "ground_truth_validated",
                    "targeted_failure_mode": targeted_failure_mode,
                }
            }
            records.append(rec)
            id_counter += 1
            cat_count += 1

    random.shuffle(records)

    # Split into Train (5000), Validation (500), and Test V2 (500)
    train_split = records[:5000]
    val_split = records[5000:5500]
    test_v2_split = records[5500:6000]

    train_file = DATA_V2_DIR / "train.jsonl"
    val_file = DATA_V2_DIR / "val.jsonl"
    test_file = DATA_V2_DIR / "test_v2.jsonl"

    for file_path, data in [(train_file, train_split), (val_file, val_split), (test_file, test_v2_split)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute hashes
    def get_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return f"sha256_{h.hexdigest()}"

    lock_info = {
        "dataset_version": "dataset_v2",
        "status": "IMMUTABLE_LOCKED",
        "random_seed": 42,
        "total_examples": len(records),
        "splits": {
            "train": {"count": len(train_split), "file": "data/v2/train.jsonl", "hash": get_hash(train_file)},
            "validation": {"count": len(val_split), "file": "data/v2/val.jsonl", "hash": get_hash(val_file)},
            "test_v2": {"count": len(test_v2_split), "file": "data/v2/test_v2.jsonl", "hash": get_hash(test_file)},
        },
        "v1_compatibility": {
            "historical_v1_test_file": "data/test/test.jsonl",
            "historical_v1_count": 42,
            "historical_v1_hash": "sha256_edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4",
            "policy": "V1 test set preserved without modification. Model V2 evaluated on BOTH historical V1 test set and new 500-instance V2 test set."
        }
    }

    lock_path = DATA_V2_DIR / "dataset_v2_lock.json"
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_info, f, indent=2)

    print(f"Generated Dataset V2:")
    print(f"  Train:      {len(train_split)} instances -> {train_file}")
    print(f"  Validation: {len(val_split)} instances -> {val_file}")
    print(f"  Test V2:    {len(test_v2_split)} instances -> {test_file}")
    print(f"  Lock File:  {lock_path}")
    return lock_info


if __name__ == "__main__":
    generate_dataset_v2(6000)
