"""
qa_engine.py
Multi-language Q&A Assistant Engine for Kavacham Traffic Safety Intelligence.
Supports 7 Languages: Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn),
Malayalam (ml), Bengali (bn), English (en).

Features:
- Unicode Language Auto-Detection + Manual Language Selection
- Audio Input Speech Recognition (transcription from microphone)
- In-depth answers with exact conflict metrics, zones, costs, and infrastructure plans
- Multilingual Speech Synthesis via gTTS (audio playback in matching language)
- Dynamic contextual visual triggers (flowcharts, tables, risk heatmaps, storyboard cards, video clips)
"""

import os
import re
import tempfile
from gtts import gTTS

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

LANGUAGES = {
    "Auto Detect": None,
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "English": "en"
}

LANG_FLAGS = {
    "hi": "🇮🇳 हिन्दी",
    "ta": "🇮🇳 தமிழ்",
    "te": "🇮🇳 తెలుగు",
    "kn": "🇮🇳 ಕನ್ನಡ",
    "ml": "🇮🇳 മലയാളം",
    "bn": "🇮🇳 বাংলা",
    "en": "🌐 English"
}


def detect_language(text):
    """Detect language based on Unicode codepoints."""
    if not text:
        return "en"
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            return "hi"  # Devanagari (Hindi)
        elif 0x0B80 <= cp <= 0x0BFF:
            return "ta"  # Tamil
        elif 0x0C00 <= cp <= 0x0C7F:
            return "te"  # Telugu
        elif 0x0C80 <= cp <= 0x0CFF:
            return "kn"  # Kannada
        elif 0x0D00 <= cp <= 0x0D7F:
            return "ml"  # Malayalam
        elif 0x0980 <= cp <= 0x09FF:
            return "bn"  # Bengali
    return "en"


def transcribe_audio(audio_bytes):
    """Transcribes audio using SpeechRecognition."""
    if not HAS_SR:
        return None, "Speech recognition library not installed."

    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio_bytes)
            tf_name = tf.name

        with sr.AudioFile(tf_name) as source:
            audio_data = r.record(source)

        # Try Google Web Speech API with auto recognition
        text = r.recognize_google(audio_data)
        os.unlink(tf_name)
        lang = detect_language(text)
        return text, lang
    except Exception as e:
        if 'tf_name' in locals() and os.path.exists(tf_name):
            os.unlink(tf_name)
        return None, str(e)


def generate_audio_response(text, lang_code, output_path):
    """Generate high quality speech audio using gTTS."""
    try:
        # Clean markdown styling for speech
        clean_text = re.sub(r'[*_#`\n\r]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Truncate to first 300 characters to keep audio brief and responsive
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."

        tts = gTTS(text=clean_text, lang=lang_code if lang_code in ['hi', 'ta', 'te', 'kn', 'ml', 'bn', 'en'] else 'en')
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"gTTS error: {e}")
        return None


def generate_safety_answer(question, selected_lang=None):
    """
    Synthesizes an in-depth, verified factual response based on the Kavacham 20s junction audit.
    Includes technical specifics for officers and clear guidelines for individuals.
    Returns: (text_answer, detected_lang, visual_triggers)
    """
    lang = selected_lang if selected_lang else detect_language(question)
    q_lower = question.lower()

    # Visual trigger flags:
    # 'conflicts', 'pedestrian', 'avoid', 'infra', 'c70'
    visual_triggers = []

    # Classify intent & context triggers
    is_conflict_query = any(k in q_lower for k in [
        "potential conflicts", "conflicts", "what conflicts", "धमकी", "விபத்து", "प्रमाదం", "ಅಪಘಾತ", "அபாயம்", "conflict"
    ])
    is_pedestrian_query = any(k in q_lower for k in [
        "pedestrian", "pedestrians", "पैदल", "பாதசாரி", "பாதசாரிகள்", "పాదచారులు", "ಪಾದಚಾರಿ", "കാൽനട", "পথচারী"
    ])
    is_avoid_query = any(k in q_lower for k in [
        "avoid", "how to avoid", "बचें", "எப்படி தவிர்", "ఎలా నివారించాలి", "ಹೇಗೆ ತಪ್ಪಿಸುವುದು", "സുരക്ഷ", "এড়াতে"
    ])
    is_infra_query = any(k in q_lower for k in [
        "government", "build", "infra", "what to build", "cost", "budget", "सरकार", "அரசு", "നിർമ്മാണം", "সরকার"
    ])
    is_c70_query = any(k in q_lower for k in ["70", "conflict 70", "near miss", "critical", "close pass"])

    if is_conflict_query or ("what" in q_lower and "potential" in q_lower):
        visual_triggers = ["table_148", "types_pie", "risk_map", "flowchart_close_pass", "video_c70"]
        if lang == "hi":
            answer = (
                "**कवचम विश्लेषण परिणाम: कुल 148 संभावित टकराव (Potential Conflicts) पाए गए हैं।**\n\n"
                "• **सत्यापित गंभीर घटना (Verified Real Near-Miss):** संघर्ष #70 (Conflict #70) - दक्षिण की ओर जा रही मोटरसाइकिल #73 और स्कूटर #83 के बीच 38.7 px/s की गति पर दूरी 72px से घटकर मात्र 24px (3px का शारीरिक फासला) रह गई। (TTC 1.37s, गंभीरता 6/6)।\n"
                "• **टकराव प्रकारों का विवरण (Top Categories):**\n"
                "  1. ऑटो / पैदल यात्री: 25 टकराव (भारी गति असंतुलन)\n"
                "  2. ऑटो / मोटरसाइकिल: 21 टकराव (गोलचक्कर पर अचानक मुड़ना)\n"
                "  3. मोटरसाइकिल / पैदल यात्री: 19 टकराव\n"
                "  4. पैदल / पैदल यात्री: 18 टकराव\n"
                "  5. कार / दोपहिया वाहन: 16 टकराव\n"
                "  6. दोपहिया / दोपहिया वाहन: 15 टकराव (जिसमें #70 शामिल है)\n"
                "• **जोखिम क्षेत्र (Risk Zones):** केंद्र गोलचक्कर में सबसे अधिक 68 टकराव (46%), दाईं ओर 45, और बाईं ओर 35 टकराव दर्ज किए गए।"
            )
        elif lang == "ta":
            answer = (
                "**கவசம் ஆய்வு முடிவுகள்: சந்திப்பில் மொத்தம் 148 சாத்தியமான முரண்பாடுகள் (Potential Conflicts) கண்டறியப்பட்டுள்ளன.**\n\n"
                "• **உறுதிசெய்யப்பட்ட விபத்து நிலை (Conflict #70):** மோட்டார் சைக்கிள் #73 மற்றும் ஸ்கூட்டர் #83 இடையே இடைவெளி 72px இலிருந்து 24px ஆகக் குறைந்தது (3px மட்டுமே இடைவெளி). TTC 1.37 வினாடிகள், தீவிரம் 6/6.\n"
                "• **முக்கிய முரண்பாட்டு வகைகள்:**\n"
                "  1. ஆட்டோ / பாதசாரி: 25 முரண்பாடுகள் (அதிக ஆபத்து)\n"
                "  2. ஆட்டோ / இருசக்கர வாகனம்: 21 முரண்பாடுகள்\n"
                "  3. இருசக்கர வாகனம் / பாதசாரி: 19 முரண்பாடுகள்\n"
                "  4. கார் / இருசக்கர வாகனம்: 16 முரண்பாடுகள்\n"
                "  5. இருசக்கர வாகனம் / இருசக்கர வாகனம்: 15 முரண்பாடுகள் (#70 உட்பட)\n"
                "• **அபாய மண்டலங்கள்:** மையப் பகுதியில் 68 முரண்பாடுகள் (46%), வலதுபுறம் 45, இடதுபுறம் 35."
            )
        elif lang == "te":
            answer = (
                "**కవచం విశ్లేషణ ఫలితాలు: ఈ జంక్షన్‌లో మొత్తం 148 సంభావ్య సంఘర్షణలు (Potential Conflicts) గుర్తించబడ్డాయి.**\n\n"
                "• **నిజమైన ప్రమాద సంఘటన (#70):** మోటార్‌సైకిల్ #73 మరియు స్కూటర్ #83 మధ్య దూరం 72px నుండి 24px కు తగ్గింది (కేవలం 3px క్లియరెన్స్). TTC 1.37 సెకన్లు, తీవ్రత 6/6.\n"
                "• **ముఖ్యమైన వర్గాలు:** ఆటో/పాదచారులు 25, ఆటో/బైక్ 21, బైక్/పాదచారులు 19, కార్/బైక్ 16, బైక్/బైక్ 15.\n"
                "• **ప్రమాద మండలాలు:** రౌండ్‌అబౌట్ కేంద్రంలో అత్యధికంగా 68 (46%), కుడివైపు 45, ఎడమవైపు 35."
            )
        elif lang == "kn":
            answer = (
                "**ಕವಚಂ ವಿಶ್ಲೇಷಣೆ: ಒಟ್ಟು 148 ಸಂಭವನೀಯ ಘರ್ಷಣೆಗಳು (Potential Conflicts) ಪತ್ತೆಯಾಗಿವೆ.**\n\n"
                "• **ನೈಜ ಅಪಘಾತ ಘಟನೆ (#70):** ಮೋಟಾರ್ ಸೈಕಲ್ #73 ಮತ್ತು ಸ್ಕೂಟರ್ #83 ನಡುವೆ ಕೇವಲ 3px ಅಂತರ ಉಳಿದಿತ್ತು (ದೂರ 72px ನಿಂದ 24px ಗೆ ಇಳಿಕೆ, TTC 1.37 ಸೆಕೆಂಡ್, ತೀವ್ರತೆ 6/6).\n"
                "• **ಪ್ರಮುಖ ವರ್ಗಗಳು:** ಆಟೋ/ಪಾದಚಾರಿ 25, ಆಟೋ/ಬೈಕ್ 21, ಬೈಕ್/ಪಾದಚಾರಿ 19, ಕಾರ್/ಬೈಕ್ 16.\n"
                "• **ಅಪಾಯ ವಲಯಗಳು:** ಕೇಂದ್ರ ವೃತ್ತದಲ್ಲಿ 68 ಘರ್ಷಣೆಗಳು (46%), ಬಲಭಾಗದಲ್ಲಿ 45, ಎಡಭಾಗದಲ್ಲಿ 35."
            )
        elif lang == "ml":
            answer = (
                "**കവചം വിശകലന ഫലങ്ങൾ: ജംഗ്ഷനിൽ ആകെ 148 സാധ്യതയുള്ള സംഘർഷങ്ങൾ (Potential Conflicts) കണ്ടെത്തി.**\n\n"
                "• **യഥാർത്ഥ അപകട സാഹചര്യം (#70):** മോട്ടോർസൈക്കിൾ #73 ഉം സ്കൂട്ടർ #83 ഉം തമ്മിലുള്ള അകലം 72px ൽ നിന്ന് 24px ആയി കുറഞ്ഞു (വെറും 3px ഗ്യാപ്പ്, TTC 1.37 സെക്കൻഡ്, തീവ്രത 6/6).\n"
                "• **പ്രധാന തരങ്ങൾ:** ഓട്ടോ/കാൽനടയാത്രക്കാർ 25, ഓട്ടോ/ബൈക്ക് 21, ബൈക്ക്/കാൽനടയാത്രക്കാർ 19, കാർ/ബൈക്ക് 16.\n"
                "• **അപകട മേഖലകൾ:** റൗണ്ട്എബൗട്ട് കേന്ദ്രത്തിൽ 68 (46%), വലതുഭാഗത്ത് 45, ഇടതുഭാഗത്ത് 35."
            )
        elif lang == "bn":
            answer = (
                "**কবচ বিশ্লেষণ রিপোর্ট: জংশনে মোট ১৪৮টি সম্ভাব্য সংঘাত (Potential Conflicts) শনাক্ত হয়েছে।**\n\n"
                "• **প্রকৃত দুর্ঘটনা পরিস্থিতি (#70):** মোটরসাইকেল #73 এবং স্কুটার #83 এর দূরত্ব ৭২ পিক্সেল থেকে কমে মাত্র ২৪ পিক্সেল (৩ পিক্সেল শারীরিক ব্যবধান) এ এসে পৌঁছায় (TTC 1.37s, তীব্রতা 6/6)।\n"
                "• **প্রধান সংঘাতের ধরন:** অটো/পথচারী ২৫, অটো/বাইক ২১, বাইক/পথচারী ১৯, গাড়ি/বাইক ১৬।\n"
                "• **ঝুঁকিপূর্ণ অঞ্চল:** কেন্দ্রের গোলচত্বরে সর্বাধিক ৬৮টি (৪৬%), ডানে ৪৫টি এবং বামে ৩৫টি।"
            )
        else:
            answer = (
                "**Kavacham Analysis Findings: 148 Potential Proximity Conflicts Identified.**\n\n"
                "• **Critical Real Near-Miss (#70):** Motorcycle #73 vs Scooter #83 in the centre roundabout. Closing speed 38.7 px/s compressed clearance from 72px down to 24px with bare 3px physical margin at frames 127-145 (TTC 1.37s, Severity 6/6).\n"
                "• **Top Conflict Categories:**\n"
                "  1. Auto-Rickshaw / Pedestrian: 25 conflicts (Vulnerable mass imbalance)\n"
                "  2. Auto-Rickshaw / Motorcycle: 21 conflicts (Unsignaled roundabout weaving)\n"
                "  3. Motorcycle / Pedestrian: 19 conflicts (Crossing intercept)\n"
                "  4. Pedestrian / Pedestrian: 18 conflicts (Sidewalk bottleneck)\n"
                "  5. Car / Motorcycle: 16 conflicts (Exit acceleration squeeze)\n"
                "  6. Two-Wheeler / Two-Wheeler: 15 conflicts (Includes Conflict #70 near-miss)\n"
                "• **Spatial Zoning:** Centre Roundabout High Risk = 68 (46%), Right Exit Medium = 45 (30%), Left Ingress Low = 35 (24%)."
            )

    elif is_pedestrian_query:
        visual_triggers = ["flowchart_ped_crossing", "map_left_zone", "infra_zebra"]
        if lang == "hi":
            answer = (
                "**पैदल यात्रियों के दुर्घटना जोखिम क्षेत्र और कारण:**\n\n"
                "• **स्थान:** सबसे अधिक पैदल दुर्घटनाएँ बाईं ओर (Left Ingress Zone) और दक्षिण प्रवेश द्वार पर दर्ज हुई हैं, जहाँ 25 ऑटो/पैदल और 19 बाइक/पैदल टकराव हुए।\n"
                "• **मुख्य कारण:** जेब्रा क्रॉसिंग और स्पीड ब्रेकर का न होना। पैदल यात्री सीधे वाहनों के रास्ते में आते हैं।\n"
                "• **निवारण उपाय:** 25 हजार रुपये की लागत से 75mm उठा हुआ स्पीड टेबल और जेब्रा क्रॉसिंग बनाना, जिससे 40% पैदल दुर्घटनाएँ तुरंत रुकेंगी।"
            )
        elif lang == "ta":
            answer = (
                "**பாதசாரிகள் விபத்துக்குள்ளாகும் பகுதிகள் மற்றும் காரணங்கள்:**\n\n"
                "• **இடங்கள்:** இடதுபுறம் (Left Ingress Zone) மற்றும் தெற்கு நுழைவுப் பகுதியில் பாதசாரிகள் அதிகம் பாதிக்கப்படுகின்றனர் (25 ஆட்டோ/பாதசாரி மற்றும் 19 பைக்/பாதசாரி முரண்பாடுகள்).\n"
                "• **காரணம்:** வரிக்குதிரை கோடுகள் (Zebra crossing) மற்றும் வேகத்தடைகள் இல்லாததால் வாகனங்கள் 35 கி.மீ வேகத்தில் நுழைகின்றன.\n"
                "• **தீர்வு:** ரூ.25,000 செலவில் ஸ்பீட் டேபிள் மற்றும் ஜீப்ரா கிராசிங் அமைப்பது 45% விபத்துகளைக் குறைக்கும்."
            )
        elif lang == "te":
            answer = (
                "**పాదచారుల ప్రమాద ప్రాంతాలు మరియు పరిష్కారాలు:**\n\n"
                "• **స్థానాలు:** ఎడమ ప్రవేశ జోన్ (Left Zone) లో పాదచారులు ఎక్కువగా ప్రమాదాల బారిన పడుతున్నారు (25 ఆటో/పాదచారుల సంఘర్షణలు).\n"
                "• **పరిష్కారం:** రూ. 25,000 ఖర్చుతో జీబ్రా క్రాసింగ్ మరియు స్పీడ్ బ్రేకర్ నిర్మించడం ద్వారా 45% ప్రమాదాలను నివారించవచ్చు."
            )
        elif lang == "kn":
            answer = (
                "**ಪಾದಚಾರಿಗಳ ಅಪಘಾತ ವಲಯಗಳು:** ಎಡಭಾಗದ ಪ್ರವೇಶ ದ್ವಾರದಲ್ಲಿ 25 ಆಟೋ/ಪಾದಚಾರಿ ಘರ್ಷಣೆಗಳು ದಾಖಲಾಗಿವೆ. ರೂ. 25,000 ವೆಚ್ಚದಲ್ಲಿ ಜೀಬ್ರಾ ಕ್ರಾಸಿಂಗ್ ಮತ್ತು ವೇಗ ನಿಯಂತ್ರಕ ಅಳವಡಿಸುವುದು ಸೂಕ್ತ ಪರಿಹಾರ."
            )
        elif lang == "ml":
            answer = (
                "**കാൽനടയാത്രക്കാരുടെ അപകട മേഖലകൾ:** ഇടതുവശത്തെ പ്രവേശന കവാടത്തിൽ 25 ഓട്ടോ/കാൽനട സംഘർഷങ്ങൾ രേഖപ്പെടുത്തിയിട്ടുണ്ട്. 25,000 രൂപ ചെലവിൽ സീബ്ര ക്രോസിംഗും സ്പീഡ് ബ്രേക്കറും സ്ഥാപിച്ചാൽ 45% അപകടങ്ങൾ തടയാം."
            )
        elif lang == "bn":
            answer = (
                "**পথচারী দুর্ঘটনা এলাকা:** বাম পাশের প্রবেশদ্বারে ২৫টি অটো/পথচারী সংঘাত ঘটেছে। ২৫,০০০ টাকা ব্যয়ে স্পিড ব্রেকার এবং জেব্রা ক্রসিং স্থাপন করলে ৪৫% ঝুঁকি হ্রাস পাবে।"
            )
        else:
            answer = (
                "**Pedestrian Vulnerability Analysis:**\n\n"
                "• **High-Risk Zones:** Southern ingress and Left Zone apron account for 25 Auto/Pedestrian and 19 Motorcycle/Pedestrian conflicts.\n"
                "• **Key Mechanisms:** Lack of painted zebra markings, no physical refuge island, and vehicle approach speeds exceeding 35 km/h.\n"
                "• **Recommended Fix:** Raised Table Zebra Crossing with 75mm ramp (Rs. 25,000) reduces pedestrian conflict probability by 45%."
            )

    elif is_avoid_query:
        visual_triggers = ["flowchart_close_pass", "flowchart_ped_crossing", "storyboard_gallery"]
        if lang == "hi":
            answer = (
                "**व्यक्तिगत सुरक्षा नियम (How Individuals Can Avoid Accidents):**\n\n"
                "• **पैदल यात्री:** कभी भी गोलचक्कर के बीच से न गुजरें। केवल किनारे से जाएं और वाहनों से 1.2 मीटर का सुरक्षित फासला रखें।\n"
                "• **दोपहिया वाहन चालक:** गोलचक्कर में गति 20 किमी/घंटा से कम रखें और साथी वाहनों से कम से कम 1.5 मीटर की दूरी बनाए रखें (संघर्ष #70 जैसा जोखिम न लें)।\n"
                "• **ऑटो/कार चालक:** मोड़ से 30 मीटर पहले इंडिकेटर दें और चौराहे पर अचानक ब्रेक या सवारी न बैठाएं।"
            )
        elif lang == "ta":
            answer = (
                "**விபத்துகளைத் தவிர்க்கும் வழிகள்:**\n\n"
                "• **பாதசாரிகள்:** நடுவட்டத்தின் ஊடே குறுக்கே நடக்காதீர்கள். ஓரங்களில் அமைக்கப்பட்டுள்ள வழிகளை மட்டுமே பயன்படுத்துங்கள்.\n"
                "• **இருசக்கர வாகன ஓட்டிகள்:** சந்திப்பில் வேகத்தை 20 கி.மீ ஆகக் குறைக்கவும்; அருகிலுள்ள வாகனத்திற்கு 1.5 மீ இடைவெளி விடவும்.\n"
                "• **ஆட்டோ ஓட்டுநர்கள்:** சந்திப்பில் இருந்து 20 மீ தூரத்திற்குள் பயணிகளை ஏற்றவோ இறக்கவோ கூடாது."
            )
        else:
            answer = (
                "**Operational Avoidance Protocols for Individuals:**\n\n"
                "• **Pedestrians:** Do not cross diagonally through the roundabout core. Wait at median safety buffers.\n"
                "• **Two-Wheelers:** Maintain >= 1.5m lateral clearance to avoid side-by-side pinch points (as seen in Conflict #70). Throttle down to < 20 km/h.\n"
                "• **Auto & Car Drivers:** Yield right of way to pedestrians at the ingress and never halt within 20m of junction corners."
            )

    elif is_infra_query or "build" in q_lower:
        visual_triggers = ["storyboard_gallery", "infra_plan_table", "anim_video"]
        if lang == "hi":
            answer = (
                "**प्रस्तावित अवसंरचना मास्टरप्लान (कुल बजट: ₹53,000 - 1 सप्ताह में पूर्ण):**\n\n"
                "1. **लेन मार्किंग (Lane Marking 100m):** केंद्र में सफेद थर्मोप्लास्टिक पट्टी (लागत: ₹15,000) - 15 बाइक-बाइक टकराव रोकेगी (-30%)।\n"
                "2. **उठा हुआ जेब्रा क्रॉसिंग + स्पीड टेबल:** दक्षिणी प्रवेश पर (लागत: ₹25,000) - 46 पैदल टकरावों को 40% तक कम करेगी।\n"
                "3. **पैदल रिफ्यूज आइलैंड:** बीच में सुरक्षित प्रतीक्षा स्थल (लागत: ₹8,000)।\n"
                "4. **उत्तल दर्पण (Convex Mirror 80cm) + रंबल स्ट्रिप्स:** अंधे मोड़ पर (लागत: ₹13,000) - दाईं ओर के 45 टकराव सुरक्षित करेगी।\n"
                "• **परिणाम:** कुल 148 टकराव घटकर 100 रह जाएंगे (30% समग्र सुरक्षा सुधार)।"
            )
        elif lang == "ta":
            answer = (
                "**அரசு செயல்படுத்த வேண்டிய உள்கட்டமைப்பு திட்டம் (மொத்த செலவு: ₹53,000):**\n\n"
                "1. **வெள்ளை கோடு லேன் மார்க்கிங் 100மீ:** மையப்பகுதியில் (செலவு ₹15,000) - இருசக்கர வாகன முரண்பாடுகளைத் தடுக்கும்.\n"
                "2. **ஸ்பீட் டேபிள் + ஜீப்ரா கிராசிங்:** தெற்குப் பகுதியில் (செலவு ₹25,000) - 46 பாதசாரி விபத்துகளை 45% குறைக்கும்.\n"
                "3. **பாதசாரி பாதுகாப்பு தடுப்பு (Refuge Island):** சாலையின் நடுவே (செலவு ₹8,000).\n"
                "4. **குவிந்த கண்ணாடி (Convex Mirror 80cm) + ரம்பிள் ஸ்ட்ரிப்ஸ்:** வலதுபுறம் (செலவு ₹13,000).\n"
                "• **முடிவு:** 1 வாரத்தில் 30% ஒட்டுமொத்த ஆபத்து குறையும்."
            )
        else:
            answer = (
                "**Tactical Civil Engineering Masterplan (Total Cost: Rs. 53,000 INR | 7 Days):**\n\n"
                "1. **Lane Marking White 100m + Solar Delineators (Rs. 15,000):** High priority in roundabout core. Eliminates two-wheeler close pass weave (Conflict #70).\n"
                "2. **Raised Table Zebra Crossing + Speed Breaker (Rs. 25,000):** Protects 46 pedestrian conflicts at southern ingress (-45% risk).\n"
                "3. **Refuge Barricade Safe Island (Rs. 8,000):** Provides median physical buffer preventing mid-crossing stranding.\n"
                "4. **Convex Mirror (80cm) + Rumble Strips (Rs. 13,000):** Secures 45 conflicts at the blind right exit.\n"
                "• **Projected Efficiency:** Decreases total conflicts from 148 down to 100 (30% reduction) and permanently eliminates critical near-misses."
            )

    else:
        # Default detailed overview
        visual_triggers = ["table_148", "types_pie", "risk_map"]
        answer = (
            f"**Kavacham Intelligent Safety Analysis (Responded in {LANG_FLAGS.get(lang, 'English')}):**\n\n"
            "This junction exhibits 148 potential geometric conflicts across 93 tracked road users over 20 seconds. "
            "Multi-modal verification confirmed 1 critical near-miss (Conflict #70: Motorcycle #73 vs Scooter #83, 3px gap, TTC 1.37s). "
            "A tactical 4-part civil engineering package costing Rs. 53,000 will reduce overall junction conflict probability by 30% to 35%."
        )

    return answer, lang, visual_triggers

