"""Vocabularies for SynJama.

Names are common given names plus neutral second names compiled by the team;
they are not taken from any real register. Caste-indicative surnames are
deliberately left out (report 8, 'Sensitive attributes').
"""
from __future__ import annotations

MALE_GIVEN = [
    "राम", "हरी", "मोहन", "सोहन", "बलवंत", "गुरमीत", "सुरेंद्र", "राजेश", "महेंद्र", "जगदीश",
    "रमेश", "सुरेश", "दिनेश", "नरेश", "विजय", "अजय", "संजय", "राजबीर", "धर्मबीर", "जसवंत",
    "करतार", "बचन", "प्रीतम", "गुरदीप", "हरभजन", "सतपाल", "ओम", "प्रेम", "किशन", "रामकिशन",
    "शिव", "भगवान", "रणजीत", "कुलदीप", "मनजीत", "सुखदेव", "बलदेव", "जोगिंदर", "रघुबीर", "चंदर",
    "दलीप", "कृष्ण", "मुकेश", "राकेश", "अशोक", "अनिल", "सुनील", "पवन", "नरेंद्र", "बिशन",
]
FEMALE_GIVEN = [
    "सीता", "गीता", "कमला", "शांति", "सरोज", "सुनीता", "अनीता", "बिमला", "कृष्णा", "राज",
    "गुरप्रीत", "हरप्रीत", "मनजीत", "जसविंदर", "परमजीत", "सुखविंदर", "संतोष", "उर्मिला", "पुष्पा", "माया",
    "लक्ष्मी", "सावित्री", "भगवती", "निर्मला", "सुमन", "रेखा", "मीना", "ऊषा", "प्रेमा", "विद्या",
]
MALE_SECOND = ["सिंह", "लाल", "कुमार", "चंद", "राम", "प्रकाश", "दास", "नाथ", "दत्त", "पाल"]
FEMALE_SECOND = ["देवी", "कौर", "रानी", "बाई", "कुमारी"]

# Relation word written on the page -> normalised relation
PAGE_RELATIONS = {"s/o": ["पुत्र", "पुत्र", "पुत्र", "पु.", "वल्द"],
                  "d/o": ["पुत्री", "पुत्री", "दुख्तर"],
                  "w/o": ["पत्नी", "पत्नी", "धर्मपत्नी", "बेवा"]}

# Land type code -> how it is written on the page
LAND_TYPE_WORDS = {
    "chahi": "चाही", "nehri": "नहरी", "barani": "बारानी", "abi": "आबी",
    "banjar_kadim": "बंजर कदीम", "banjar_jadid": "बंजर जदीद", "gair_mumkin": "गैर मुमकिन",
}
LAND_TYPE_WEIGHTS = {
    "chahi": 0.30, "nehri": 0.25, "barani": 0.20, "abi": 0.05,
    "banjar_kadim": 0.08, "banjar_jadid": 0.05, "gair_mumkin": 0.07,
}
IRRIGATION_OF = {"chahi": "चाही", "nehri": "नहरी", "abi": "आबी"}  # others are rain-fed: बारानी

PATTI_NAMES = ["पत्ती कलां", "पत्ती खुर्द", "पत्ती अव्वल", "पत्ती दोयम", "पत्ती सोयम", "पत्ती 1", "पत्ती 2"]

VILLAGE_PARTS_A = ["राम", "किशन", "सुल्तान", "नंगल", "खेड़ी", "माजरा", "बहादुर", "गढ़ी", "धर्म", "शेर"]
VILLAGE_PARTS_B = ["पुर", "गढ़", "नगर", "खेड़ा", "वाला", "माजरा", "पुरा", " कलां", " खुर्द"]

SHARE_WORD = "हिस्सा"
KHUDKASHT = "खुदकाश्त"
TENANT_TAG = "(मुजारा)"
MUTATION_PREFIX = "इंतकाल नं."
DATE_PREFIX = "दि."
PENDING_WORD = "जेरे गौर"
TOTAL_WORD = "कुल"

# Printed column headers of the 12-column Jamabandi form (report Figure 1)
COLUMN_HEADERS = [
    "नंबर खेवट", "नंबर खतौनी", "नाम पत्ती / तरफ", "नाम मालिक मय हिस्सा", "नाम काश्तकार",
    "चाही / नहरी", "नंबर खसरा", "रकबा व किस्म जमीन", "मालगुजारी", "अबवाब", "लगान", "कैफियत / इंतकाल",
]
# Relative column widths; they sum to 1. The printed form is the same on every page.
COLUMN_WIDTHS = [0.05, 0.055, 0.065, 0.23, 0.12, 0.06, 0.075, 0.11, 0.05, 0.045, 0.045, 0.095]

# Benign spelling variants: (from, to) pairs applied to a name in the digital register
SPELLING_SWAPS = [
    ("सिंह", "सिंघ"), ("सिंघ", "सिंह"), ("कौर", "कोर"), ("ी", "ि"), ("ि", "ी"), ("ू", "ु"), ("ु", "ू"),
    ("ंद", "न्द"), ("न्द", "ंद"), ("ंत", "न्त"), ("श", "स"), ("बीर", "वीर"), ("कृ", "क्रि"),
]
