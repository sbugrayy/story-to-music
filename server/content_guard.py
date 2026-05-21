"""
İçerik moderasyon — girdi ve çıktı için.

Politika (CLAUDE.md'den):
- Nefret söylemi, şiddet çağrısı, cinsel istismar → ret
- Yasadışı faaliyet talimatı, terör yüceltmesi → ret
- Sadece küfür odaklı, yaratıcı içerik içermeyen talepler → ret
- Şarkı sözü bağlamında hafif küfür → izinli

Implementasyon: keyword + regex tabanlı hızlı tarama. Borderline durumlar için
ileride Groq classifier eklenebilir, ama v1'de saf lokal.
"""

from __future__ import annotations

import re

# ── Hard Block: kategoriler ───────────────────────────────────────────────────

# Türkçe nefret söylemi / şiddet çağrısı pattern'leri
_HATE_VIOLENCE = [
    r"\b(öldür|gebert|katlet|infaz|asacaks|gömeceğ)\w*\s+(her|tüm|bütün)",
    r"\b(soykırım|katliam|linç)\b",
    r"\b(yokedil|temizlen|kovul)\w*\s+(bütün|tüm|her)",
]

# Cinsel istismar / çocuk istismarı
_SEXUAL_ABUSE = [
    r"\b(çocuk|bebek|küçük)\s*\w*\s*(porno|cinsel|seks|tecavüz)",
    r"\b(taciz|tecavüz)\s*\w*\s+(çocuk|bebek|küçük)",
]

# Yasadışı faaliyet talimatı (nasıl yapılır?)
_ILLEGAL_INSTRUCTION = [
    r"\bnasıl\s+(bomba|patlayıcı|silah)\s+(yap|üret)",
    r"\b(uyuşturucu|eroin|kokain)\s+nasıl\s+(yap|üret)",
    r"\bhow\s+to\s+(make|build)\s+(bomb|weapon|drug)",
]

# Terör yüceltmesi
_TERROR_PRAISE = [
    r"\b(yaşasın|hurra|şanı)\s+(daeş|işid|pkk|el[\s-]?kaide)\b",
]

HARD_BLOCK_PATTERNS = _HATE_VIOLENCE + _SEXUAL_ABUSE + _ILLEGAL_INSTRUCTION + _TERROR_PRAISE

# Prompt injection (server-internal, ayrı modülde de var ama input guard ilk seviyede yakalasın)
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"önceki\s+talimat",
    r"sen\s+artık\s+bir",
]


# ── API ───────────────────────────────────────────────────────────────────────

def check_input(text: str) -> tuple[bool, str]:
    """
    Girdi metnini güvenlik açısından kontrol et.

    Returns:
        (is_allowed, reason). is_allowed=True ise reason=""
    """
    if not text or not text.strip():
        return False, "Boş metin gönderilemez."

    low = text.lower()

    # Hard block patterns
    for pattern in HARD_BLOCK_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return False, "İçerik politikası ihlali: nefret söylemi, şiddet, cinsel istismar veya yasadışı talimat içeriği üretilemez."

    # Prompt injection
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return False, "Geçersiz girdi: sistem talimatı bypass denemesi tespit edildi."

    # Sadece küfür odaklı içerik (kelime başına >%40 küfür)
    if _is_profanity_dominant(low):
        return False, "İçerik tamamen küfür/argo odaklı, yaratıcı içerik üretilemez."

    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """
    Model çıktısını güvenlik açısından kontrol et.

    Eğitim verisi temiz olduğu için risk düşük, ama yine de tarıyoruz.
    """
    if not text or not text.strip():
        return False, "Boş çıktı."

    low = text.lower()
    for pattern in HARD_BLOCK_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return False, "Çıktı içerik politikasını ihlal ediyor."

    return True, ""


# ── Helpers ───────────────────────────────────────────────────────────────────

# Şarkı sözünde tolere edilebilen yaygın küfür/argo (Türkçe)
_LIGHT_PROFANITY = {
    "lan", "ulan", "ya", "be", "kahretsin", "boku", "boş", "sik", "amk",
}

# Hard küfür — bir tekiyle bile genelde sorun yaratmaz, ama dominantsa kötü
_HARD_PROFANITY = {
    "amına", "amına koyayım", "siktir", "orospu", "piç", "pezevenk", "şerefsiz",
}


def _is_profanity_dominant(text: str) -> bool:
    """Metnin >%40'ı küfür ise yaratıcı içerik yoktur, ret."""
    words = re.findall(r"[a-zçğıöşü]+", text.lower())
    if len(words) < 5:
        return False
    profane = sum(1 for w in words if w in _LIGHT_PROFANITY or w in _HARD_PROFANITY)
    return profane / len(words) > 0.40
