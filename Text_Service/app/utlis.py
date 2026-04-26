import re

def arabert_preprocess(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"[يى]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"\u0640", "", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
