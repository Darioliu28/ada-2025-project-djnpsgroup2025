def contains_econ_word(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    clean = _normalize(text)
    # match veloce: parola intera o sottostringa (per r/subreddit composti)
    words = set(clean.split())
    for kw in ECON_KEYWORDS:
        if kw in words or kw in clean:
            return True
    return False




    
def classify_with_keyword(name: str):
    if not isinstance(name, str):
        return ("Other / unclassified", None, [])
    s = name.lower()
    all_hits = []
    chosen_topic = "Other / unclassified"
    chosen_kw = None
    for topic, kws in MACRO_TOPICS.items():
        for kw in kws:
            if kw in s:
                all_hits.append(kw)
                if chosen_kw is None:  # first match = main keyword
                    chosen_topic = topic
                    chosen_kw = kw
    return (chosen_topic, chosen_kw, all_hits)






    def classify_topic(name: str) -> str:
    if not isinstance(name, str):
        return "Other / unclassified"
    s = name.lower()
    for topic, kws in MACRO_TOPICS.items():
        for kw in kws:
            if kw in s:
                return topic
    return "Other / unclassified"