import pandas as pd
import time
import matplotlib.pyplot as plt

t0 = time.time()
df = pd.concat([df_body, df_title], ignore_index=True)
df = df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].dropna()


ECON_KEYWORDS = {

    "economy","economic","economics","macro","micro","econometrics",
    "growth","recession","inflation","deflation","stagflation","productivity",
    "gdp","policy","monetary","fiscal","budget","tax","taxation","debt",
    "deficit","surplus","inequality","poverty","development","sustainability",


    "finance","financial","bank","banking","credit","loan","interest",
    "investment","investing","investor","dividend","equity","capital",
    "bond","bonds","securities","market","markets","stock","stocks",
    "index","indices","fund","funds","etf","hedge","portfolio","trading",
    "broker","exchange","nyse","nasdaq","wallstreet","dowjones","s&p",
    "forex","currency","currencies","money","wealth","cryptocurrency",
    "crypto","bitcoin","ethereum","blockchain","token","nft",


    "commodity","commodities","oil","petrol","gas","naturalgas","energy",
    "coal","uranium","electricity","power","renewable","solar","wind",
    "hydrogen","gold","silver","platinum","palladium","copper","iron",
    "steel","aluminium","nickel","zinc","lead","lithium","cobalt",
    "rareearth","grain","wheat","corn","soybean","rice","coffee","sugar",
    "cotton","timber","water","mining","agriculture","agricultural",


    "industry","industries","manufacturing","commerce","business",
    "entrepreneur","entrepreneurship","startup","venture","innovation",
    "supply","demand","trade","export","import","globalization","logistics",
    "transport","shipping",


    "centralbank","ecb","fed","imf","worldbank","oecd","wto","regulation",
    "governance","public","reform","subsidy","aid","stimulus","bailout",
    "spending","taxpayer","treasury"
}

def _normalize(s: str) -> str:
    """lowercase + replaces non-alphanumeric characters with space"""
    return "".join(ch.lower() if ch.isalnum() else " " for ch in s)

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


all_subs = pd.unique(pd.concat([df["SOURCE_SUBREDDIT"], df["TARGET_SUBREDDIT"]], ignore_index=True))


econ_subs = sorted([s for s in all_subs if contains_econ_word(s)], key=lambda x: x.lower())


econ_list_df = pd.DataFrame({"SUBREDDIT": econ_subs})
econ_list_df.to_csv("economic_subreddit_list.csv", index=False)


print(f"Subreddit totali unici: {len(all_subs):,}")
print(f"Subreddit con termini economici: {len(econ_subs):,}")
print("First 30:")
print(econ_list_df.head(30).to_string(index=False))

print(f"\nTempo totale: {time.time() - t0:.2f}s")
print("Saved: economic_subreddit_list.csv")




econ_df = pd.read_csv("economic_subreddit_list.csv")
econ_df["SUBREDDIT"] = econ_df["SUBREDDIT"].str.lower().str.strip()


country_approved["subreddit"] = country_approved["subreddit"].str.lower().str.strip()
country_approved["predicted_country"] = country_approved["predicted_country"].str.strip()


econ_set = set(econ_df["SUBREDDIT"])


country_approved["IS_ECONOMIC"] = country_approved["subreddit"].isin(econ_set)


econ_matches = country_approved[country_approved["IS_ECONOMIC"]]


print(f"Total subreddits in dataset: {len(country_approved):,}")
print(f"Economic subreddits found: {len(econ_matches):,}\n")

print("Examples of economic subreddits found:")
print(econ_matches.head(20))


econ_matches.to_csv("economic_subreddits_in_country_approved.csv", index=False)
print("\nFile salvato: economic_subreddits_in_country_approved.csv")




country_approved = pd.read_csv(data_folder + "subreddit_matches_approved.csv", sep=",")
econ_df = pd.read_csv("economic_subreddits.csv")


print("Main dataset columns:", country_approved.columns.tolist())
print("Economic dataset columns:", econ_df.columns.tolist())


country_approved["subreddit"] = (
    country_approved["subreddit"].astype(str).str.lower().str.strip()
)

econ_df["SOURCE_SUBREDDIT"] = econ_df["SOURCE_SUBREDDIT"].astype(str).str.lower().str.strip()
econ_df["TARGET_SUBREDDIT"] = econ_df["TARGET_SUBREDDIT"].astype(str).str.lower().str.strip()


econ_subs = set(econ_df["SOURCE_SUBREDDIT"]).union(set(econ_df["TARGET_SUBREDDIT"]))


all_subs = set(country_approved["subreddit"])


econ_in_country = sorted(econ_subs.intersection(all_subs))

print(f"\nTotal economic subreddits (SOURCE∪TARGET): {len(econ_subs):,}")
print(f"Total subreddits in main dataset: {len(all_subs):,}")
print(f"Economic subreddits found in main dataset: {len(econ_in_country):,}\n")

print("Examples of matches found:")
print(econ_in_country[:30])


pd.DataFrame({"ECON_IN_COUNTRY": econ_in_country}).to_csv("econ_in_country_matches.csv", index=False)
print("\nFile salvato: econ_in_country_matches.csv ")





print("Main dataset columns:", country_approved.columns.tolist())
print("Economic dataset columns:", econ_df.columns.tolist())


country_approved["subreddit"] = (
    country_approved["subreddit"].astype(str).str.lower().str.strip()
)

econ_df["SOURCE_SUBREDDIT"] = econ_df["SOURCE_SUBREDDIT"].astype(str).str.lower().str.strip()
econ_df["TARGET_SUBREDDIT"] = econ_df["TARGET_SUBREDDIT"].astype(str).str.lower().str.strip()


geo_subs = set(country_approved["subreddit"])


econ_df["IS_SOURCE_GEO"] = econ_df["SOURCE_SUBREDDIT"].isin(geo_subs)
econ_df["IS_TARGET_GEO"] = econ_df["TARGET_SUBREDDIT"].isin(geo_subs)


econ_geo_links = econ_df[(econ_df["IS_SOURCE_GEO"]) | (econ_df["IS_TARGET_GEO"])]


print(f"Total economic edges: {len(econ_df):,}")
print(f"Edges with at least one geographic subreddit: {len(econ_geo_links):,}\n")

print("Examples of edges found:")
print(econ_geo_links.head(20)[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]])


econ_geo_links.to_csv("economic_links_with_geo.csv", index=False)
print("\nFile salvato: economic_links_with_geo.csv ")






MACRO_TOPICS = {
    "Economics": [
        "economy","economic","economics","macro","micro","econometrics",
        "growth","recession","inflation","deflation","stagflation","productivity",
        "gdp","policy","monetary","fiscal","budget","tax","taxation","debt",
        "deficit","surplus","inequality","poverty","development","sustainability"
    ],
    
    "Finance and Markets": [
        "finance","financial","bank","banking","credit","loan","interest",
        "investment","investing","investor","dividend","equity","capital",
        "bond","bonds","securities","market","markets","stock","stocks",
        "index","indices","fund","funds","etf","hedge","portfolio","trading",
        "broker","exchange","nyse","nasdaq","wallstreet","dowjones","s&p",
        "forex","currency","currencies","money","wealth","cryptocurrency",
        "crypto","bitcoin","ethereum","blockchain","token","nft"
    ],
    
    "Commodities": [
        "commodity","commodities","gold","silver","platinum","palladium",
        "copper","iron","steel","aluminium","nickel","zinc","lead","lithium",
        "cobalt","rareearth","grain","wheat","corn","soybean","rice","coffee",
        "sugar","cotton","timber","water","mining","agriculture","agricultural"
    ],
    
    "Energy": [
        "oil","petrol","gas","naturalgas","energy","coal","uranium","electricity",
        "power","renewable","solar","wind","hydrogen"
    ],
    
    "Businesses": [
        "industry","industries","manufacturing","commerce","business",
        "entrepreneur","entrepreneurship","startup","venture","innovation",
        "supply","demand","trade","export","import","globalization","logistics",
        "transport","shipping"
    ],
    
    "Istitutions / Policies": [
        "centralbank","ecb","fed","imf","worldbank","oecd","wto","regulation",
        "governance","public","reform","subsidy","aid","stimulus","bailout",
        "spending","taxpayer","treasury"
    ]
}

out = subreddits_economici["SUBREDDIT"].apply(classify_with_keyword).apply(pd.Series)
out.columns = ["macro_tema", "keyword_corrispondente", "tutte_le_keyword_trovate"]


out["n_keyword_match"] = out["tutte_le_keyword_trovate"].apply(len)


subreddits_economici = pd.concat([subreddits_economici, out], axis=1)

print(subreddits_economici.head(15))





df_economici = pd.read_csv("economic_subreddit_list.csv")


df_economici["SUBREDDIT"] = df_economici["SUBREDDIT"].astype(str).str.lower().str.strip()

print(f"Totale subreddit: {len(df_economici):,}")
print(df_economici.head())


MACRO_TOPICS = {
    "Economics": [
        "economy","economic","economics","macro","micro","econometrics",
        "growth","recession","inflation","deflation","stagflation","productivity",
        "gdp","policy","monetary","fiscal","budget","tax","taxation","debt",
        "deficit","surplus","inequality","poverty","development","sustainability"
    ],
    
    "Finance and Markets": [
        "finance","financial","bank","banking","credit","loan","interest",
        "investment","investing","investor","dividend","equity","capital",
        "bond","bonds","securities","market","markets","stock","stocks",
        "index","indices","fund","funds","etf","hedge","portfolio","trading",
        "broker","exchange","nyse","nasdaq","wallstreet","dowjones","s&p",
        "forex","currency","currencies","money","wealth","cryptocurrency",
        "crypto","bitcoin","ethereum","blockchain","token","nft"
    ],
    
    "Commodities": [
        "commodity","commodities","gold","silver","platinum","palladium",
        "copper","iron","steel","aluminium","nickel","zinc","lead","lithium",
        "cobalt","rareearth","grain","wheat","corn","soybean","rice","coffee",
        "sugar","cotton","timber","water","mining","agriculture","agricultural"
    ],
    
    "Energy": [
        "oil","petrol","gas","naturalgas","energy","coal","uranium","electricity",
        "power","renewable","solar","wind","hydrogen"
    ],
    
    "Businesses": [
        "industry","industries","manufacturing","commerce","business",
        "entrepreneur","entrepreneurship","startup","venture","innovation",
        "supply","demand","trade","export","import","globalization","logistics",
        "transport","shipping"
    ],
    
    "Istitutions / Policies": [
        "centralbank","ecb","fed","imf","worldbank","oecd","wto","regulation",
        "governance","public","reform","subsidy","aid","stimulus","bailout",
        "spending","taxpayer","treasury"
    ]
}


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


out = df_economici["SUBREDDIT"].apply(classify_with_keyword).apply(pd.Series)
out.columns = ["macro_topic", "matched_keyword", "all_keywords_found"]

# Add also the number of matched keywords
out["n_keyword_match"] = out["all_keywords_found"].apply(len)

# Merge the results with the main dataset
df_economici = pd.concat([df_economici, out], axis=1)

print(df_economici.head(15))
df_economici["macro_topic"].value_counts()



FILE = "economic_links_with_geo.csv"  # change to .xls if needed

try:
    if FILE.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(FILE)
    else:
        df = pd.read_csv(FILE)
except Exception:
    # Fallback: try both
    try:
        df = pd.read_csv(FILE)
    except Exception:
        df = pd.read_excel(FILE)


req = {"SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"}
missing = req - set(df.columns)
if missing:
    raise KeyError(f"Missing columns {missing}. Found columns: {df.columns.tolist()}")

# Normalize subreddit names
df["SOURCE_SUBREDDIT"] = df["SOURCE_SUBREDDIT"].astype(str).str.lower().str.strip()
df["TARGET_SUBREDDIT"] = df["TARGET_SUBREDDIT"].astype(str).str.lower().str.strip()


def classify_topic(name: str) -> str:
    if not isinstance(name, str):
        return "Probably Energy"
    s = name.lower()
    for topic, kws in MACRO_TOPICS.items():
        for kw in kws:
            if kw in s:
                return topic
    return "Probably Energy"


df["SOURCE_CATEGORY"] = df["SOURCE_SUBREDDIT"].apply(classify_topic)
df["TARGET_CATEGORY"] = df["TARGET_SUBREDDIT"].apply(classify_topic)


print(df[["SOURCE_SUBREDDIT","SOURCE_CATEGORY","TARGET_SUBREDDIT","TARGET_CATEGORY"]].head(15))

out_name = "economic_links_with_geo_labeled.csv"
df.to_csv(out_name, index=False)
print(f"\n File saved: {out_name}")




