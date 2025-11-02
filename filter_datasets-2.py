import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rapidfuzz import fuzz
import re
from tqdm import tqdm

# --- 3. Matching function ---
def find_match(subreddit, country_map):
    name = subreddit.lower().replace("r/", "")
    # Split on underscores and digits
    tokens = re.split(r'[^a-z]+', name)
    joined = ''.join(tokens)

    # Direct token match
    for token in tokens + [joined]:
        if token in country_map:
            return country_map[token]
        
    # Starts/ends with full country names (e.g. norwaynews)
    for key in country_map:
        cleaned_key = key.replace(' ', '')
        if len(cleaned_key) < 4:  # ignore short codes
            continue
        if name.startswith(cleaned_key) or name.endswith(cleaned_key):
            return country_map[key]
        
    # Fuzzy match (high threshold, long names only)
    if len(name) > 6:
        best_match, best_score = None, 0
        for key, country in country_map.items():
            if len(key) < 4:  # skip short country codes
                continue
            score = fuzz.partial_ratio(key, name)
            if score > 90 and score > best_score:
                best_match, best_score = country, score
        return best_match
        
    return None

def filter_countries(origin_folder, final_folder):
    df_title = pd.read_csv(origin_folder+"soc-redditHyperlinks-title.tsv", sep="\t")
    df_body = pd.read_csv(origin_folder+"soc-redditHyperlinks-body.tsv", sep="\t")

    #Modify the data format
    time_format = '%Y-%m-%d %H:%M:%S'
    df_title['TIMESTAMP'] = pd.to_datetime(df_title['TIMESTAMP'], format=time_format)
    df_body['TIMESTAMP'] = pd.to_datetime(df_body['TIMESTAMP'], format=time_format)
    post_props_cols = [
    "num_chars", "num_chars_no_space", "frac_alpha", "frac_digits",
    "frac_upper", "frac_spaces", "frac_special", "num_words",
    "num_unique_words", "num_long_words", "avg_word_length",
    "num_unique_stopwords", "frac_stopwords", "num_sentences",
    "num_long_sentences", "avg_chars_per_sentence", "avg_words_per_sentence",
    "readability_index", "sent_pos", "sent_neg", "sent_compound",
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We",
    "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article",
    "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future",
    "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
    "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo",
    "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech",
    "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat",
    "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept",
    "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body",
    "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ",
    "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev",
    "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death",
    "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
    ]

    # Split POST_PROPERTIES into columns
    df_body[post_props_cols] = df_body["PROPERTIES"].str.split(",", expand=True).astype(float)
    df_title[post_props_cols] = df_title["PROPERTIES"].str.split(",", expand=True).astype(float)

    # Drop the old string column
    df_body = df_body.drop(columns=["PROPERTIES"])
    df_title = df_title.drop(columns=["PROPERTIES"])
    
    df_combined = pd.concat([df_body, df_title], ignore_index=True)

    # Combine both source and target columns into a single Pandas Series
    all_subreddits_series = pd.concat([df_combined['SOURCE_SUBREDDIT'], df_combined['TARGET_SUBREDDIT']])

    # Get the unique values from this combined series and convert to a list
    unique_subreddit_list = all_subreddits_series.unique().tolist()

    # --- Build base map ---
    country_map = {}

    # A dictionary mapping native country names to standardized English names
    native_names_map = {
        # ----- EUROPE -----
        'germany': 'Germany',
        'deutschland': 'Germany',
        'german': 'Germany',
        'alemania': 'Germany',
        'allemagne': 'Germany',
        'france': 'France',
        'français': 'France',
        'french': 'France',
        'republique francaise': 'France',
        'italy': 'Italy',
        'italia': 'Italy',
        'italian': 'Italy',
        'italiano': 'Italy',
        'repubblica italiana': 'Italy',
        'spain': 'Spain',
        'españa': 'Spain',
        'espana': 'Spain',
        'spanish': 'Spain',
        'espanol': 'Spain',
        'united kingdom': 'United Kingdom',
        'great britain': 'United Kingdom',
        'britain': 'United Kingdom',
        'british': 'United Kingdom',
        'england': 'United Kingdom',
        'netherlands': 'Netherlands',
        'nederland': 'Netherlands',
        'holland': 'Netherlands',
        'dutch': 'Netherlands',
        'oland': 'Netherlands',
        'portugal': 'Portugal',
        'portuguese': 'Portugal',
        'switzerland': 'Switzerland',
        'schweiz': 'Switzerland',
        'suisse': 'Switzerland',
        'svizzera': 'Switzerland',
        'swiss': 'Switzerland',
        'austria': 'Austria',
        'österreich': 'Austria',
        'austrian': 'Austria',
        'belgium': 'Belgium',
        'belgië': 'Belgium',
        'belgique': 'Belgium',
        'belgian': 'Belgium',
        'greece': 'Greece',
        'ellada': 'Greece',
        'Ελλάδα': 'Greece',
        'greek': 'Greece',
        'hellas': 'Greece',
        'poland': 'Poland',
        'polska': 'Poland',
        'polish': 'Poland',
        'sweden': 'Sweden',
        'sverige': 'Sweden',
        'swedish': 'Sweden',
        'svenska': 'Sweden',
        'norway': 'Norway',
        'norge': 'Norway',
        'norsk': 'Norway',
        'norwegian': 'Norway',
        'denmark': 'Denmark',
        'danmark': 'Denmark',
        'danish': 'Denmark',
        'dansk': 'Denmark',
        'finland': 'Finland',
        'suomi': 'Finland',
        'finnish': 'Finland',
        'iceland': 'Iceland',
        'íslensku': 'Iceland',
        'russia': 'Russia',
        'rossiya': 'Russia',
        'Россия': 'Russia',
        'russian': 'Russia',
        'ukraine': 'Ukraine',
        'ukrayina': 'Ukraine',
        'Україна': 'Ukraine',
        'ukrainian': 'Ukraine',
        'hungary': 'Hungary',
        'magyarország': 'Hungary',
        'romania': 'Romania',
        'românia': 'Romania',
        'czechia': 'Czechia',
        'czech': 'Czechia',
        'česko': 'Czechia',
        'czech republic': 'Czechia',
        'croatia': 'Croatia',
        'hrvatska': 'Croatia',
        'serbia': 'Serbia',
        'srbija': 'Serbia',
        'lietuva': 'Lithuania',
        'lithuania': 'Lithuania',
        'latvija': 'Latvia',
        'latvia': 'Latvia',
        'eesti': 'Estonia',
        'estonia': 'Estonia',
        'ireland': 'Ireland',
        'eire': 'Ireland',
        'irish': 'Ireland',

        # ----- EUROPA (3 Città Principali) -----
        'berlin': 'Germany',
        'hamburg': 'Germany',
        'munich': 'Germany',
        'paris': 'France',
        'marseille': 'France',
        'lyon': 'France',
        'rome': 'Italy',
        'milan': 'Italy',
        'naples': 'Italy',
        'madrid': 'Spain',
        'barcelona': 'Spain',
        'valencia': 'Spain',
        'london': 'United Kingdom',
        'birmingham': 'United Kingdom',
        'leeds': 'United Kingdom',
        'amsterdam': 'Netherlands',
        'rotterdam': 'Netherlands',
        'the hague': 'Netherlands',
        'lisbon': 'Portugal',
        'porto': 'Portugal',
        'sintra': 'Portugal',
        'zurich': 'Switzerland',
        'geneva': 'Switzerland',
        'basel': 'Switzerland',
        'vienna': 'Austria',
        'graz': 'Austria',
        'linz': 'Austria',
        'brussels': 'Belgium',
        'antwerp': 'Belgium',
        'ghent': 'Belgium',
        'athens': 'Greece',
        'thessaloniki': 'Greece',
        'patras': 'Greece',
        'warsaw': 'Poland',
        'kraków': 'Poland',
        'wrocław': 'Poland',
        'stockholm': 'Sweden',
        'gothenburg': 'Sweden',
        'malmö': 'Sweden',
        'oslo': 'Norway',
        'bergen': 'Norway',
        'stavanger': 'Norway',
        'copenhagen': 'Denmark',
        'aarhus': 'Denmark',
        'odense': 'Denmark',
        'helsinki': 'Finland',
        'tampere': 'Finland',
        'turku': 'Finland',
        'reykjavík': 'Iceland',
        'kopavogur': 'Iceland',
        'hafnarfjörður': 'Iceland',
        'moscow': 'Russia',
        'saint petersburg': 'Russia',
        'novosibirsk': 'Russia',
        'kyiv': 'Ukraine',
        'kharkiv': 'Ukraine',
        'odesa': 'Ukraine',
        'budapest': 'Hungary',
        'debrecen': 'Hungary',
        'szeged': 'Hungary',
        'bucharest': 'Romania',
        'cluj-napoca': 'Romania',
        'iași': 'Romania',
        'prague': 'Czechia',
        'brno': 'Czechia',
        'ostrava': 'Czechia',
        'zagreb': 'Croatia',
        'split': 'Croatia',
        'rijeka': 'Croatia',
        'belgrade': 'Serbia',
        'novi sad': 'Serbia',
        'niš': 'Serbia',
        'vilnius': 'Lithuania',
        'kaunas': 'Lithuania',
        'klaipėda': 'Lithuania',
        'riga': 'Latvia',
        'daugavpils': 'Latvia',
        'liepāja': 'Latvia',
        'tallinn': 'Estonia',
        'tartu': 'Estonia',
        'narva': 'Estonia',
        'dublin': 'Ireland',
        'cork': 'Ireland',
        'limerick': 'Ireland',

        # ----- AMERICHE (Alias Paesi) -----
        'united states': 'USA',
        'usa': 'USA',
        'america': 'USA',
        'american': 'USA',
        'united states of america': 'USA',
        'canada': 'Canada',
        'canadian': 'Canada',
        'mexico': 'Mexico',
        'méxico': 'Mexico',
        'mejico': 'Mexico',
        'mexican': 'Mexico',
        'brazil': 'Brazil',
        'brasil': 'Brazil',
        'brazilian': 'Brazil',
        'argentina': 'Argentina',
        'argentine': 'Argentina',
        'colombia': 'Colombia',
        'peru': 'Peru',
        'perú': 'Peru',
        'chile': 'Chile',
        'venezuela': 'Venezuela',

        # ----- AMERICHE (3 Città Principali) -----
        'toronto': 'Canada',
        'montreal': 'Canada',
        'calgary': 'Canada',
        'mexico city': 'Mexico',
        'tijuana': 'Mexico',
        'león': 'Mexico',
        'são paulo': 'Brazil',
        'rio de janeiro': 'Brazil',
        'brasília': 'Brazil',
        'buenos aires': 'Argentina',
        'córdoba': 'Argentina',
        'rosario': 'Argentina',
        'bogotá': 'Colombia',
        'medellín': 'Colombia',
        'cali': 'Colombia',
        'lima': 'Peru',
        'arequipa': 'Peru',
        'trujillo': 'Peru',
        'santiago': 'Chile',
        'puente alto': 'Chile',
        'antofagasta': 'Chile',
        'caracas': 'Venezuela',
        'maracaibo': 'Venezuela',
        'valencia': 'Venezuela',

        # ----- STATI UNITI D'AMERICA (Stati e Capitali) -----
        'new york city': 'USA', # Città più grande
        'los angeles': 'USA',    # 2a Città più grande
        'chicago': 'USA',        # 3a Città più grande
        # Stati
        'alabama': 'USA',
        'alaska': 'USA',
        'arizona': 'USA',
        'arkansas': 'USA',
        'california': 'USA',
        'colorado': 'USA',
        'connecticut': 'USA',
        'delaware': 'USA',
        'florida': 'USA',
        'georgia': 'USA',
        'hawaii': 'USA',
        'idaho': 'USA',
        'illinois': 'USA',
        'indiana': 'USA',
        'iowa': 'USA',
        'kansas': 'USA',
        'kentucky': 'USA',
        'louisiana': 'USA',
        'maine': 'USA',
        'maryland': 'USA',
        'massachusetts': 'USA',
        'michigan': 'USA',
        'minnesota': 'USA',
        'mississippi': 'USA',
        'missouri': 'USA',
        'montana': 'USA',
        'nebraska': 'USA',
        'nevada': 'USA',
        'new hampshire': 'USA',
        'new jersey': 'USA',
        'new mexico': 'USA',
        'new york': 'USA',
        'north carolina': 'USA',
        'north dakota': 'USA',
        'ohio': 'USA',
        'oklahoma': 'USA',
        'oregon': 'USA',
        'pennsylvania': 'USA',
        'rhode island': 'USA',
        'south carolina': 'USA',
        'south dakota': 'USA',
        'tennessee': 'USA',
        'texas': 'USA',
        'utah': 'USA',
        'vermont': 'USA',
        'virginia': 'USA',
        'washington': 'USA',
        'west virginia': 'USA',
        'wisconsin': 'USA',
        'wyoming': 'USA',
        # Capitali di Stato
        'montgomery': 'USA',
        'juneau': 'USA',
        'phoenix': 'USA',
        'little rock': 'USA',
        'sacramento': 'USA',
        'denver': 'USA',
        'hartford': 'USA',
        'dover': 'USA',
        'tallahassee': 'USA',
        'atlanta': 'USA',
        'honolulu': 'USA',
        'boise': 'USA',
        'springfield': 'USA',
        'indianapolis': 'USA',
        'des moines': 'USA',
        'topeka': 'USA',
        'frankfort': 'USA',
        'baton rouge': 'USA',
        'augusta': 'USA',
        'annapolis': 'USA',
        'boston': 'USA',
        'lansing': 'USA',
        'saint paul': 'USA',
        'jackson': 'USA',
        'jefferson city': 'USA',
        'helena': 'USA',
        'lincoln': 'USA',
        'carson city': 'USA',
        'concord': 'USA',
        'trenton': 'USA',
        'santa fe': 'USA',
        'albany': 'USA',
        'raleigh': 'USA',
        'bismarck': 'USA',
        'columbus': 'USA',
        'oklahoma city': 'USA',
        'salem': 'USA',
        'harrisburg': 'USA',
        'providence': 'USA',
        'columbia': 'USA',
        'pierre': 'USA',
        'nashville': 'USA',
        'austin': 'USA',
        'salt lake city': 'USA',
        'montpelier': 'USA',
        'richmond': 'USA',
        'olympia': 'USA',
        'charleston': 'USA',
        'madison': 'USA',
        'cheyenne': 'USA',

        # ----- ASIA E OCEANIA (Alias Paesi) -----
        'china': 'China',
        'zhongguo': 'China',
        '中国': 'China',
        'chinese': 'China',
        'prc': 'China',
        'japan': 'Japan',
        'nippon': 'Japan',
        'nihon': 'Japan',
        '日本': 'Japan',
        'japanese': 'Japan',
        'india': 'India',
        'bharat': 'India',
        'भारत': 'India',
        'indian': 'India',
        'south korea': 'South Korea',
        'hanguk': 'South Korea',
        '한국': 'South Korea',
        'korea': 'South Korea',
        'republic of korea': 'South Korea',
        'north korea': 'North Korea',
        'dprk': 'North Korea',
        'australia': 'Australia',
        'australian': 'Australia',
        'new zealand': 'New Zealand',
        'kiwi': 'New Zealand',
        'indonesia': 'Indonesia',
        'vietnam': 'Vietnam',
        'việtnam': 'Vietnam',
        'thailand': 'Thailand',
        'thai': 'Thailand',
        'philippines': 'Philippines',
        'pinoy': 'Philippines',
        'pilipinas': 'Philippines',
        'malaysia': 'Malaysia',
        'singapore': 'Singapore',
        'pakistan': 'Pakistan',
        'bangladesh': 'Bangladesh',

        # ----- ASIA E OCEANIA (3 Città Principali) -----
        'shanghai': 'China',
        'beijing': 'China',
        'guangzhou': 'China',
        'tokyo': 'Japan',
        'yokohama': 'Japan',
        'osaka': 'Japan',
        'mumbai': 'India',
        'delhi': 'India',
        'kolkata': 'India',
        'seoul': 'South Korea',
        'busan': 'South Korea',
        'incheon': 'South Korea',
        'pyongyang': 'North Korea',
        'chongjin': 'North Korea',
        'hamhung': 'North Korea',
        'sydney': 'Australia',
        'melbourne': 'Australia',
        'brisbane': 'Australia',
        'auckland': 'New Zealand',
        'christchurch': 'New Zealand',
        'wellington': 'New Zealand',
        'jakarta': 'Indonesia',
        'surabaya': 'Indonesia',
        'bandung': 'Indonesia',
        'ho chi minh city': 'Vietnam',
        'hanoi': 'Vietnam',
        'hai phong': 'Vietnam',
        'bangkok': 'Thailand',
        'mueang samut prakan': 'Thailand',
        'nonthaburi': 'Thailand',
        'quezon city': 'Philippines',
        'manila': 'Philippines',
        'davao city': 'Philippines',
        'kuala lumpur': 'Malaysia',
        'petaling jaya': 'Malaysia',
        'klang': 'Malaysia',
        'bedok': 'Singapore', # Area di pianificazione più grande
        'sengkang': 'Singapore', # 2a Area di pianificazione
        'jurong west': 'Singapore', # 3a Area di pianificazione
        'karachi': 'Pakistan',
        'lahore': 'Pakistan',
        'faisalabad': 'Pakistan',
        'dhaka': 'Bangladesh',
        'chittagong': 'Bangladesh',
        'gazipur': 'Bangladesh',

        # ----- MEDIO ORIENTE E AFRICA (Alias Paesi) -----
        'turkey': 'Turkey',
        'türkiye': 'Turkey',
        'turkish': 'Turkey',
        'iran': 'Iran',
        'persia': 'Iran',
        'saudi arabia': 'Saudi Arabia',
        'ksa': 'Saudi Arabia',
        'al-arabiyyah as-saudiyyah': 'Saudi Arabia',
        'egypt': 'Egypt',
        'misr': 'Egypt',
        'egyptian': 'Egypt',
        'south africa': 'South Africa',
        'suid-afrika': 'South Africa',
        'nigeria': 'Nigeria',
        'kenya': 'Kenya',
        'morocco': 'Morocco',
        'maroc': 'Morocco',
        'israel': 'Israel',
        'palestine': 'Palestine',
        'state of palestine': 'Palestine',
        'filastin': 'Palestine',
        'uae': 'United Arab Emirates',
        'united arab emirates': 'United Arab Emirates',
        'emirates': 'United Arab Emirates',

        # ----- MEDIO ORIENTE E AFRICA (3 Città Principali) -----
        'istanbul': 'Turkey',
        'ankara': 'Turkey',
        'izmir': 'Turkey',
        'tehran': 'Iran',
        'mashhad': 'Iran',
        'isfahan': 'Iran',
        'riyadh': 'Saudi Arabia',
        'jeddah': 'Saudi Arabia',
        'mecca': 'Saudi Arabia',
        'cairo': 'Egypt',
        'alexandria': 'Egypt',
        'giza': 'Egypt',
        'johannesburg': 'South Africa',
        'cape town': 'South Africa',
        'durban': 'South Africa',
        'lagos': 'Nigeria',
        'kano': 'Nigeria',
        'ibadan': 'Nigeria',
        'nairobi': 'Kenya',
        'mombasa': 'Kenya',
        'kisumu': 'Kenya',
        'casablanca': 'Morocco',
        'rabat': 'Morocco',
        'fes': 'Morocco',
        'jerusalem': 'Israel',
        'tel aviv': 'Israel',
        'haifa': 'Israel',
        'dubai': 'United Arab Emirates',
        'abu dhabi': 'United Arab Emirates',
        'sharjah': 'United Arab Emirates',
    }
    country_map.update(native_names_map)
    
    matches = {}

    for s in unique_subreddit_list:
        country = find_match(s, country_map)
        if country:
            matches[s] = country

    df_combined["SOURCE_COUNTRY"] = df_combined["SOURCE_SUBREDDIT"].map(matches)
    df_combined["TARGET_COUNTRY"] = df_combined["TARGET_SUBREDDIT"].map(matches)

    df_country = df_combined.dropna(subset=['SOURCE_COUNTRY', 'TARGET_COUNTRY'], how='all')

    new_order = ["SOURCE_SUBREDDIT","SOURCE_COUNTRY", "TARGET_SUBREDDIT", "TARGET_COUNTRY", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", 
        "num_chars", "num_chars_no_space", "frac_alpha", "frac_digits",
        "frac_upper", "frac_spaces", "frac_special", "num_words",
        "num_unique_words", "num_long_words", "avg_word_length",
        "num_unique_stopwords", "frac_stopwords", "num_sentences",
        "num_long_sentences", "avg_chars_per_sentence", "avg_words_per_sentence",
        "readability_index", "sent_pos", "sent_neg", "sent_compound",
        "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We",
        "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article",
        "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future",
        "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
        "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo",
        "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech",
        "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat",
        "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept",
        "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body",
        "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ",
        "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev",
        "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death",
        "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"]

    df_country = df_country[new_order]
    # Salva il DataFrame in un file CSV
    df_country.to_csv(final_folder+"df_country.csv", index=False)
    df_matches_list = pd.DataFrame(matches.items(), columns=['Subreddit', 'Country'])
    df_matches_list.to_csv(final_folder+'country_matches_map.csv', index=False)

def filter_sports(countries_dataset, country_matches_list, unique_sub_list, final_folder):
    # --- 1. Define the Sport Map (Alias -> Standard Category) ---
    # This map uses the same logic as your 'country_map'
    sport_map = {
        # --- General Terms ---
        'sports': 'Sport', 'o_subredditslympics': 'Sport', 'worldcup': 'Sport', 'athletics': 'Sport',
        'trackandfield': 'Sport', 'wintergames': 'Sport', 'commonwealthgames': 'Sport', 
        'xgames': 'Sport', 'sport': 'Sport',

        # --- Soccer ---
        'soccer': 'Soccer', 'football': 'Soccer', 'futbol': 'Soccer', 'futebol': 'Soccer', 
        'calcio': 'Soccer', 'fussball': 'Soccer', 'premierleague': 'Soccer', 'laliga': 'Soccer', 
        'seriea': 'Soccer', 'bundesliga': 'Soccer', 'ligue1': 'Soccer', 'championsleague': 'Soccer',
        'mls': 'Soccer', 'copaamerica': 'Soccer', 'euros': 'Soccer', 'uefa': 'Soccer', 
        'fifa': 'Soccer', 'epl': 'Soccer',

        # --- American Sports ---
        'nfl': 'American Football', 'americanfootball': 'American Football', 'cfl': 'American Football',
        'basketball': 'Basketball', 'nba': 'Basketball', 'wnba': 'Basketball', 'euroleague': 'Basketball',
        'baseball': 'Baseball', 'mlb': 'Baseball',
        'hockey': 'Hockey', 'nhl': 'Hockey',
        'ncaaf': 'NCAA Football', 'ncaab': 'NCAA Basketball',

        # --- Racquet & Ball Sports ---
        'tennis': 'Tennis', 'badminton': 'Badminton', 'squash': 'Squash', 
        'tabletennis': 'Tennis', 'pingpong': 'Tennis',
        'volleyball': 'Volleyball', 'handball': 'Handball', 'waterpolo': 'Water Polo',

        # --- Motorsports ---
        'motorsport': 'Motorsport', 'racing': 'Motorsport',
        'formula1': 'Formula 1', 'f1': 'Formula 1',
        'nascar': 'Motorsport', 'motogp': 'Motorsport', 'indycar': 'Motorsport', 
        'wec': 'Motorsport', 'rally': 'Motorsport', 'supercars': 'Motorsport',

        # --- Combat Sports ---
        'ufc': 'Combat Sports', 'mma': 'Combat Sports', 'boxing': 'Combat Sports', 
        'wrestling': 'Combat Sports', 'bjj': 'Combat Sports', 'kickboxing': 'Combat Sports', 
        'judo': 'Combat Sports', 'karate': 'Combat Sports',

        # --- Strength & Fitness Sports ---
        'powerlifting': 'Fitness', 'bodybuilding': 'Fitness', 'weightlifting': 'Fitness', 
        'strongman': 'Fitness', 'crossfit': 'Fitness',

        # --- Aquatic Sports ---
        'swimming': 'Aquatics', 'surfing': 'Aquatics', 'sailing': 'Aquatics', 
        'diving': 'Aquatics', 'rowing': 'Aquatics', 'triathlon': 'Aquatics',

        # --- Winter & Outdoor Sports ---
        'skiing': 'Winter Sports', 'snowboarding': 'Winter Sports', 
        'cycling': 'Cycling', 'climbing': 'Climbing', 'running': 'Running', 
        'marathon': 'Running', 'skateboarding': 'Skateboarding', 'hiking': 'Hiking',

        # --- Other Global Sports ---
        'cricket': 'Cricket', 'rugby': 'Rugby', 'rugbyunion': 'Rugby', 
        'golf': 'Golf', 'pga': 'Golf',
        'darts': 'Darts', 'snooker': 'Snooker',

        # --- eSports ---
        'esports': 'eSports', 'gaming': 'eSports', # 'gaming' is often used for eSports
        'leagueoflegends': 'eSports', 'lol': 'eSports', 'csgo': 'eSports', 
        'dota2': 'eSports', 'valorant': 'eSports', 'competitiveoverwatch': 'eSports', 
        'starcraft': 'eSports', 'rocketleague': 'eSports'
    }
    df_matches_countries = pd.read_csv(country_matches_list)
    matches = df_matches_countries.set_index('Subreddit')['Country'].to_dict()
    # --- 2. Loop to Find Sport Matches ---

    # Get the list of subreddits ALREADY identified as countries, to exclude them
    country_subs_set = set(matches.keys()) 

    # Dictionary for the new sport matches
    sport_matches = {}

    unique_sub_df = pd.read_csv(unique_sub_list, header=None)
    unique_subreddit_list = unique_sub_df.iloc[:, 0].tolist()

    for s in tqdm(unique_subreddit_list):
        # SKIP if the subreddit has ALREADY been identified as a country
        if s in country_subs_set:
            continue
            
        # RE-USE your find_country function, but pass it the sport_map
        sport_category = find_match(s, sport_map)
        
        if sport_category:
            sport_matches[s] = sport_category # Ex: {'r/formula1racing': 'Formula 1'}

    df_country=pd.read_csv(countries_dataset)
    df_country["SPORT_SOURCE"] = df_country["SOURCE_SUBREDDIT"].map(sport_matches)
    df_country["SPORT_TARGET"] = df_country["TARGET_SUBREDDIT"].map(sport_matches)

    new_order_s = ["SOURCE_SUBREDDIT","SOURCE_COUNTRY", "SPORT_SOURCE", "TARGET_SUBREDDIT", "TARGET_COUNTRY", "SPORT_TARGET", "POST_ID", "TIMESTAMP", "LINK_SENTIMENT", 
        "num_chars", "num_chars_no_space", "frac_alpha", "frac_digits",
        "frac_upper", "frac_spaces", "frac_special", "num_words",
        "num_unique_words", "num_long_words", "avg_word_length",
        "num_unique_stopwords", "frac_stopwords", "num_sentences",
        "num_long_sentences", "avg_chars_per_sentence", "avg_words_per_sentence",
        "readability_index", "sent_pos", "sent_neg", "sent_compound",
        "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We",
        "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article",
        "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future",
        "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
        "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
        "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo",
        "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech",
        "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat",
        "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept",
        "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body",
        "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ",
        "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev",
        "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death",
        "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"]

    df_country = df_country[new_order_s]
    # 1. Define your two conditions

    # Condition 1: (SPORT_SOURCE is not NaN) AND (TARGET_COUNTRY is not NaN)
    cond_1 = (df_country['SPORT_SOURCE'].notna()) & (df_country['TARGET_COUNTRY'].notna())

    # Condition 2: (SPORT_TARGET is not NaN) AND (SOURCE_COUNTRY is not NaN)
    cond_2 = (df_country['SPORT_TARGET'].notna()) & (df_country['SOURCE_COUNTRY'].notna())

    # 2. Apply the filter
    # Keep rows that satisfy Condition 1 OR Condition 2
    df_country_sport = df_country[cond_1 | cond_2].copy()
    df_country_sport.to_csv(final_folder+"df_country_sport.csv", index=False)


origin_f = str("/Users/noemiortona/Documents/MTE/ADA/data_p/")
final_f = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/")
countries_d = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/df_country.csv")
country_m = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/country_matches_map.csv")
unique_sub_f = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/unique_subreddits.csv")

filter_countries(origin_f, final_f)
filter_sports(countries_d, country_m, unique_sub_f, final_f)