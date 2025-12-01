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

def filter_sports(df_post_with_1_country, country_matches_list, final_folder):
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
    matches = df_matches_countries.set_index('subreddit')['country'].to_dict()
    # --- 2. Loop to Find Sport Matches ---

    # Get the list of subreddits ALREADY identified as countries, to exclude them
    country_subs_set = set(matches.keys()) 

    # Dictionary for the new sport matches
    sport_matches = {}
    all_subreddits_series = pd.concat([df_post_with_1_country['SOURCE_SUBREDDIT'], df_post_with_1_country['TARGET_SUBREDDIT']])
    unique_subreddit_list = all_subreddits_series.unique().tolist()

    for s in tqdm(unique_subreddit_list):
        # SKIP if the subreddit has ALREADY been identified as a country
        if s in country_subs_set:
            continue
            
        # RE-USE your find_country function, but pass it the sport_map
        sport_category = find_match(s, sport_map)
        
        if sport_category:
            sport_matches[s] = sport_category # Ex: {'r/formula1racing': 'Formula 1'}

    df_sport_matches = pd.DataFrame(list(sport_matches.items()), columns=['subreddit', 'sport'])
    df_sport_matches.to_csv(final_folder + "df_country_sport_map.csv", index=False)
    df_sport_matches.to_csv(final_folder+"df_country_sport_map.csv", index=False)

    return sport_matches
