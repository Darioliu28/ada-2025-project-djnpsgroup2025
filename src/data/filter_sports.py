import pandas as pd
from rapidfuzz import fuzz
import re
from tqdm import tqdm

def find_match(subreddit, country_map):
    name = subreddit.lower().replace("r/", "")
    tokens = re.split(r'[^a-z]+', name)
    joined = ''.join(tokens)

    for token in tokens + [joined]:
        if token in country_map:
            return country_map[token]
        
    for key in country_map:
        cleaned_key = key.replace(' ', '')
        if len(cleaned_key) < 4:  
            continue
        if name.startswith(cleaned_key) or name.endswith(cleaned_key):
            return country_map[key]
        
    if len(name) > 6:
        best_match, best_score = None, 0
        for key, country in country_map.items():
            if len(key) < 4: 
                continue
            score = fuzz.partial_ratio(key, name)
            if score > 90 and score > best_score:
                best_match, best_score = country, score
        return best_match
        
    return None

def filter_sports(df_post_with_1_country, country_matches_list, final_folder):

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

    country_subs_set = set(matches.keys()) 

    sport_matches = {}
    all_subreddits_series = pd.concat([df_post_with_1_country['SOURCE_SUBREDDIT'], df_post_with_1_country['TARGET_SUBREDDIT']])
    unique_subreddit_list = all_subreddits_series.unique().tolist()

    for s in tqdm(unique_subreddit_list):
        if s in country_subs_set:
            continue
            
        sport_category = find_match(s, sport_map)
        
        if sport_category:
            sport_matches[s] = sport_category 

    df_sport_matches = pd.DataFrame(list(sport_matches.items()), columns=['subreddit', 'sport'])
    df_sport_matches.to_csv(final_folder + "df_country_sport_map.csv", index=False)
    df_sport_matches.to_csv(final_folder+"df_country_sport_map.csv", index=False)

    return sport_matches
