# data_processing.py
import pandas as pd
from thefuzz import fuzz, process

def load_country_subreddits(country_file):
    """
    Loads the approved subreddits file and prepares it.
    
    Returns:
        pd.DataFrame: DataFrame with 'subreddit' and 'predicted_country' columns.
    """
    print(f"Loading approved subreddits from: {country_file}")
    
    # We use header=None and rename based on our previous bug fix (KeyError)
    try:
        df_country = pd.read_csv(country_file)
        # Check if 'subreddit' column exists, if not, retry
        if 'subreddit' not in df_country.columns:
            print("  No header found. Retrying with header=None.")
            df_country = pd.read_csv(country_file, header=None)
            df_country = df_country.rename(columns={0: "subreddit", 1: "predicted_country"})
            
    except Exception as e:
        print(f"  Error loading, trying with header=None. Error: {e}")
        df_country = pd.read_csv(country_file, header=None)
        df_country = df_country.rename(columns={0: "subreddit", 1: "predicted_country"})

    df_country['subreddit'] = df_country['subreddit'].str.lower()
    return df_country

def load_embeddings(embeddings_file):
    """
    Loads and prepares the embeddings file.
    
    Returns:
        pd.DataFrame: DataFrame with 'subreddit' and embedding columns.
    """
    print(f"Loading embeddings from: {embeddings_file}")
    df_embeddings = pd.read_csv(embeddings_file, header=None)
    df_embeddings.rename(columns={0: "subreddit"}, inplace=True)
    df_embeddings['subreddit'] = df_embeddings['subreddit'].str.lower()
    return df_embeddings

def load_post_data(title_file, body_file):
    """
    Loads and combines the title and body post files.
    
    Returns:
        pd.DataFrame: The combined 'df_combined' DataFrame with all posts.
    """
    print("Loading and combining post data...")
    df_title = pd.read_csv(title_file, sep="\\t", engine='python')
    df_body = pd.read_csv(body_file, sep="\\t", engine='python')
    
    df_combined = pd.concat([df_body, df_title], ignore_index=True)
    df_combined["SOURCE_SUBREDDIT"] = df_combined["SOURCE_SUBREDDIT"].str.lower()
    df_combined["TARGET_SUBREDDIT"] = df_combined["TARGET_SUBREDDIT"].str.lower()
    return df_combined

def filter_posts_by_country(df_combined, df_country):
    """
    Filters the main posts DataFrame based on the approved country list.
    
    Returns:
        A tuple containing:
        - df_post_with_1_country: Posts with at least one approved subreddit.
        - df_post_between_countries: Posts between two approved subreddits.
    """
    print("Filtering posts by country...")
    country_sub_set = set(df_country['subreddit'])
    
    df_post_with_1_country = df_combined[
        df_combined["SOURCE_SUBREDDIT"].isin(country_sub_set) |
        df_combined["TARGET_SUBREDDIT"].isin(country_sub_set)
    ].reset_index(drop=True)
    
    df_post_between_countries = df_combined[
        df_combined["SOURCE_SUBREDDIT"].isin(country_sub_set) &
        df_combined["TARGET_SUBREDDIT"].isin(country_sub_set)
    ].reset_index(drop=True)
    
    print("Post filtering complete.")
    return df_post_with_1_country, df_post_between_countries

def filter_embeddings_by_country(df_embeddings, df_countries):
    """
    
    Returns:
        - df_embeddings_filtered: The filtered embeddings DataFrame..
    """
    print("Filtering embeddings...")
    
    # 1. Filter embeddings by approved list
    countries_subs_set = set(df_countries['subreddit'])
    df_embeddings_filtered = df_embeddings[
        df_embeddings['subreddit'].isin(countries_subs_set)
    ].reset_index(drop=True)
    
    print("Embedding filtering complete.")
    return df_embeddings_filtered


## -- Functions used in filter_politic_subreddits.ipynb --

def _build_keyword_maps(ideologies_dict):   
    """
    Helper function to create keyword maps from the main ideology dictionary.
    Includes versions with and without underscores.
    
    Returns:
        tuple: (keyword_to_ideology_map, all_keywords_list)
    """
    keyword_to_ideology_map = {}
    all_keywords_to_search = set()

    for ideology_label, keyword_list in ideologies_dict.items():
        for keyword in keyword_list:
            # Add the keyword itself
            all_keywords_to_search.add(keyword)
            keyword_to_ideology_map[keyword] = ideology_label
            
            # Add a version without underscores (e.g., 'alt_right' -> 'altright')
            if '_' in keyword:
                kw_no_underscore = keyword.replace('_', '')
                all_keywords_to_search.add(kw_no_underscore)
                keyword_to_ideology_map[kw_no_underscore] = ideology_label

    all_keywords_list = list(all_keywords_to_search)
    print(f"Created a reverse map for {len(keyword_to_ideology_map)} keywords.")
    print(f"Searching for {len(all_keywords_list)} unique political/ideology keywords.")
    
    return keyword_to_ideology_map, all_keywords_list

def find_political_subreddits(df_posts, ideologies_dict, score_threshold=95, limit_per_keyword=50):
    """
    Discovers "strictly-politic" subreddits from the entire post dataset
    using fuzzy matching against a keyword list.
    
    
    Args:
        df_posts: DataFrame containing all posts (to get all subreddit names).
        ideologies_dict: The large POLITICAL_IDEOLOGIES dictionary.
        score_threshold (int): Fuzz match score required (default 95).
        limit_per_keyword (int): Max matches to check per keyword (default 50).
        
    Returns:
        DataFrame of discovered political subreddits, fields:
        ['Matched Subreddit', 'Ideology', 'Search Term']
    """
    
    # --- Step 1: Create Keyword List and Ideology Map ---
    keyword_to_ideology_map, all_keywords_list = _build_keyword_maps(ideologies_dict)
    
    # --- Step 2: Get Subreddit Lists ---
    # Get all unique subreddits from the posts data
    all_subs_list = list(pd.concat([
        df_posts['SOURCE_SUBREDDIT'], 
        df_posts['TARGET_SUBREDDIT']
    ]).unique())
    
    
    print(f"Searching through {len(all_subs_list)} total unique subreddits.")

    # --- Step 3: Find All Matching Subreddits ---
    political_results = []
    print(f"Starting search with score threshold {score_threshold}...")

    for keyword in all_keywords_list:
            matches = process.extract(keyword, all_subs_list, 
                                    scorer=fuzz.partial_ratio, 
                                    limit=limit_per_keyword)
            
            for sub, score in matches:
                if score >= score_threshold:

                    if  (len(sub) > 4):
                        political_results.append({
                            'Search Term': keyword,
                            'Matched Subreddit': sub,
                            'Ideology': keyword_to_ideology_map[keyword],
                            'Score': score
                        })

    print("Search complete.")

    # --- Step 4: Clean and Return Results ---
    if not political_results:
        print("No political subreddits found matching the criteria.")
        return pd.DataFrame(columns=['Matched Subreddit', 'Ideology', 'Search Term'])
        
    df_politics_temp = pd.DataFrame(political_results)

    # Sort by score (highest score wins) and remove duplicates
    df_politics = df_politics_temp.sort_values(by='Score', ascending=False)
    df_politics = df_politics.drop_duplicates(subset=['Matched Subreddit'])
    df_politics = df_politics.reset_index(drop=True)

    # Re-order columns
    df_politics = df_politics[['Matched Subreddit', 'Ideology', 'Search Term']]

    print(f"\nFound {len(df_politics)} relevant political ideology subreddits.")
    
    return df_politics

def load_country_exp(matches, df_combined):
    time_format = '%Y-%m-%d %H:%M:%S'
    df_combined['TIMESTAMP'] = pd.to_datetime(df_combined['TIMESTAMP'], format=time_format)
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
    df_combined[post_props_cols] = df_combined["PROPERTIES"].str.split(",", expand=True).astype(float)
    df_combined = df_combined.drop(columns=["PROPERTIES"])
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
    return df_country