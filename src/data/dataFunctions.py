import pandas as pd

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

    return df_combined

def filter_posts_by_country(df_combined, df_country):
    """
    Filters the main posts DataFrame based on the approved country list,
    adds source/target country columns, and expands the 'properties' column.
    
    Returns:
        A tuple containing:
        - df_post_with_1_country: Posts with at least one approved subreddit.
        - df_post_between_countries: Posts between two approved subreddits.
    """
    print("Filtering posts by country and enriching data...")
    
    # 1. Create a set for fast filtering and a dictionary for mapping
    # Assuming df_country has 'subreddit' and 'country' columns
    country_sub_set = set(df_country['subreddit'])
    sub_to_country_map = dict(zip(df_country['subreddit'], df_country['country']))
    
    # 2. Filter DataFrames based on subreddit presence
    # Condition: At least one subreddit (Source OR Target) is in the approved list
    df_post_with_1_country = df_combined[
        df_combined["SOURCE_SUBREDDIT"].isin(country_sub_set) |
        df_combined["TARGET_SUBREDDIT"].isin(country_sub_set)
    ].reset_index(drop=True)
    
    # Condition: Both subreddits (Source AND Target) are in the approved list
    df_post_between_countries = df_combined[
        df_combined["SOURCE_SUBREDDIT"].isin(country_sub_set) &
        df_combined["TARGET_SUBREDDIT"].isin(country_sub_set)
    ].reset_index(drop=True)
    
    # Helper function to process the resulting DataFrames (prevents code duplication)
    def process_dataframe(df):
        if df.empty:
            return df
            
        # Add source_country and target_country columns
        # Map subreddits to countries; returns NaN if not found in the map
        df['source_country'] = df['SOURCE_SUBREDDIT'].map(sub_to_country_map)
        df['target_country'] = df['TARGET_SUBREDDIT'].map(sub_to_country_map)
        
        # Split the 'properties' column if it exists
        if 'properties' in df.columns:
            # Expand dictionaries inside 'properties' into separate columns
            properties_expanded = df['properties'].apply(pd.Series)
            
            # Remove the original 'properties' column and concatenate the new expanded columns
            df = pd.concat([df.drop(['properties'], axis=1), properties_expanded], axis=1)
            
        return df

    # 3. Apply transformations (mapping and splitting) to the filtered DataFrames
    print("Splitting properties and mapping countries...")
    df_post_with_1_country = process_dataframe(df_post_with_1_country)
    df_post_between_countries = process_dataframe(df_post_between_countries)
    
    print("Post processing complete.")
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

