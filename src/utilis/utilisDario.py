
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms import community
import itertools
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm 
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# === Columns FOR POST PROPERTIES ===
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

# === COUNTRY INTERACTIONS PROCESSING ===
def process_country_interactions(df_combined, mapping_csv_path, remove_self_loops=True):
    """
    Loads a country mapping file, merges it with interaction data,
    and aggregates country-to-country interactions.

    Args:
        df_combined (pd.DataFrame): The main DataFrame of interactions 
                                    (must have 'SOURCE_SUBREDDIT' and 'TARGET_SUBREDDIT').
        mapping_csv_path (str): Path to the CSV file containing 
                                'subreddit' and 'country'/'predicted_country' columns.
        remove_self_loops (bool): If True, removes interactions where 
                                  source_country == target_country.

    Returns:
        tuple: (country_interactions, merged_valid)
            - country_interactions (pd.DataFrame): Aggregated counts 
              (source_country, target_country, n_interactions).
            - merged_valid (pd.DataFrame): The full merged DataFrame with 
              'source_country' and 'target_country' columns, filtered to non-NaN rows.
    """
    
    # --- 1. Load and clean mapping file ---
    try:
        mapping_df = pd.read_csv(mapping_csv_path)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_csv_path}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on error

    mapping_df.columns = [c.strip().lower() for c in mapping_df.columns]
    
    # Standardize country column name
    if 'country' not in mapping_df.columns and 'predicted_country' in mapping_df.columns:
        mapping_df.rename(columns={'predicted_country': 'country'}, inplace=True)
        
    # Check for required columns
    if 'subreddit' not in mapping_df.columns or 'country' not in mapping_df.columns:
        print(f"Error: Mapping file {mapping_csv_path} must have 'subreddit' and 'country' columns.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 2. Merge country info onto source and target subreddits ---
    merged = (
        df_combined
        .merge(mapping_df[['subreddit', 'country']], how='left',
               left_on='SOURCE_SUBREDDIT', right_on='subreddit')
        .rename(columns={'country': 'source_country'})
        .drop(columns=['subreddit'])
        .merge(mapping_df[['subreddit', 'country']], how='left',
               left_on='TARGET_SUBREDDIT', right_on='subreddit')
        .rename(columns={'country': 'target_country'})
        .drop(columns=['subreddit'])
    )

    # --- 3. Keep only interactions where both countries are known ---
    merged_valid = merged.dropna(subset=['source_country', 'target_country'])

    # --- 4. Aggregate counts of inter-country interactions ---
    country_interactions = (
        merged_valid.groupby(['source_country', 'target_country'])
        .size()
        .reset_index(name='n_interactions')
        .sort_values(by='n_interactions', ascending=False)
    )

    if remove_self_loops:
        country_interactions = country_interactions.query("source_country != target_country")

    return country_interactions, merged_valid


def map_posts_to_countries(df_posts, mapping_csv_path):
    """
    Merges a post DataFrame with a country mapping file based on SOURCE_SUBREDDIT.

    Args:
        df_posts (pd.DataFrame): The main DataFrame of posts (e.g., df_combined). 
                                 Must have 'SOURCE_SUBREDDIT'.
        mapping_csv_path (str): Path to the CSV file containing 
                                'subreddit' and 'country'/'predicted_country'.

    Returns:
        pd.DataFrame: The original df_posts with a new 'country' column.
                      Rows without a country match will have NaN 
                      in the 'country' column.
    """
    
    # --- 1. Load and clean mapping file ---
    try:
        mapping_df = pd.read_csv(mapping_csv_path)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_csv_path}")
        # Return original df with an empty 'country' col as a precaution
        return df_posts.assign(country=pd.NA) 

    mapping_df.columns = [c.strip().lower() for c in mapping_df.columns]
    
    # Standardize country column name
    if 'country' not in mapping_df.columns and 'predicted_country' in mapping_df.columns:
        mapping_df.rename(columns={'predicted_country': 'country'}, inplace=True)
        
    if 'subreddit' not in mapping_df.columns or 'country' not in mapping_df.columns:
        print(f"Error: Mapping file {mapping_csv_path} must have 'subreddit' and 'country'.")
        return df_posts.assign(country=pd.NA)

    # --- 2. Merge country info onto source subreddit ---
    # We use a left merge to keep all original posts
    df_with_countries = df_posts.merge(
        mapping_df[['subreddit', 'country']],
        how='left',
        left_on='SOURCE_SUBREDDIT',
        right_on='subreddit'
    ).drop(columns=['subreddit']) # Drop the redundant 'subreddit' col
    
    return df_with_countries

# === 1. CLUSTERS WITH EMBEDDING ANALYSIS ===

def prepare_embeddings_for_clustering(df_emb):
    """
    Separates subreddit labels from embedding features and scales the features.
    
    Args:
        df_emb (pd.DataFrame): The raw embeddings dataframe. Assumes column 0 
                               is 'subreddit' and all others are features.
    
    Returns:
        tuple: (scaled_features, subreddit_labels)
    """
    print("Preparing embeddings for clustering...")
    
    # 1. Separate labels and features
    if 'subreddit' not in df_emb.columns:
        print("Renaming column 0 to 'subreddit'.")
        df_emb = df_emb.rename(columns={0: 'subreddit'})
        
    subreddit_labels = df_emb['subreddit'].values
    # Get all columns that are not 'subreddit'
    feature_cols = [col for col in df_emb.columns if col != 'subreddit']
    features = df_emb[feature_cols].values
    
    # 2. Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"Data prepared: {len(subreddit_labels)} items, {scaled_features.shape[1]} features.")
    return scaled_features, subreddit_labels

def calculate_kmeans_elbow_wide(scaled_data, k_values_list, n_samples=5000):
    """
    Calculates the K-Means "inertia" for a specific list of k values.
    This is for testing a wide, sparse range (e.g., 20, 50, 100, 200).
    
    Uses a random sample of the data for speed.
    
    Args:
        scaled_data (np.array): The scaled feature data.
        k_values_list (list): A list of integers to test (e.g., [20, 40, 60]).
        n_samples (int): Number of samples to use for this calculation.

    Returns:
        pd.DataFrame: A DataFrame with 'k' and 'inertia'.
    """
    print(f"Calculating K-Means elbow (wide range)... (using a sample of {n_samples} items)")
    
    # Subsample the data for speed
    if len(scaled_data) > n_samples:
        indices = np.random.choice(scaled_data.shape[0], n_samples, replace=False)
        data_sample = scaled_data[indices]
    else:
        data_sample = scaled_data
        
    inertia = []
    
    for k in k_values_list:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data_sample)
        inertia.append(kmeans.inertia_)
        
    elbow_df = pd.DataFrame({'k': k_values_list, 'inertia': inertia})
    print("Wide-range elbow calculations complete.")
    return elbow_df

def run_clustering_and_tsne(scaled_data, subreddit_labels, n_clusters, 
                            min_cluster_size=500, perplexity=50):
    """
    Runs K-Means, filters clusters by size, and runs t-SNE for visualization.
    
    Args:
        scaled_data (np.array): The full scaled feature data.
        subreddit_labels (np.array): The corresponding subreddit names.
        n_clusters (int): The initial number of clusters to create (e.g., 1000).
        min_cluster_size (int): The minimum number of members for a cluster 
                                to be "valid" and kept.
        perplexity (int): The perplexity value for t-SNE. A lower value (e.g., 5-10)
                          focuses on local structure and can create clearer
                          visual separation. A higher value (e.g., 50-100)
                          focuses on global structure.

    Returns:
        tuple: 
            (tsne_df_filtered, all_cluster_labels)
            - tsne_df_filtered: DataFrame for plotting (sampled, 2D)
            - all_cluster_labels: Full array of cluster IDs for all 51k+ items
    """
    # --- 1. Final K-Means Clustering ---
    print(f"Running final K-Means clustering with k={n_clusters} on {len(scaled_data)} items...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    
    all_cluster_labels = kmeans.fit_predict(scaled_data)
    print("K-Means complete.")

    # --- 2. Filter Clusters by Size ---
    print(f"Finding valid clusters with > {min_cluster_size} members...")
    cluster_labels_unique, cluster_counts = np.unique(all_cluster_labels, return_counts=True)
    valid_cluster_labels = cluster_labels_unique[cluster_counts > min_cluster_size]
    valid_clusters_set = set(valid_cluster_labels)
    
    n_valid = len(valid_clusters_set)
    n_total = len(cluster_labels_unique)
    print(f"Found {n_valid} valid clusters (out of {n_total}) meeting the size criteria.")

    if n_valid == 0:
        print("No clusters met the size criteria. Try a lower `min_cluster_size`.")
        return pd.DataFrame(columns=['subreddit', 'cluster', 'tsne_x', 'tsne_y']), all_cluster_labels

    # --- 3. t-SNE Dimensionality Reduction ---
    n_tsne_samples = 15000 
    if len(scaled_data) > n_tsne_samples:
        print(f"Running t-SNE with perplexity={perplexity}... (sampling {n_tsne_samples} items)")
        indices = np.random.choice(scaled_data.shape[0], n_tsne_samples, replace=False)
        
        data_sample = scaled_data[indices]
        label_sample = subreddit_labels[indices]
        cluster_sample = all_cluster_labels[indices]
    else:
        print(f"Running t-SNE with perplexity={perplexity} on all items...")
        data_sample = scaled_data
        label_sample = subreddit_labels
        cluster_sample = all_cluster_labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(data_sample)
    print("t-SNE complete.")

    # --- 4. Combine and Filter Results ---
    print("Combining and filtering t-SNE results...")
    tsne_df = pd.DataFrame({
        'subreddit': label_sample,
        'cluster': cluster_sample,
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1]
    })
    
    tsne_df_filtered = tsne_df[tsne_df['cluster'].isin(valid_clusters_set)].copy()
    tsne_df_filtered['cluster'] = tsne_df_filtered['cluster'].astype(str)
    
    print(f"Returning {len(tsne_df_filtered)} items (from the sample) that belong to the {n_valid} valid clusters.")
    
    return tsne_df_filtered, all_cluster_labels

def get_cluster_samples(subreddit_labels, all_cluster_labels, n_samples=10):
    """
    Inspects the clusters by returning a random sample of subreddit names
    from each one.
    
    Args:
        subreddit_labels (np.array): The full list of 51k+ subreddit names.
        all_cluster_labels (np.array): The full list of 51k+ cluster assignments.
        n_samples (int): Number of subreddits to show from each cluster.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are
              lists of sample subreddit names.
    """
    print("Getting samples from each cluster...")
    df = pd.DataFrame({
        'subreddit': subreddit_labels,
        'cluster': all_cluster_labels
    })
    
    cluster_samples = {}
    
    # Group by the cluster ID
    for cluster_id, group in df.groupby('cluster'):
        
        # Get a random sample, or all of them if the group is too small
        if len(group) > n_samples:
            sample = group.sample(n_samples, random_state=42)
        else:
            sample = group
            
        cluster_samples[cluster_id] = sample['subreddit'].tolist()
        
    print(f"Returning samples for {len(cluster_samples)} total clusters.")
    return cluster_samples


# === 2. EMBEDDING-FACTION ANALYSIS ===

def find_strict_subreddits(df_countries, df_embeddings):
    """
    Identifies "strict" subreddits which are more similar to their own country's
    subreddits than to any other country's subreddits based on embeddings.
    """
    df_approved_emb = df_countries.merge(df_embeddings, on="subreddit", how="inner")
    emb_cols = [c for c in df_embeddings.columns if c != "subreddit"]
    
    country_to_embeddings = {}
    for country, group in df_approved_emb.groupby("predicted_country"):
        country_to_embeddings[country] = group[emb_cols].values

    strict_subs = []
    for _, row in df_approved_emb.iterrows():
        subreddit = row["subreddit"]
        country = row["predicted_country"]
        sub_emb = row[emb_cols].values.reshape(1, -1)
        
        same_country_emb = country_to_embeddings[country]
        sim_same = cosine_similarity(sub_emb, same_country_emb).mean()
        
        other_countries = [c for c in country_to_embeddings.keys() if c != country]
        sim_others_max = 0
        if other_countries:
            sim_others = [
                cosine_similarity(sub_emb, country_to_embeddings[oc]).mean() 
                for oc in other_countries
            ]
            sim_others_max = max(sim_others) if sim_others else 0
        
        if sim_same > sim_others_max:
            strict_subs.append(subreddit)
            
    df_strict_approved = df_countries[df_countries["subreddit"].isin(strict_subs)].reset_index(drop=True)
    return df_strict_approved

def find_closest_dissimilar_subreddits(df_strict_approved, df_embeddings):
    """
    Finds the *most similar* (highest cosine similarity) subreddit 
    from a *different* country for each strict subreddit.
    """
    strict_subs_set = set(df_strict_approved["subreddit"])
    df_emb_strict = df_embeddings[df_embeddings["subreddit"].isin(strict_subs_set)].reset_index(drop=True)
    
    embedding_cols = [c for c in df_emb_strict.columns if c != "subreddit"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df_emb_strict[embedding_cols])
    
    subreddit_names = df_emb_strict["subreddit"].tolist()
    sub_to_country = dict(zip(df_strict_approved["subreddit"], df_strict_approved["predicted_country"]))
    
    sim_matrix = cosine_similarity(X, X)
    
    most_similar_subreddits = []
    for i, sub in enumerate(subreddit_names):
        
        similarities = sim_matrix[i].copy()
        
        # Mask out subreddits from the same country
        for j, other_sub in enumerate(subreddit_names):
            if sub_to_country.get(other_sub) == sub_to_country.get(sub) or i == j:
                # Set to a very low number so argmax won't pick it
                similarities[j] = -np.inf 
        
        closest_idx = similarities.argmax()
        closest_sub = subreddit_names[closest_idx]
        
        most_similar_subreddits.append({
            "subreddit": sub,
            "predicted_country": sub_to_country.get(sub),
            "most_similar_subreddit": closest_sub,
            "most_similar_country": sub_to_country.get(closest_sub),
            "similarity_score": similarities[closest_idx]
        })
        
    return pd.DataFrame(most_similar_subreddits)


# === 3. NETWORK-BASED FACTION ANALYSIS WITH POSITIVE POSTS ===

def calculate_country_activity(df_post_with_1_country, df_countries):
    """Counts total posts originating from each country's subreddits."""
    sub_to_country = df_countries.set_index("subreddit")["predicted_country"].to_dict()
    df_posts = df_post_with_1_country.copy()
    df_posts["country"] = df_posts["SOURCE_SUBREDDIT"].map(sub_to_country)
    df_posts = df_posts.dropna(subset=["country"])
    
    country_activity = (
        df_posts.groupby("country")
        .size()
        .reset_index(name="num_posts")
        .sort_values("num_posts", ascending=False)
        .reset_index(drop=True)
    )
    return country_activity

def get_signed_country_links(df_post_with_1_country, df_countries):
    """
    Aggregates all links (positive and negative) between countries.
    """
    df_posts = _map_countries(df_post_with_1_country.copy(), df_countries)
    df_posts = df_posts[df_posts['source_country'] != df_posts['target_country']]

    # Count positive links
    positive_links = (
        df_posts[df_posts["LINK_SENTIMENT"] == 1]
        .groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="positive_posts")
    )
    
    # Count negative links
    negative_links = (
        df_posts[df_posts["LINK_SENTIMENT"] == -1]
        .groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="negative_posts")
    )
    
    # Merge and fill NaNs
    signed_links = pd.merge(
        positive_links, 
        negative_links, 
        on=["source_country", "target_country"], 
        how="outer"
    ).fillna(0)
    
    # Calculate net sentiment
    signed_links["net_sentiment"] = signed_links["positive_posts"] - signed_links["negative_posts"]
    return signed_links

def _map_countries(df_posts, df_countries):
    """Helper to map source/target subreddits to countries."""
    sub_to_country = df_countries.set_index("subreddit")["predicted_country"].to_dict()
    df_posts['source_country'] = df_posts['SOURCE_SUBREDDIT'].map(sub_to_country)
    df_posts['target_country'] = df_posts['TARGET_SUBREDDIT'].map(sub_to_country)
    return df_posts.dropna(subset=["source_country", "target_country"])

def get_positive_country_links(df_post_with_1_country, df_countries):
    """Counts positive-sentiment posts between countries."""
    df_positive = df_post_with_1_country[df_post_with_1_country["LINK_SENTIMENT"] == 1].copy()
    df_positive = _map_countries(df_positive, df_countries)
    
    country_links = (
        df_positive.groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="num_positive_posts")
    )
    return country_links

def build_interaction_graph(country_links_df, weight_col="num_positive_posts"):
    """Builds a NetworkX graph from a country links DataFrame."""
    G = nx.Graph()
    for _, row in country_links_df.iterrows():
        source, target, weight = row["source_country"], row["target_country"], row[weight_col]
        if source != target:
            if G.has_edge(source, target):
                G[source][target]["weight"] += weight
            else:
                G.add_edge(source, target, weight=weight)
    return G

def detect_factions(graph):
    """Detects modularity-based communities in a graph."""
    communities = community.greedy_modularity_communities(graph, weight="weight")
    
    factions_df = pd.DataFrame([
        {"country": country, "faction": i}
        for i, comm_set in enumerate(communities)
        for country in comm_set
    ])
    
    factions_summary = (
        factions_df.sort_values("faction")
        .groupby("faction")["country"]
        .apply(list)
        .reset_index(name="countries")
    )
    factions_summary["num_countries"] = factions_summary["countries"].apply(len)
    factions_summary = factions_summary.sort_values("num_countries", ascending=False).reset_index(drop=True)
    
    return factions_summary, factions_df

def diagnose_unfactioned_countries(df_countries, factions_df, df_post_with_1_country):
    """
    Finds countries that were approved but did not end up in a faction,
    and provides a reason why.

    Args:
        df_approved (pd.DataFrame): DataFrame of approved subreddits and countries.
        factions_df (pd.DataFrame): DataFrame of countries and their assigned faction.
        df_post_with_1_country (pd.DataFrame): DataFrame of all posts with at least
                                               one approved subreddit.

    Returns:
        pd.DataFrame: A summary of missing countries and the reason they are missing.
    """
    print("Diagnosing unfactioned countries...")

    # --- 1. Map countries to all posts ---
    # This is needed for all subsequent logic
    # We use the internal helper _map_countries
    print("Mapping countries to post data...")
    sub_to_country = df_countries.drop_duplicates(subset='subreddit').set_index("subreddit")["predicted_country"].to_dict()
    
    # Create a mapped copy of the posts DataFrame
    df_posts_mapped = df_post_with_1_country.copy()
    df_posts_mapped['source_country'] = df_posts_mapped['SOURCE_SUBREDDIT'].map(sub_to_country)
    df_posts_mapped['target_country'] = df_posts_mapped['TARGET_SUBREDDIT'].map(sub_to_country)
    
    # --- 2. Create helper DataFrames ---
    df_positive_posts = df_posts_mapped[df_posts_mapped["LINK_SENTIMENT"] == 1].copy()
    df_negative_posts = df_posts_mapped[df_posts_mapped["LINK_SENTIMENT"] == -1].copy()

    # --- 3. Identify country sets ---
    all_countries = set(df_countries["predicted_country"].dropna().unique())
    countries_in_factions = set(factions_df["country"].unique())
    countries_not_in_faction = sorted(all_countries - countries_in_factions)

    print(f"Total approved countries: {len(all_countries)}")
    print(f"Countries in factions: {len(countries_in_factions)}")
    print(f"Countries to diagnose: {len(countries_not_in_faction)}")

    # --- 4. Diagnose why they are missing ---
    reasons = defaultdict(list)
    
    # Create a set of country subreddits for faster filtering
    country_subreddit_map = df_countries.groupby('predicted_country')['subreddit'].apply(set).to_dict()

    for country in countries_not_in_faction:
        
        country_subs = country_subreddit_map.get(country, set())
        if not country_subs:
            reasons[country].append("no subreddits found in df_approved")
            continue

        # (a) Check total posts (any sentiment)
        # Note: We use the *original* df_post_with_1_country for this check
        total_posts_df = df_post_with_1_country[
            df_post_with_1_country["SOURCE_SUBREDDIT"].isin(country_subs) |
            df_post_with_1_country["TARGET_SUBREDDIT"].isin(country_subs)
        ]
        if total_posts_df.empty:
            reasons[country].append("no posts at all")
            continue

        # (b) Check for positive posts with *other* countries
        # We use the pre-filtered df_positive_posts for this
        pos_links = df_positive_posts[
            ((df_positive_posts["source_country"] == country) & (df_positive_posts["target_country"] != country)) |
            ((df_positive_posts["target_country"] == country) & (df_positive_posts["source_country"] != country))
        ]
        
        if pos_links.empty:
            # (c) Check for negative posts
            neg_links = df_negative_posts[
                (df_negative_posts["source_country"] == country) |
                (df_negative_posts["target_country"] == country)
            ]
            if not neg_links.empty:
                reasons[country].append("only negative or self-posts (no positive links to others)")
            else:
                reasons[country].append("only neutral or self-posts (no positive/negative links to others)")
        else:
            reasons[country].append("had positive posts but was isolated (not linked strongly enough to form faction)")

    # --- 5. Combine results in a summary dataframe ---
    if not reasons:
        print("No missing countries found.")
        return pd.DataFrame(columns=["country", "reason"])

    missing_countries_summary = pd.DataFrame([
        {"country": c, "reason": ", ".join(r)}
        for c, r in reasons.items()
    ]).sort_values("country").reset_index(drop=True)

    print("Diagnosis complete.")
    return missing_countries_summary

def detect_normalized_factions(df_post_with_1_country, df_countries):
    """
    Detects factions based on normalized positive interaction weights.
    Weight = N_posts(A,B) / sqrt(TotalPosts(A) * TotalPosts(B))
    """
    df_positive = df_post_with_1_country[df_post_with_1_country["LINK_SENTIMENT"] == 1].copy()
    df_positive = _map_countries(df_positive, df_countries)
    
    country_links = (
        df_positive.groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="num_positive_posts")
    )
    
    country_activity = (
        pd.concat([df_positive["source_country"], df_positive["target_country"]])
        .value_counts()
        .rename_axis("country")
        .reset_index(name="total_posts")
    )
    
    country_links = country_links.merge(
        country_activity.rename(columns={"country": "source_country", "total_posts": "source_total"}), 
        on="source_country"
    ).merge(
        country_activity.rename(columns={"country": "target_country", "total_posts": "target_total"}), 
        on="target_country"
    )
    
    country_links["normalized_weight"] = country_links["num_positive_posts"] / \
                                         np.sqrt(country_links["source_total"] * country_links["target_total"])

    G_norm = build_interaction_graph(country_links, weight_col="normalized_weight")
    return detect_factions(G_norm)


# === 4. TEMPORAL & ACTIVITY ANALYSIS ===

def map_countries_to_posts(df_posts, df_countries):
    """Helper to map source/target subreddits to countries."""
    df_links_with_countries = _map_countries(df_posts.copy(), df_countries)
    df_links_with_countries["TIMESTAMP"] = pd.to_datetime(df_links_with_countries["TIMESTAMP"], errors='coerce')
    df_links_with_countries["year_quarter"] = df_links_with_countries["TIMESTAMP"].dt.to_period("Q")
    return df_links_with_countries.dropna(subset=['year_quarter'])
    
def analyze_factions_over_time(df_post_between_countries):
    """Calculates factions for each quarter based on raw post counts."""
    quarterly_summary = []
    
    for period, group in df_post_between_countries.groupby("year_quarter"):
        positive_group = group[group["LINK_SENTIMENT"] == 1]
        if positive_group.empty:
            continue
            
        quarter_links = (
            positive_group.groupby(["source_country", "target_country"])
            .size()
            .reset_index(name="num_positive_posts")
        )
        
        G_quarter = build_interaction_graph(quarter_links, weight_col="num_positive_posts")
        if G_quarter.number_of_nodes() == 0:
            continue
            
        summary, _ = detect_factions(G_quarter)
        summary["year_quarter"] = str(period)
        quarterly_summary.append(summary)

    return pd.concat(quarterly_summary).reset_index(drop=True)

def analyze_source_normalized_factions_over_time(df_post_between_countries):
    """
    Calculates factions for each quarter, normalizing by source country post count.
    """
    quarterly_summary = []
    
    for period, group in df_post_between_countries.groupby("year_quarter"):
        country_total_posts = group.groupby("source_country").size().reset_index(name="total_posts")
        
        group = group.merge(country_total_posts, on="source_country", how="left")
        group["normalized_weight"] = 1 / group["total_posts"]
        
        positive_group = group[group["LINK_SENTIMENT"] == 1]
        if positive_group.empty:
            continue

        quarter_links = (
            positive_group.groupby(["source_country", "target_country"])["normalized_weight"]
            .sum()
            .reset_index(name="weighted_interaction")
        )
        
        G_quarter = build_interaction_graph(quarter_links, weight_col="weighted_interaction")
        if G_quarter.number_of_nodes() == 0:
            continue
            
        summary, _ = detect_factions(G_quarter)
        summary["year_quarter"] = str(period)
        quarterly_summary.append(summary)
        
    return pd.concat(quarterly_summary).reset_index(drop=True)

def find_stable_pairs(quarterly_factions_summary_df):
    """Finds pairs of countries that frequently appear in the same faction."""
    pair_counter = Counter()
    total_quarters = quarterly_factions_summary_df["year_quarter"].nunique()
    
    for _, quarter_group in quarterly_factions_summary_df.groupby("year_quarter"):
        for _, row in quarter_group.iterrows():
            countries = row["countries"]
            for pair in itertools.combinations(sorted(countries), 2):
                pair_counter[pair] += 1
                
    stable_pairs_df = pd.DataFrame([
        {"country1": pair[0], "country2": pair[1], "quarters_together": count}
        for pair, count in pair_counter.items()
    ])
    stable_pairs_df["fraction_quarters_together"] = stable_pairs_df["quarters_together"] / total_quarters
    stable_pairs_df = stable_pairs_df[stable_pairs_df["fraction_quarters_together"] < 1]
    return stable_pairs_df.sort_values("fraction_quarters_together", ascending=False).reset_index(drop=True)

def calculate_loyalty_scores(quarterly_factions_summary_df):
    """
    Calculates loyalty scores for countries based on partner stability.
    Score < 1 means the country switches partners.
    """
    country_quarterly_allies = defaultdict(list)
    for _, row in quarterly_factions_summary_df.iterrows():
        countries = set(row["countries"])
        for country in countries:
            allies = countries - {country}
            country_quarterly_allies[country].append(allies)
            
    loyalty_list = []
    for country, quarterly_allies in country_quarterly_allies.items():
        ally_counter = Counter(itertools.chain.from_iterable(quarterly_allies))
        if not ally_counter:
            continue
        max_count = max(ally_counter.values())
        loyalty_score = max_count / len(quarterly_allies)
        loyalty_list.append({
            "country": country,
            "loyalty_score": loyalty_score,
            "total_quarters": len(quarterly_allies)
        })
        
    loyalty_df = pd.DataFrame(loyalty_list)
    return loyalty_df[loyalty_df["loyalty_score"] < 1].sort_values("loyalty_score", ascending=False).reset_index(drop=True)

def find_switch_triggers(quarterly_factions_summary_df, df_post_between_countries):
    """
    Analyzes if negative posts in a *previous* quarter correlate with
    a faction switch in the *current* quarter.
    """
    df_neg = df_post_between_countries[df_post_between_countries["LINK_SENTIMENT"] == -1].copy()
    if df_neg.empty:
        return pd.DataFrame(columns=[
            "country", "from_faction", "to_faction", 
            "previous_quarter", "switch_quarter", 
            "neg_posts_prior", "neg_allies_involved"
        ])

    # Create helper structures for quick lookup
    country_faction_over_time = defaultdict(list)
    for _, row in quarterly_factions_summary_df.iterrows():
        period = pd.Period(row["year_quarter"], freq='Q')
        for country in row["countries"]:
            country_faction_over_time[country].append((period, row["faction"]))

    quarter_faction_members = {
        (pd.Period(row["year_quarter"], freq='Q'), row["faction"]): set(row["countries"])
        for _, row in quarterly_factions_summary_df.iterrows()
    }

    # Identify switches
    switches = []
    for country, history in country_faction_over_time.items():
        history.sort(key=lambda x: x[0])
        for i in range(1, len(history)):
            prev_q, prev_f = history[i-1]
            curr_q, curr_f = history[i]
            if curr_q == prev_q + 1 and prev_f != curr_f:
                switches.append({
                    "country": country, "from_faction": prev_f, "to_faction": curr_f,
                    "previous_quarter": prev_q, "switch_quarter": curr_q
                })
    switches_df = pd.DataFrame(switches)
    if switches_df.empty:
        return pd.DataFrame(columns=[
            "country", "from_faction", "to_faction", 
            "previous_quarter", "switch_quarter", 
            "neg_posts_prior", "neg_allies_involved"
        ])

    # Check for negative posts in the PREVIOUS quarter
    def check_neg_posts(row):
        old_allies = quarter_faction_members.get((row["previous_quarter"], row["from_faction"]), set())
        old_allies -= {row["country"]}
        if not old_allies:
            return pd.Series([0, []])
        
        neg_posts = df_neg[
            (df_neg["year_quarter"] == row["previous_quarter"]) &
            (
                ((df_neg["source_country"] == row["country"]) & (df_neg["target_country"].isin(old_allies))) |
                ((df_neg["target_country"] == row["country"]) & (df_neg["source_country"].isin(old_allies)))
            )
        ]
        
        involved = pd.unique(neg_posts[["source_country", "target_country"]].values.ravel('K'))
        involved = sorted([c for c in involved if c != row["country"]])
        return pd.Series([len(neg_posts), involved])

    switches_df[["neg_posts_prior", "neg_allies_involved"]] = switches_df.apply(check_neg_posts, axis=1)
    
    return switches_df[switches_df["neg_posts_prior"] > 0].sort_values("neg_posts_prior", ascending=False).reset_index(drop=True)

# === 5. RECIPROCITY ANALYSIS ===

def response_global(df_country):
    ## --- 1. DataFrame Preparation ---
    df_clean = df_country.dropna(subset=['SOURCE_COUNTRY', 'TARGET_COUNTRY'])
    df_clean['TIMESTAMP'] = pd.to_datetime(df_clean['TIMESTAMP'])
    
    # Sort the entire DataFrame by TIMESTAMP *once* at the beginning.
    # This is fundamental to our logic.
    df_clean = df_clean.sort_values(by='TIMESTAMP')

    # Create a 'pair_key' using frozenset to group (A, B) and (B, A) together
    df_clean['pair_key'] = df_clean.apply(
        lambda row: frozenset([row['SOURCE_COUNTRY'], row['TARGET_COUNTRY']]),
        axis=1
    )
    # Filter out pairs that only have one interaction
    pair_counts = df_clean['pair_key'].value_counts()
    valid_pairs = pair_counts[pair_counts > 1].index
    df_analysis = df_clean[df_clean['pair_key'].isin(valid_pairs)]

    # This print statement was missing in the translation request, adding it in English
    print(f"DataFrame ready. Analyzing {len(valid_pairs)} pairs.")

    ## --- 2. Global Totals Calculation (Deterministic Logic) ---

    window_days = 7
    total_initiators_global = 0
    total_responses_global = 0

    # Iterate over each unique pair with a progress bar
    for pair in tqdm(valid_pairs, desc="Analyzing pairs"):
        
        # Skip A->A self-interactions (which have len < 2)
        if len(pair) < 2:
            continue
            
        # Get all interactions for this pair
        # They will ALREADY be sorted by TIMESTAMP thanks to the initial sort
        df_pair = df_analysis[df_analysis['pair_key'] == pair]

        # --- Find the "True" Initiator (A) ---
        # Get the very first interaction ever for this pair
        first_interaction = df_pair.iloc[0]
        
        # Define roles based on that first interaction
        # country_A is the initiator, country_B is the responder
        country_A = first_interaction['SOURCE_COUNTRY']
        country_B = first_interaction['TARGET_COUNTRY']
        
        # --- Get all interactions for these defined roles ---
        
        # Get all A -> B interactions (initiations)
        df_A_to_B = df_pair[
            (df_pair['SOURCE_COUNTRY'] == country_A) & 
            (df_pair['TARGET_COUNTRY'] == country_B)
        ]
        # Get all B -> A interactions (responses)
        df_B_to_A = df_pair[
            (df_pair['SOURCE_COUNTRY'] == country_B) & 
            (df_pair['TARGET_COUNTRY'] == country_A)
        ]

        # If the "responder" (B) has never posted, 
        # we cannot measure reciprocity for this pair
        if df_A_to_B.empty or df_B_to_A.empty:
            continue

        # --- Calculate conditional probability ---
        df_A_to_B = df_A_to_B.copy()
        df_B_to_A = df_B_to_A.copy()
        # Copy timestamps to new columns to preserve them after the merge
        df_A_to_B['TIMESTAMP_A'] = df_A_to_B['TIMESTAMP']
        df_B_to_A['TIMESTAMP_B'] = df_B_to_A['TIMESTAMP']

        # Find the first response (B->A) that occurred *after* each initiation (A->B)
        merged = pd.merge_asof(
            df_A_to_B,  # Already sorted by TIMESTAMP
            df_B_to_A,  # Already sorted by TIMESTAMP
            on='TIMESTAMP',          # The column to merge on
            direction='forward',   # Find the first response *after* the initiation
            suffixes=('_A', '_B')
        )
        
        # Calculate the time delta
        merged['response_time'] = merged['TIMESTAMP_B'] - merged['TIMESTAMP_A']
        
        # Check if the response occurred within the 7-day window
        merged['responded_within_7_days'] = (
            merged['response_time'] <= pd.Timedelta(days=window_days)
        )

        # --- Update global counters ---
        total_initiators_global += len(df_A_to_B)
        total_responses_global += merged['responded_within_7_days'].sum()
        
    ## --- 3. Global Results Analysis ---

    # Calculate the final probability, handling the division by zero case
    if total_initiators_global == 0:
        print("\nWARNING: No 'initiator' interactions found. Cannot calculate probability.")
        conditioned_prob = 0
    else:
        conditioned_prob = total_responses_global / total_initiators_global

    # Print the final result (using A and B)
    print("\n--- Global Reciprocity Analysis ---")
    print(f"Total 'initiator' interactions (A->B) analyzed: {total_initiators_global}")
    print(f"Total responses (B->A) within 7 days: {total_responses_global:.0f}")
    print(f"GLOBAL Conditional Probability P(B->A | A->B in 7d): {conditioned_prob:.2%}")

def response_intra_country(df_combined):
    
    # 1. Filter for INTRA-COUNTRY interactions
    #    - Countries must be the same
    #    - Countries must not be NaN
    #    - Subreddits must be DIFFERENT
    df_intra = df_combined[
        (df_combined['SOURCE_COUNTRY'] == df_combined['TARGET_COUNTRY']) &
        (df_combined['SOURCE_COUNTRY'].notna()) &
        (df_combined['SOURCE_SUBREDDIT'] != df_combined['TARGET_SUBREDDIT'])
    ].copy()

    # 2. Ensure TIMESTAMP is datetime and SORT
    df_intra['TIMESTAMP'] = pd.to_datetime(df_intra['TIMESTAMP'])
    # SINGLE Sort at the beginning: this is fundamental
    df_intra = df_intra.sort_values(by='TIMESTAMP')

    print(f"Found {len(df_intra)} intra-country interactions (between different subreddits).")

    ## --- 2. Identify all unique SUBREDDIT pairs ---
    df_intra['pair_key'] = df_intra.apply(
        lambda row: frozenset([row['SOURCE_SUBREDDIT'], row['TARGET_SUBREDDIT']]),
        axis=1
    )
    pair_counts = df_intra['pair_key'].value_counts()
    valid_pairs = pair_counts[pair_counts > 1].index
    df_analysis = df_intra[df_intra['pair_key'].isin(valid_pairs)]

    print(f"Analyzing {len(valid_pairs)} unique subreddit pairs.")

    ## --- 3. Global Totals Calculation (Deterministic Logic) ---
    window_days = 7
    total_initiators_global = 0
    total_responses_global = 0

    # Iterate over each UNIQUE subreddit pair
    for pair in tqdm(valid_pairs, desc="Analyzing subreddit pairs"):
        
        # --- Find the "True" Initiator (Sub_A) ---
        
        # Get all interactions for the pair (already sorted by TIMESTAMP)
        df_pair = df_analysis[df_analysis['pair_key'] == pair]
        
        # Find the very first interaction ever
        first_interaction = df_pair.iloc[0]
        
        # Define roles based on that first interaction
        Sub_A = first_interaction['SOURCE_SUBREDDIT'] # Initiator
        Sub_B = first_interaction['TARGET_SUBREDDIT'] # Responder
        
        # --- End of deterministic logic ---

        # Define "Initiation" (A -> B) and "Response" (B -> A)
        df_A_to_B = df_pair[
            (df_pair['SOURCE_SUBREDDIT'] == Sub_A) & 
            (df_pair['TARGET_SUBREDDIT'] == Sub_B)
        ]
        df_B_to_A = df_pair[
            (df_pair['SOURCE_SUBREDDIT'] == Sub_B) & 
            (df_pair['TARGET_SUBREDDIT'] == Sub_A)
        ]

        # If the "responder" Sub_B never posted, skip
        if df_A_to_B.empty or df_B_to_A.empty:
            continue

        # --- Calculate conditional probability ---
        df_A_to_B = df_A_to_B.copy()
        df_B_to_A = df_B_to_A.copy()
        df_A_to_B['TIMESTAMP_A'] = df_A_to_B['TIMESTAMP']
        df_B_to_A['TIMESTAMP_B'] = df_B_to_A['TIMESTAMP']

        merged = pd.merge_asof(
            df_A_to_B, # Already sorted
            df_B_to_A, # Already sorted
            on='TIMESTAMP',
            direction='forward',
            suffixes=('_A', '_B')
        )
        
        # Calculate the time delta
        merged['response_time'] = merged['TIMESTAMP_B'] - merged['TIMESTAMP_A']
        # Check if the response occurred within the 7-day window
        merged['responded_within_7_days'] = (
            merged['response_time'] <= pd.Timedelta(days=window_days)
        )

        # --- Update global counters ---
        total_initiators_global += len(df_A_to_B)
        total_responses_global += merged['responded_within_7_days'].sum()
        
    ## --- 4. Global Results Analysis ---
    cond_prob = 0.0
    if total_initiators_global > 0:
        cond_prob = total_responses_global / total_initiators_global
    else:
        # Print in English
        print("No initiator-response interactions found.")

    # Print in English
    print("\n--- Global INTRA-COUNTRY Reciprocity Analysis ---")
    print(f"Total 'initiator' interactions (SubA -> SubB) analyzed: {total_initiators_global}")
    print(f"Total responses (SubB -> SubA) within 7 days: {total_responses_global:.0f}")
    print(f"GLOBAL Conditional Probability P(SubB->SubA | SubA->SubB in 7d): {cond_prob:.2%}")

# === 6. RESPONSE SIMILARITY ANALYSIS ===

def find_reciprocity_pairs_and_similarity(df_interactions, features_list, matches_csv):
    """
    Finds A->B => B->A pairs (all sentiments) within 7 days 
    and calculates cosine similarity using only the 'features_list'.
    """
    WINDOW_DAYS = 7
    country_subs_list = pd.read_csv(matches_csv)['Subreddit'].tolist()
    # TRIGGER = Any sub posting to a country
    df_triggers = df_interactions[
        (df_interactions['TARGET_SUBREDDIT'].isin(country_subs_list)) &
        (df_interactions['SOURCE_SUBREDDIT'] != df_interactions['TARGET_SUBREDDIT'])
    ]
        
    # RESPONSES = A country posting to any sub
    df_responses = df_interactions[
        (df_interactions['SOURCE_SUBREDDIT'].isin(country_subs_list)) &
        (df_interactions['SOURCE_SUBREDDIT'] != df_interactions['TARGET_SUBREDDIT'])
    ]

    response_lookup = {}
    for (source_country, target_sub), group in df_responses.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']):
        response_lookup[(source_country, target_sub)] = group[['TIMESTAMP']].reset_index() 

    similarity_scores = []
        
    if len(df_triggers) == 0: return []

    # Iterate over triggers
    for trigger_id, trigger_post in tqdm(df_triggers.iterrows(), total=len(df_triggers)):
        source_A = trigger_post['SOURCE_SUBREDDIT']
        target_B = trigger_post['TARGET_SUBREDDIT']
        time_A = trigger_post['TIMESTAMP']
            
        response_key = (target_B, source_A)
            
        if response_key in response_lookup:
            possible_responses = response_lookup[response_key]
                
            time_start = time_A
            time_end = time_A + pd.Timedelta(days=WINDOW_DAYS)
                
            valid_responses = possible_responses[
                (possible_responses['TIMESTAMP'] > time_start) & 
                (possible_responses['TIMESTAMP'] <= time_end)
            ]
                
            if not valid_responses.empty:
                first_response_row = valid_responses.iloc[0]
                first_response_id = first_response_row['POST_ID'] 
                    
                try:
                    # Use only the 'features_list' (style vector)
                    vector_trigger = df_interactions.loc[[trigger_id], features_list].values
                    vector_response = df_interactions.loc[[first_response_id], features_list].values
                        
                    similarity = cosine_similarity(vector_trigger, vector_response)[0][0]
                    similarity_scores.append(similarity)
                except KeyError:
                    pass 
                        
    return similarity_scores

def response_similarity(df_combined, matches_csv):
    # --- 1. Style Feature Definition ---
    df_combined['TIMESTAMP'] = pd.to_datetime(df_combined['TIMESTAMP'])
    style_features_list = [
        # Tone/Sentiment Measures (your VADER)
        'sent_pos',
        'sent_neg',
        'sent_compound',

        # Psychological Style: Function Words (LIWC)
        'LIWC_I', 'LIWC_We',
        'LIWC_You', 'LIWC_SheHe', 'LIWC_They',
        
        # Psychological Style: Other (LIWC)
        'LIWC_Assent', 'LIWC_Dissent', 'LIWC_Nonflu', 'LIWC_Filler'
    ]
    # --------------------------------------------------------


    # --- 2. Data Preparation ---
    # Ensure 'df_combined', 'post_props_cols', 'country_subs_list' exist.

    # Verify that the chosen style features exist
    for col in style_features_list:
        if col not in df_combined.columns:
            # This check should not fail now
            raise ValueError(f"Style feature '{col}' was not found in df_combined.")

    # Columns for analysis: only style features + necessary ones
    other_needed_cols = ['POST_ID', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT',
                        'TIMESTAMP', 'LINK_SENTIMENT']
    # We use 'style_features_list' to filter
    df_analysis = df_combined[other_needed_cols + style_features_list].copy()
    df_analysis = df_analysis.set_index('POST_ID')

    print("Analysis DataFrame ready.")

    WINDOW_DAYS = 7

    # --- 3. Main Analysis Execution (Reciprocity) ---
    reciprocity_similarities = find_reciprocity_pairs_and_similarity(
        df_analysis, 
        style_features_list,  # <-- We pass the style list
        matches_csv
    )

    # --- 4. Baseline Analysis Execution (Random Control) ---
    baseline_similarities = []
    if not reciprocity_similarities:
        print("\nNo reciprocity pairs found. Cannot run baseline.")
    else:
        print("\n--- Starting Baseline Analysis (Control Group) ---")
        
        # Number of samples to create (same as test group)
        N = len(reciprocity_similarities)
        print(f"Creating {N} random post pairs for comparison...")
        
        # Extract all style vectors
        all_style_vectors_df = df_analysis[style_features_list]
        
        # Sample N random posts for 'Group A'
        random_vectors_A = all_style_vectors_df.sample(n=N, replace=True).values
        
        # Sample N random posts for 'Group B'
        random_vectors_B = all_style_vectors_df.sample(n=N, replace=True).values
        
        # Calculate similarity for each random pair
        for i in tqdm(range(N)):
            sim = cosine_similarity([random_vectors_A[i]], [random_vectors_B[i]])[0][0]
            baseline_similarities.append(sim)

        print("Baseline analysis complete.")

    # --- 5. Statistical Comparison and Visualization ---
    if reciprocity_similarities and baseline_similarities:
        sim_series_reciprocal = pd.Series(reciprocity_similarities, name='Reciprocal')
        sim_series_baseline = pd.Series(baseline_similarities, name='Random')
        
        print("\n--- Statistics (Test Group: Reciprocal) ---")
        print(sim_series_reciprocal.describe())
        
        print("\n--- Statistics (Control Group: Random) ---")
        print(sim_series_baseline.describe())
        
        # Statistical test (t-test)
        try:
            # T-test (if data is ~normal)
            t_stat, p_value = ttest_ind(sim_series_reciprocal, sim_series_baseline, 
                                        equal_var=False, alternative='greater')
            print(f"\n--- T-Test (Reciprocal > Random) ---")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")

            # Mann-Whitney U (more robust if data is not normal)
            mwu_stat, mwu_p_value = mannwhitneyu(sim_series_reciprocal, sim_series_baseline, 
                                                alternative='greater')
            print(f"\n--- Mann-Whitney U Test (Reciprocal > Random) ---")
            print(f"U-statistic: {mwu_stat:.4f}")
            print(f"P-value: {mwu_p_value:.4f}")

            # Interpretation
            print("\n--- Test Interpretation ---")
            if mwu_p_value < 0.05: # We use the U-test p-value (95% confidence level)
                print(f"SIGNIFICANT RESULT (p < 0.05):")
                print("The linguistic style of reciprocal response posts is SIGNIFICANTLY MORE SIMILAR")
                print("than that of two random posts. There is evidence of stylistic mirroring.")
            else:
                print("NON-SIGNIFICANT RESULT (p >= 0.05):")
                print("There is no significant statistical difference between the similarity of")
                print("reciprocal pairs and that of random pairs. No evidence of mirroring.")

        except Exception as e:
            print(f"Error during statistical test: {e}")

        # Plot (KDE - Kernel Density Estimate)
        plt.figure(figsize=(12, 7))
        sns.kdeplot(sim_series_reciprocal, fill=True, label='Reciprocal Similarity (Test)', clip=(-1, 1))
        sns.kdeplot(sim_series_baseline, fill=True, label='Random Similarity (Control)', clip=(-1, 1))
        
        # Calculate and plot the medians
        median_reciprocal = sim_series_reciprocal.median()
        median_baseline = sim_series_baseline.median()
        
        plt.axvline(median_reciprocal, color=sns.color_palette()[0], linestyle='--', 
                    label=f'Reciprocal Median: {median_reciprocal:.2f}')
        plt.axvline(median_baseline, color=sns.color_palette()[1], linestyle=':', 
                    label=f'Random Median: {median_baseline:.2f}')
        
        plt.title('Style Similarity Distribution: Reciprocal Pairs vs. Random Pairs')
        plt.xlabel('Cosine Similarity (Based on Style Vector)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
        plt.close()

    else:
        print("\nAnalysis not completed (missing reciprocity or baseline data).")

# === 7. POLITICAL IDEOLOGY ANALYSIS ===

def analyze_political_activity(df_countries, df_politics):
    """
    Analyzes the activity of political subreddits based on all posts.
    
    Calculates:
    1. Total posts involving a political subreddit (as source or target).
    2. The most active political subreddit.
    3. The most active political ideology.
    
    Args:
        df_countries (pd.DataFrame): All posts.
        df_politics (pd.DataFrame): The DataFrame of political subreddits, 
                                    containing 'Matched Subreddit' and 'Ideology'.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    print("Analyzing political subreddit activity...")
    
    # --- Step 1: Filter posts involving political subreddits ---
    politic_subs_set = set(df_politics['Matched Subreddit'])
    
    df_politic_posts = df_countries[
        df_countries['SOURCE_SUBREDDIT'].isin(politic_subs_set) |
        df_countries['TARGET_SUBREDDIT'].isin(politic_subs_set)
    ].copy()
    
    total_posts = len(df_politic_posts)
    if total_posts == 0:
        print("No posts found involving the provided political subreddits.")
        return {
            'total_political_posts': 0,
            'most_active_subreddit': None,
            'most_active_ideology': None,
            'ideology_activity_summary': pd.Series(),
            'subreddit_activity_summary': pd.Series()
        }

    # --- Step 2: Find most active subreddit ---
    
    # Combine all mentions of subreddits from these posts
    all_mentions = pd.concat([
        df_politic_posts['SOURCE_SUBREDDIT'], 
        df_politic_posts['TARGET_SUBREDDIT']
    ])
    
    # Filter this list to *only* include the political subreddits
    # (This correctly attributes activity to them, not to their neighbors)
    politic_mentions_only = all_mentions[all_mentions.isin(politic_subs_set)]
    
    # Get counts for each political subreddit
    subreddit_activity = politic_mentions_only.value_counts()
    most_active_sub_name = subreddit_activity.index[0]
    most_active_sub_count = subreddit_activity.iloc[0]
    
    # --- Step 3: Find most active ideology ---
    
    # Create the subreddit -> ideology map
    ideology_map = df_politics.set_index('Matched Subreddit')['Ideology']
    
    # Convert activity Series to DataFrame for easier mapping
    subreddit_activity_df = subreddit_activity.reset_index()
    subreddit_activity_df.columns = ['subreddit', 'count']
    
    # Map ideologies
    subreddit_activity_df['ideology'] = subreddit_activity_df['subreddit'].map(ideology_map)
    
    # Sum activity by ideology
    ideology_activity = subreddit_activity_df.groupby('ideology')['count'].sum().sort_values(ascending=False)
    most_active_ideology_name = ideology_activity.index[0]
    most_active_ideology_count = ideology_activity.iloc[0]

    # --- Step 4: Prepare results ---
    results = {
        'total_political_posts': total_posts,
        'most_active_subreddit': {
            'name': most_active_sub_name,
            'posts': int(most_active_sub_count)
        },
        'most_active_ideology': {
            'name': most_active_ideology_name,
            'posts': int(most_active_ideology_count)
        },
        'ideology_activity_summary': ideology_activity,
        'subreddit_activity_summary': subreddit_activity
    }
    
    print("Activity analysis complete.")
    return results

def analyze_pure_cross_interactions(df_posts, df_countries, df_politics, link_sentiment=1, year=2015):
    """
    Analyzes interactions between 'pure' country subreddits and 'pure'
    political subreddits, based on the provided dataframes.

    'Pure' means subreddits are mutually exclusive between the two sets.
    
    Args:
        df_posts (pd.DataFrame): All posts, must include 'TIMESTAMP', 
                                    'LINK_SENTIMENT', 'SOURCE_SUBREDDIT', 
                                    'TARGET_SUBREDDIT'.
        df_countries (pd.DataFrame): Approved country subreddits.
        df_politics (pd.DataFrame): Discovered political subreddits (from 
                                    find_political_subreddits).
        link_sentiment (int, optional): Filter by sentiment. 
                                        1=positive, -1=negative, 0=neutral, 
                                        None=all. Defaults to 1.
        year (int, optional): Filter by a specific year. 
                              None=all years. Defaults to 2015.

    Returns:
        pd.DataFrame: A summary table with Countries as rows, 
                      Ideologies as columns, and interaction counts.
    """
    print("--- Step 1: Create Mutually Exclusive Subreddit Sets ---")
    
    # 1. Create sets
    country_subs_all = set(df_countries['subreddit'])
    politic_subs_all = set(df_politics['Matched Subreddit'])

    # 2. Create "pure" sets
    country_subs_pure = country_subs_all - politic_subs_all
    politic_subs_pure = politic_subs_all - country_subs_all
    
    print(f"Loaded {len(country_subs_pure)} 'Pure' Country Subreddits.")
    print(f"Loaded {len(politic_subs_pure)} 'Pure' Political Subreddits.")

    # --- Step 2: Filter df_posts ---
    print("\nFiltering DataFrame...")
    df_filtered = df_posts.copy()

    # 2a. Convert TIMESTAMP
    try:
        # Only convert if it's not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['TIMESTAMP']):
            df_filtered['TIMESTAMP'] = pd.to_datetime(df_filtered['TIMESTAMP'])
        print("TIMESTAMP column is datetime.")
    except Exception as e:
        print(f"ERROR: Could not convert TIMESTAMP column: {e}")
        return pd.DataFrame() # Fail gracefully if timestamp is bad

    # 2b. Filter by sentiment
    if link_sentiment is not None:
        df_filtered = df_filtered[df_filtered['LINK_SENTIMENT'] == link_sentiment]
        print(f"Rows after LINK_SENTIMENT == {link_sentiment} filter: {len(df_filtered)}")
    
    # 2c. Filter by year
    if year is not None:
        df_filtered = df_filtered[df_filtered['TIMESTAMP'].dt.year == year]
        print(f"Rows after Year == {year} filter: {len(df_filtered)}")

    # --- Step 3: Filter for Interactions (Both Directions) ---
    print("\nFiltering for pure cross-interactions (both directions)...")
    
    condition1 = (
        df_filtered['SOURCE_SUBREDDIT'].isin(country_subs_pure) &
        df_filtered['TARGET_SUBREDDIT'].isin(politic_subs_pure)
    )
    condition2 = (
        df_filtered['SOURCE_SUBREDDIT'].isin(politic_subs_pure) &
        df_filtered['TARGET_SUBREDDIT'].isin(country_subs_pure)
    )
    
    df_country_politic_interactions = df_filtered[condition1 | condition2].copy()
    
    if df_country_politic_interactions.empty:
        print("Found 0 interactions matching all criteria. Returning empty summary.")
        return pd.DataFrame(columns=['left', 'right', 'center_or_other', 'Total_Interactions'])
        
    print(f"Found {len(df_country_politic_interactions)} total interactions matching all criteria.")

    # --- Step 4: Create Mapping Dictionaries ---
    
    # 4a. Country map
    country_map = df_countries.drop_duplicates(subset='subreddit').set_index('subreddit')['predicted_country'].to_dict()
    print(f"Loaded {len(country_map)} country mappings.")

    # 4b. Ideology map
    ideology_map = df_politics.set_index('Matched Subreddit')['Ideology'].to_dict()
    print(f"Loaded {len(ideology_map)} ideology mappings.")
    
    # --- Step 5: Create Temporary Columns for Aggregation ---
    
    # 5a. Unified 'Country' column
    df_country_politic_interactions['Country'] = (
        df_country_politic_interactions['SOURCE_SUBREDDIT'].map(country_map).fillna(
            df_country_politic_interactions['TARGET_SUBREDDIT'].map(country_map)
        )
    )

    # 5b. Unified 'Ideology' column
    df_country_politic_interactions['Ideology'] = (
        df_country_politic_interactions['SOURCE_SUBREDDIT'].map(ideology_map).fillna(
            df_country_politic_interactions['TARGET_SUBREDDIT'].map(ideology_map)
        )
    )
    print("Temporary 'Country' and 'Ideology' columns created.")

    # --- Step 6: Create the Final Summary DataFrame ---
    print("\nAggregating results...")
    
    df_final = df_country_politic_interactions.dropna(subset=['Country', 'Ideology'])
    
    if df_final.empty:
        print("No valid interactions after mapping. Returning empty summary.")
        return pd.DataFrame(columns=['left', 'right', 'center_or_other', 'Total_Interactions'])

    df_ideology_summary = df_final.groupby(['Country', 'Ideology']).size().unstack(level='Ideology', fill_value=0)
    
    # Ensure all three ideology columns exist, even if there's no data
    for col in ['left', 'right', 'center_or_other']:
        if col not in df_ideology_summary.columns:
            df_ideology_summary[col] = 0

    # Add 'Total_Interactions'
    df_ideology_summary['Total_Interactions'] = df_ideology_summary.sum(axis=1)
    
    # Sort
    df_ideology_summary = df_ideology_summary.sort_values('Total_Interactions', ascending=False)
    
    print("\n--- Analysis Complete ---")
    return df_ideology_summary


from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List, Dict, Set


ECON_KEYWORDS: Set[str] = {
    # Economics / macro
    "economy","economic","economics","macro","micro","econometrics",
    "growth","recession","inflation","deflation","stagflation","productivity",
    "gdp","policy","monetary","fiscal","budget","tax","taxation","debt",
    "deficit","surplus","inequality","poverty","development","sustainability",
    # Finance & markets
    "finance","financial","bank","banking","credit","loan","interest",
    "investment","investing","investor","dividend","equity","capital",
    "bond","bonds","securities","market","markets","stock","stocks",
    "index","indices","fund","funds","etf","hedge","portfolio","trading",
    "broker","exchange","nyse","nasdaq","wallstreet","dowjones","dow","sp500",
    "forex","currency","currencies","money","wealth","cryptocurrency",
    "crypto","bitcoin","ethereum","blockchain","token","nft",
    # Commodities / energy / business / trade
    "commodity","commodities","oil","petrol","gas","naturalgas","energy",
    "coal","uranium","electricity","power","renewable","solar","wind",
    "hydrogen","gold","silver","platinum","palladium","copper","iron",
    "steel","aluminium","nickel","zinc","lead","lithium","cobalt","rareearth",
    "grain","wheat","corn","soybean","rice","coffee","sugar","cotton",
    "timber","water","mining","agriculture","agricultural",
    "industry","industries","manufacturing","commerce","business",
    "entrepreneur","entrepreneurship","startup","venture","innovation",
    "supply","demand","trade","export","import","globalization","logistics",
    "transport","shipping",
    # Institutions / policy
    "centralbank","ecb","fed","imf","worldbank","oecd","wto","regulation",
    "governance","public","reform","subsidy","aid","stimulus","bailout",
    "spending","taxpayer","treasury"
}

MACRO_TOPICS: Dict[str, List[str]] = {
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
        "broker","exchange","nyse","nasdaq","wallstreet","dowjones","dow","sp500",
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
    "Institutions / Policies": [
        "centralbank","ecb","fed","imf","worldbank","oecd","wto","regulation",
        "governance","public","reform","subsidy","aid","stimulus","bailout",
        "spending","taxpayer","treasury"
    ]
}

def normalize(s: str) -> str:
    """Lowercase + sostituisce non-alfanumerici con spazio."""
    if not isinstance(s, str):
        return ""
    return "".join(ch.lower() if ch.isalnum() else " " for ch in s).strip()

def normalize_series(ser: pd.Series) -> pd.Series:
    return ser.astype(str).str.lower().str.strip()


_WORD_PATS = [re.compile(rf"\b{re.escape(kw)}\b") for kw in sorted(ECON_KEYWORDS)]

def contains_econ_word(text: str) -> bool:
    """True se la stringa contiene almeno una keyword economica."""
    if not isinstance(text, str) or not text:
        return False
    s = normalize(text)
    for pat in _WORD_PATS:
        if pat.search(s):
            return True
    return any(kw in s for kw in ECON_KEYWORDS)


def classify_topic(name: str) -> str:
    """Ritorna il macro-topic (o 'Other / unclassified')."""
    s = normalize(name)
    for topic, kws in MACRO_TOPICS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", s) or kw in s:
                return topic
    return "Other / unclassified"


def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p)

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def build_economic_links_with_geo_labeled(
    edges: pd.DataFrame,
    geo_df: pd.DataFrame,
    filter_economic: bool = True,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    - Normalizza nomi subreddit
    - (opz) filtra edges 'economici' (almeno una estremità matcha keyword)
    - Seleziona edges con almeno un estremo in geo_df['subreddit']
    - Aggiunge SOURCE_CATEGORY / TARGET_CATEGORY
    - Ritorna DF finale
    """
    req = {"SOURCE_SUBREDDIT","TARGET_SUBREDDIT"}
    missing = req - set(edges.columns)
    if missing:
        raise KeyError(f"Mancano colonne {missing}. Trovate: {edges.columns.tolist()}")

    if "subreddit" not in geo_df.columns:
        raise KeyError("Attesa colonna 'subreddit' in geo_df")

    edges = edges.copy()
    edges["SOURCE_SUBREDDIT"] = normalize_series(edges["SOURCE_SUBREDDIT"])
    edges["TARGET_SUBREDDIT"] = normalize_series(edges["TARGET_SUBREDDIT"])

    geo_set = set(normalize_series(geo_df["subreddit"]))

    if filter_economic:
        mask_econ = (
            edges["SOURCE_SUBREDDIT"].apply(contains_econ_word) |
            edges["TARGET_SUBREDDIT"].apply(contains_econ_word)
        )
        edges = edges[mask_econ]

    mask_geo = (
        edges["SOURCE_SUBREDDIT"].isin(geo_set) |
        edges["TARGET_SUBREDDIT"].isin(geo_set)
    )
    edges = edges[mask_geo]

    if drop_duplicates:
        edges = edges.drop_duplicates(subset=["SOURCE_SUBREDDIT","TARGET_SUBREDDIT"])

    edges["SOURCE_CATEGORY"] = edges["SOURCE_SUBREDDIT"].apply(classify_topic)
    edges["TARGET_CATEGORY"] = edges["TARGET_SUBREDDIT"].apply(classify_topic)
    return edges


