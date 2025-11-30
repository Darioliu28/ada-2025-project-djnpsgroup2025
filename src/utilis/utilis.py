from __future__ import annotations
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
# === CLUSTERS WITH EMBEDDING ANALYSIS ===

def prepare_embeddings_for_clustering(df_emb):
    """
    Separates subreddit labels from embedding features and scales the features.
    
    Args:
        df_emb (pd.DataFrame): The raw embeddings dataframe. Assumes column 0 
                               is 'subreddit' and all others are features.
    
    Returns:
        tuple: (scaled_features, subreddit_labels)
    """
    # Separates    
    subreddit_labels = df_emb['subreddit'].values

    feature_cols = [col for col in df_emb.columns if col != 'subreddit']
    features = df_emb[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"Data prepared: {len(subreddit_labels)} items, {scaled_features.shape[1]} features.")
    return scaled_features, subreddit_labels

def calculate_kmeans_elbow_wide(scaled_data, k_values_list, n_samples=5000):
    """
    Calculates the K-Means "inertia" for a specific list of k values.
    
    Uses a random sample of the data for speed.
    
    Args:
        scaled_data (np.array): The scaled feature data.
        k_values_list (list): A list of integers to test (e.g., [20, 40, 60]).
        n_samples (int): Number of samples to use for this calculation.

    Returns:
        pd.DataFrame: A DataFrame with 'k' and 'inertia'.
    """

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
    print(f"Running K-Means clustering with k={n_clusters} on {len(scaled_data)} items...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    
    all_cluster_labels = kmeans.fit_predict(scaled_data)

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

    tsne_df = pd.DataFrame({
        'subreddit': label_sample,
        'cluster': cluster_sample,
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1]
    })
    
    tsne_df_filtered = tsne_df[tsne_df['cluster'].isin(valid_clusters_set)].copy()
    tsne_df_filtered['cluster'] = tsne_df_filtered['cluster'].astype(str)
        
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
    df = pd.DataFrame({
        'subreddit': subreddit_labels,
        'cluster': all_cluster_labels
    })
    
    cluster_samples = {}
    
    for cluster_id, group in df.groupby('cluster'):
        
        # Get a random sample, or all of them if the group is too small
        if len(group) > n_samples:
            sample = group.sample(n_samples, random_state=42)
        else:
            sample = group
            
        cluster_samples[cluster_id] = sample['subreddit'].tolist()
        
    return cluster_samples


# === COUNTRY INTERACTION ANALYSIS

def calculate_normalized_interactions(country_post_counts, df_post_between_countries):
    total_posts_map = country_post_counts.to_dict()

    raw_interactions = (
        df_post_between_countries.groupby(['source_country', 'target_country'])
        .size()
        .reset_index(name='n_interactions')
    )
    raw_interactions = raw_interactions.query("source_country != target_country").copy()
    raw_interactions['sorted_pair'] = raw_interactions.apply(
        lambda row: tuple(sorted([row['source_country'], row['target_country']])), axis=1
    )

    df_undirected = (
        raw_interactions.groupby('sorted_pair')['n_interactions']
        .sum()
        .reset_index()
    )

    df_undirected[['Country_A', 'Country_B']] = pd.DataFrame(
        df_undirected['sorted_pair'].tolist(), index=df_undirected.index
    )

    df_undirected['total_posts_A'] = df_undirected['Country_A'].map(total_posts_map)
    df_undirected['total_posts_B'] = df_undirected['Country_B'].map(total_posts_map)

    df_undirected['norm_log'] = (
        df_undirected['n_interactions'] / 
        np.log1p(df_undirected['total_posts_A'] + df_undirected['total_posts_B'])
    )

    df_final = df_undirected.sort_values(by='norm_log', ascending=False)

    return df_final


# === EMBEDDING-FACTION ANALYSIS ===

def find_strict_subreddits(df_countries, df_embeddings):
    """
    Identifies "strict" subreddits which are more similar to their own country's
    subreddits than to any other country's subreddits based on embeddings.
    """
    df_approved_emb = df_countries.merge(df_embeddings, on="subreddit", how="inner")
    emb_cols = [c for c in df_embeddings.columns if c != "subreddit"]
    
    country_to_embeddings = {}
    for country, group in df_approved_emb.groupby("country"):
        country_to_embeddings[country] = group[emb_cols].values

    strict_subs = []
    for _, row in df_approved_emb.iterrows():
        subreddit = row["subreddit"]
        country = row["country"]
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
    Finds the most similar (highest cosine similarity) subreddit 
    from a different country for each strict subreddit.
    """
    strict_subs_set = set(df_strict_approved["subreddit"])
    df_emb_strict = df_embeddings[df_embeddings["subreddit"].isin(strict_subs_set)].reset_index(drop=True)
    
    embedding_cols = [c for c in df_emb_strict.columns if c != "subreddit"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df_emb_strict[embedding_cols])
    
    subreddit_names = df_emb_strict["subreddit"].tolist()
    sub_to_country = dict(zip(df_strict_approved["subreddit"], df_strict_approved["country"]))
    
    sim_matrix = cosine_similarity(X, X)
    
    most_similar_subreddits = []
    for i, sub in enumerate(subreddit_names):
        
        similarities = sim_matrix[i].copy()
        
        # Mask out subreddits from the same country
        for j, other_sub in enumerate(subreddit_names):
            if sub_to_country.get(other_sub) == sub_to_country.get(sub) or i == j:
                similarities[j] = -np.inf 
        
        closest_idx = similarities.argmax()
        closest_sub = subreddit_names[closest_idx]
        
        most_similar_subreddits.append({
            "subreddit": sub,
            "country": sub_to_country.get(sub),
            "most_similar_subreddit": closest_sub,
            "most_similar_country": sub_to_country.get(closest_sub),
            "similarity_score": similarities[closest_idx]
        })
        
    return pd.DataFrame(most_similar_subreddits)


# === NETWORK-BASED FACTION ANALYSIS WITH POSITIVE POSTS ===

def _map_countries(df_posts, df_countries):
    """Helper to map source/target subreddits to countries."""
    sub_to_country = df_countries.set_index("subreddit")["country"].to_dict()
    df_posts['source_country'] = df_posts['SOURCE_SUBREDDIT'].map(sub_to_country)
    df_posts['target_country'] = df_posts['TARGET_SUBREDDIT'].map(sub_to_country)
    return df_posts.dropna(subset=["source_country", "target_country"])

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

def detect_normalized_factions(df_post_between_countries):
    """
    Detects factions based on normalized positive interaction weights.
    Weight = N_posts(A,B) / (TotalPosts(A) * TotalPosts(B))
    """
    df_positive = df_post_between_countries[df_post_between_countries["LINK_SENTIMENT"] == 1].copy()
    
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
                                         (country_links["source_total"] * country_links["target_total"])

    G_norm = build_interaction_graph(country_links, weight_col="normalized_weight")
    return detect_factions(G_norm)


# === TEMPORAL & ACTIVITY ANALYSIS ===

def map_countries_to_posts(df_posts, df_countries, period):
    """Helper to map source/target subreddits to countries."""
    df_links_with_countries = _map_countries(df_posts.copy(), df_countries)
    df_links_with_countries["TIMESTAMP"] = pd.to_datetime(df_links_with_countries["TIMESTAMP"], errors='coerce')
    df_links_with_countries["year"] = df_links_with_countries["TIMESTAMP"].dt.to_period(period)
    return df_links_with_countries.dropna(subset=['year'])

def analyze_source_normalized_factions_over_time(df_post_between_countries):
    """
    Calculates factions for each quarter, normalizing by source country post count.
    """
    timely_summary = []
    
    for period, group in df_post_between_countries.groupby("year"):
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
        summary["year"] = str(period)
        timely_summary.append(summary)
        
    return pd.concat(timely_summary).reset_index(drop=True)

def find_stable_pairs(quarterly_factions_summary_df):
    """Finds pairs of countries that frequently appear in the same faction."""
    pair_counter = Counter()
    total_quarters = quarterly_factions_summary_df["year"].nunique()
    
    for _, quarter_group in quarterly_factions_summary_df.groupby("year"):
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


# === RECIPROCITY ANALYSIS ===

def response_global(df_country):
    
    # Sort the entire DataFrame by TIMESTAMP *once* at the beginning.
    # This is fundamental to our logic.
    df_country = df_country.sort_values(by='TIMESTAMP')

    # Create a 'pair_key' using frozenset to group (A, B) and (B, A) together
    df_country['pair_key'] = df_country.apply(
        lambda row: frozenset([row['source_country'], row['target_country']]),
        axis=1
    )
    # Filter out pairs that only have one interaction
    pair_counts = df_country['pair_key'].value_counts()
    valid_pairs = pair_counts[pair_counts > 1].index
    df_analysis = df_country[df_country['pair_key'].isin(valid_pairs)]

    print(f"DataFrame ready. Analyzing {len(valid_pairs)} pairs.")

    ## --- 2. Global Totals Calculation (Deterministic Logic) ---

    window_days = 7
    total_initiators_global = 0
    total_responses_global = 0

    # Iterate over each unique pair with a progress bar
    for pair in tqdm(valid_pairs, desc="Analyzing pairs"):
        
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
        country_A = first_interaction['source_country']
        country_B = first_interaction['target_country']
        
        # --- Get all interactions for these defined roles ---
        
        # Get all A -> B interactions (initiations)
        df_A_to_B = df_pair[
            (df_pair['source_country'] == country_A) & 
            (df_pair['target_country'] == country_B)
        ]
        # Get all B -> A interactions (responses)
        df_B_to_A = df_pair[
            (df_pair['source_country'] == country_B) & 
            (df_pair['target_country'] == country_A)
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
        (df_combined['source_country'] == df_combined['target_country']) &
        (df_combined['source_country'].notna()) &
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
        print("No initiator-response interactions found.")

    print("\n--- Global INTRA-COUNTRY Reciprocity Analysis ---")
    print(f"Total 'initiator' interactions (SubA -> SubB) analyzed: {total_initiators_global}")
    print(f"Total responses (SubB -> SubA) within 7 days: {total_responses_global:.0f}")
    print(f"GLOBAL Conditional Probability P(SubB->SubA | SubA->SubB in 7d): {cond_prob:.2%}")


# === RESPONSE SIMILARITY ANALYSIS ===

def find_reciprocity_pairs_and_similarity(df_interactions, features_list, df_countries):
    """
    Finds A->B => B->A pairs (all sentiments) within 7 days 
    and calculates cosine similarity using only the 'features_list'.
    """
    WINDOW_DAYS = 7
    country_subs_list = df_countries['subreddit'].tolist()
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
                    vector_trigger = df_interactions.loc[[trigger_id], features_list].values
                    vector_response = df_interactions.loc[[first_response_id], features_list].values
                        
                    similarity = cosine_similarity(vector_trigger, vector_response)[0][0]
                    similarity_scores.append(similarity)
                except KeyError:
                    pass 
                        
    return similarity_scores

def response_similarity(df_combined, df_countries):
    # --- 1. Style Feature Definition ---
    style_features_list = [
        # Tone/Sentiment Measures 
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

    # Verify that the chosen style features exist
    for col in style_features_list:
        if col not in df_combined.columns:
            raise ValueError(f"Style feature '{col}' was not found in df_combined.")

    other_needed_cols = ['POST_ID', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT',
                        'TIMESTAMP', 'LINK_SENTIMENT']

    df_analysis = df_combined[other_needed_cols + style_features_list].copy()
    df_analysis = df_analysis.set_index('POST_ID')

    print("Analysis DataFrame ready.")

    WINDOW_DAYS = 7

    # --- 3. Main Analysis Execution (Reciprocity) ---
    reciprocity_similarities = find_reciprocity_pairs_and_similarity(
        df_analysis, 
        style_features_list,  
        df_countries
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
            # T-test 
            t_stat, p_value = ttest_ind(sim_series_reciprocal, sim_series_baseline, 
                                        equal_var=False, alternative='greater')
            print(f"\n--- T-Test (Reciprocal > Random) ---")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")

            # Mann-Whitney U 
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
        
        # Calculate and plot the means
        mean_reciprocal = sim_series_reciprocal.mean()
        mean_baseline = sim_series_baseline.mean()
        median_reciprocal = sim_series_reciprocal.median()
        median_baseline = sim_series_baseline.median()
        
        plt.axvline(mean_reciprocal, color=sns.color_palette()[0], linestyle='--', 
                    label=f'Reciprocal Mean: {mean_reciprocal:.2f}')
        plt.axvline(mean_baseline, color=sns.color_palette()[1], linestyle=':', 
                    label=f'Random Mean: {mean_baseline:.2f}')
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