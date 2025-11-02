import pandas as pd
import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def response_global(df_csv):
    df_country = pd.read_csv(df_csv)
    
    ## --- 1. DataFrame Preparation ---
    df_clean = df_country.dropna(subset=['SOURCE_COUNTRY', 'TARGET_COUNTRY'])
    df_clean['TIMESTAMP'] = pd.to_datetime(df_clean['TIMESTAMP'])
    
    # Ordina l'intero DataFrame per TIMESTAMP *una sola volta* all'inizio.
    # Questo è fondamentale per la nostra logica.
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

    print(f"DataFrame ready. Analyzing {len(valid_pairs)} pairs.")

    ## --- 2. Global Totals Calculation (Deterministic Logic) ---

    window_days = 7
    total_initiators_global = 0
    total_responses_global = 0

    # Itera su ogni coppia unica con una barra di avanzamento
    for pair in tqdm(valid_pairs, desc="Analyzing pairs"):
        
        # Skip A->A self-interactions (which have len < 2)
        if len(pair) < 2:
            continue
            
        # Ottieni tutte le interazioni per questa coppia
        # Saranno GIÀ ordinate per TIMESTAMP grazie all'ordinamento iniziale
        df_pair = df_analysis[df_analysis['pair_key'] == pair]

        # --- Trova il "Vero" Iniziatore ---
        # Prendi la primissima interazione in assoluto per questa coppia
        first_interaction = df_pair.iloc[0]
        
        # Definisci i ruoli in base a quella prima interazione
        C1_initiator = first_interaction['SOURCE_COUNTRY']
        C2_responder = first_interaction['TARGET_COUNTRY']
        
        # --- Ottieni tutte le interazioni per questi ruoli definiti ---
        
        # Get all C1 -> C2 interactions (inizi)
        df_C1_to_C2 = df_pair[
            (df_pair['SOURCE_COUNTRY'] == C1_initiator) & 
            (df_pair['TARGET_COUNTRY'] == C2_responder)
        ]
        # Get all C2 -> C1 interactions (risposte)
        df_C2_to_C1 = df_pair[
            (df_pair['SOURCE_COUNTRY'] == C2_responder) & 
            (df_pair['TARGET_COUNTRY'] == C1_initiator)
        ]

        # Se il "risponditore" (C2) non ha mai postato, 
        # non possiamo misurare la reciprocità per questa coppia
        if df_C1_to_C2.empty or df_C2_to_C1.empty:
            continue

        # --- Calculate conditional probability ---
        df_C1_to_C2 = df_C1_to_C2.copy()
        df_C2_to_C1 = df_C2_to_C1.copy()
        # Copia i timestamp in nuove colonne per preservarli dopo il merge
        df_C1_to_C2['TIMESTAMP_A'] = df_C1_to_C2['TIMESTAMP']
        df_C2_to_C1['TIMESTAMP_B'] = df_C2_to_C1['TIMESTAMP']

        # Trova la prima risposta (C2->C1) avvenuta *dopo* ogni inizio (C1->C2)
        merged = pd.merge_asof(
            df_C1_to_C2,  # Già ordinato per TIMESTAMP
            df_C2_to_C1,  # Già ordinato per TIMESTAMP
            on='TIMESTAMP',          # La colonna su cui unire
            direction='forward',   # Trova la prima risposta *dopo* l'inizio
            suffixes=('_A', '_B')
        )
        
        # Calcola il delta temporale
        merged['response_time'] = merged['TIMESTAMP_B'] - merged['TIMESTAMP_A']
        
        # Controlla se la risposta è avvenuta nella finestra di 7 giorni
        merged['responded_within_7_days'] = (
            merged['response_time'] <= pd.Timedelta(days=window_days)
        )

        # --- Update global counters ---
        total_initiators_global += len(df_C1_to_C2)
        total_responses_global += merged['responded_within_7_days'].sum()
        
    ## --- 3. Global Results Analysis ---

    # Calcola la probabilità finale, gestendo il caso di divisione per zero
    if total_initiators_global == 0:
        print("\nATTENZIONE: Nessuna interazione di 'inizio' trovata. Impossibile calcolare la probabilità.")
        prob_globale_condizionata = 0
    else:
        prob_globale_condizionata = total_responses_global / total_initiators_global

    # Stampa il risultato finale
    print("\n--- Global Reciprocity Analysis ---")
    print(f"Total 'initiator' interactions (A->B) analyzed: {total_initiators_global}")
    print(f"Total responses (B->A) within 7 days: {total_responses_global:.0f}")
    print(f"GLOBAL Conditional Probability P(B->A | A->B in 7d): {prob_globale_condizionata:.2%}")

def response_intra_country(df_csv):
    df_combined = pd.read_csv(df_csv)
    
    # 1. Filter for INTRA-COUNTRY interactions
    df_intra = df_combined[
        (df_combined['SOURCE_COUNTRY'] == df_combined['TARGET_COUNTRY']) &
        (df_combined['SOURCE_COUNTRY'].notna()) &
        (df_combined['SOURCE_SUBREDDIT'] != df_combined['TARGET_SUBREDDIT'])
    ].copy()

    # 2. Ensure TIMESTAMP is datetime and SORT
    df_intra['TIMESTAMP'] = pd.to_datetime(df_intra['TIMESTAMP'])
    # Ordinamento UNICO all'inizio: questo è fondamentale
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

    # Itera su ogni coppia UNICA di subreddit
    for pair in tqdm(valid_pairs, desc="Analyzing subreddit pairs"):
        
        # --- LOGICA CORRETTA PER L'INIZIATORE ---
        
        # Ottieni tutte le interazioni della coppia (già ordinate per TIMESTAMP)
        df_pair = df_analysis[df_analysis['pair_key'] == pair]
        
        # Trova la primissima interazione in assoluto
        first_interaction = df_pair.iloc[0]
        
        # Definisci i ruoli in base a quella prima interazione
        Sub_C1_Initiator = first_interaction['SOURCE_SUBREDDIT']
        Sub_C2_Responder = first_interaction['TARGET_SUBREDDIT']
        
        # --- Fine logica corretta ---

        # Definisci "Inizio" (C1 -> C2) e "Risposta" (C2 -> C1)
        df_C1_to_C2 = df_pair[
            (df_pair['SOURCE_SUBREDDIT'] == Sub_C1_Initiator) & 
            (df_pair['TARGET_SUBREDDIT'] == Sub_C2_Responder)
        ]
        df_C2_to_C1 = df_pair[
            (df_pair['SOURCE_SUBREDDIT'] == Sub_C2_Responder) & 
            (df_pair['TARGET_SUBREDDIT'] == Sub_C1_Initiator)
        ]

        # Se il "risponditore" C2 non ha mai postato, salta
        if df_C1_to_C2.empty or df_C2_to_C1.empty:
            continue

        # --- Calculate conditional probability ---
        df_C1_to_C2 = df_C1_to_C2.copy()
        df_C2_to_C1 = df_C2_to_C1.copy()
        df_C1_to_C2['TIMESTAMP_A'] = df_C1_to_C2['TIMESTAMP']
        df_C2_to_C1['TIMESTAMP_B'] = df_C2_to_C1['TIMESTAMP']

        merged = pd.merge_asof(
            df_C1_to_C2, # Già ordinato
            df_C2_to_C1, # Già ordinato
            on='TIMESTAMP',
            direction='forward',
            suffixes=('_A', '_B')
        )
        
        merged['response_time'] = merged['TIMESTAMP_B'] - merged['TIMESTAMP_A']
        merged['responded_within_7_days'] = (
            merged['response_time'] <= pd.Timedelta(days=window_days)
        )

        # --- Update global counters ---
        total_initiators_global += len(df_C1_to_C2)
        total_responses_global += merged['responded_within_7_days'].sum()
        
    ## --- 4. Global Results Analysis ---
    prob_globale_condizionata = 0.0
    if total_initiators_global > 0:
        prob_globale_condizionata = total_responses_global / total_initiators_global
    else:
        print("No initiator-response interactions found.")

    print("\n--- Global INTRA-COUNTRY Reciprocity Analysis ---")
    print(f"Total 'initiator' interactions (A -> B) analyzed: {total_initiators_global}")
    print(f"Total responses (B --> A) within 7 days: {total_responses_global:.0f}")
    print(f"GLOBAL Conditional Probability P(SB --> A | A -> B in 7d): {prob_globale_condizionata:.2%}")

def analysis_sports(df_country_sports, final_folder):
    # Create a new, clean DataFrame for this specific analysis
    df_analysis = pd.read_csv(df_country_sports)

    # Create one 'Country' column
    # It takes the value from 'SOURCE_COUNTRY'. If that is NaN,
    # it fills it with the value from 'TARGET_COUNTRY'.
    df_analysis['Country'] = df_analysis['SOURCE_COUNTRY'].fillna(df_analysis['TARGET_COUNTRY'])

    # Create one 'Sport' column
    # It uses the same logic for the sport columns.
    df_analysis['Sport'] = df_analysis['SPORT_SOURCE'].fillna(df_analysis['SPORT_TARGET'])

    # Now we have a clean DataFrame with 'Country' and 'Sport' columns
    print("Consolidated 'Country' and 'Sport' columns.")

    # --- 3. Aggregate Interactions ---
    # Now we can perform the simple groupby you wanted
    # We group by the new clean columns and count the occurrences
    agg_interactions = df_analysis.groupby(['Country', 'Sport']).size().reset_index(name='total_interactions')


    # --- 4. Find the Top Sport for Each Country ---
    # This is the final step you asked for

    # Find the index (row number) of the maximum interaction count *within* each country group
    idx_of_max_per_group = agg_interactions.groupby('Country')['total_interactions'].idxmax()

    # Select only those rows using .loc[]
    df_top_sport_per_country = agg_interactions.loc[idx_of_max_per_group]

    # Sort the final list by interaction count for readability
    df_top_sport_per_country = df_top_sport_per_country.sort_values('total_interactions', ascending=False)

    # --- 5. Display Results ---
    print("\n--- Top Sport per Country (Ranked by Interaction Count) ---")
    print(df_top_sport_per_country)

    # --- Save to CSV ---
    df_top_sport_per_country.to_csv(final_folder+'top_sport_per_country.csv', index=False)

def response_similarity(df_csv, matches_csv):
    # --- 1. Style Feature Definition ---
    df_combined = pd.read_csv(df_csv)
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


    # --- 3. Reciprocity Analysis Function ---
    WINDOW_DAYS = 7

    def find_reciprocity_pairs_and_similarity(df_interactions, features_list):
        """
        Finds A->B => B->A pairs (all sentiments) within 7 days 
        and calculates cosine similarity using only the 'features_list'.
        """
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

        print("Indexing all response events...")
        response_lookup = {}
        for (source_country, target_sub), group in df_responses.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']):
            response_lookup[(source_country, target_sub)] = group[['TIMESTAMP']].reset_index() 

        similarity_scores = []
        
        print(f"Analyzing {len(df_triggers)} 'trigger' posts (all sentiments)...")
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

    # --- 4. Main Analysis Execution (Reciprocity) ---
    print("--- Starting Reciprocity Analysis (Test Group) ---")
    reciprocity_similarities = find_reciprocity_pairs_and_similarity(
        df_analysis, 
        style_features_list  # <-- We pass the style list
    )

    # --- 5. Baseline Analysis Execution (Random Control) ---
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

    # --- 6. Statistical Comparison and Visualization ---
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

df_csv = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/df_country.csv")
df_cs = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/df_country_sport.csv")
final_f = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/")
matches_country_csv = str("/Users/noemiortona/Documents/MTE/ADA/ada-2025-project-djnpsgroup2025/data/country_matches_map.csv")
response_global(df_csv)
response_intra_country(df_csv)
analysis_sports(df_cs, final_f)
response_similarity(df_csv, matches_country_csv)