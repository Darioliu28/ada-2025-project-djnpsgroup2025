import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys
from IPython.display import display
import os
import pickle

# Set pandas display option
pd.set_option("display.max_columns", 100)

# Constants
POST_PROPS_COLS = [
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

def load_dataframes(path, title_filename, body_filename):
    """Loads the title and body hyperlink dataframes."""
    try:
        df_title_path = os.path.join(path, title_filename)
        df_body_path = os.path.join(path, body_filename)
        
        df_title = pd.read_csv(df_title_path, sep="\t")
        df_body = pd.read_csv(df_body_path, sep="\t")
        
        print(f"Successfully loaded '{title_filename}' and '{body_filename}'.")
        return df_title, df_body
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please make sure the data files are in the correct path.")
        return None, None
    except Exception as e:
        print(f"Error loading dataframes: {e}", file=sys.stderr)
        return None, None

def split_properties_columns(df_to_split, col_name_to_split="PROPERTIES", new_col_names=POST_PROPS_COLS):
    """Splits the 'PROPERTIES' column into multiple feature columns."""
    if col_name_to_split in df_to_split.columns:
        try:
            # Split the column
            split_data = df_to_split[col_name_to_split].str.split(",", expand=True)
            
            # Ensure the number of split columns matches the new column names
            if split_data.shape[1] == len(new_col_names):
                df_to_split[new_col_names] = split_data.astype(float)
                df_to_split.drop(columns=[col_name_to_split], inplace=True)
                print(f"Successfully split '{col_name_to_split}' column.")
            else:
                print(f"Warning: Mismatch in split columns ({split_data.shape[1]}) and new column names ({len(new_col_names)}). Skipping split.")
        except Exception as e:
            print(f"Error splitting '{col_name_to_split}' column: {e}", file=sys.stderr)
    else:
        print(f"'{col_name_to_split}' column not found in DataFrame. Skipping split.")


def build_weighted_graph_from_df(df):
    """
    Counts hyperlink frequency between subreddits,
    calculates weights, and builds a directed graph from a DataFrame.
    
    Weight = 100 / (number of hyperlinks from x to y)
    """
    if 'SOURCE_SUBREDDIT' not in df.columns or 'TARGET_SUBREDDIT' not in df.columns:
        print("Error: DataFrame must contain 'SOURCE_SUBREDDIT' and 'TARGET_SUBREDDIT' columns.", file=sys.stderr)
        return None, None

    print("Calculating edge frequencies...")
    edge_counts = df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).size()
    edge_counts_df = edge_counts.reset_index(name='hyperlink_count')

    edge_counts_df['weight'] = 100.0 / edge_counts_df['hyperlink_count']

    print("Building directed graph...")
    G = nx.DiGraph()
    
    all_nodes = set(df['SOURCE_SUBREDDIT']).union(set(df['TARGET_SUBREDDIT']))
    G.add_nodes_from(all_nodes)
    
    for _, row in edge_counts_df.iterrows():
        G.add_edge(row['SOURCE_SUBREDDIT'], 
                   row['TARGET_SUBREDDIT'], 
                   weight=row['weight'])
    
    print(f"Graph built successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    source_subreddits = set(df['SOURCE_SUBREDDIT'])
    
    return G, source_subreddits

def find_all_shortest_paths(graph, source_nodes):
    """
    Runs Dijkstra's algorithm from each source node to all other nodes.
    Returns a dictionary: {source_node: {target_node: shortest_path_length, ...}, ...}
    """
    print(f"\nFinding shortest paths from {len(source_nodes)} source subreddits...")
    all_paths = {}
    total_sources = len(source_nodes)

    for i, source in enumerate(source_nodes):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  ({i+1}/{total_sources}) Calculating paths from: {source}...")
        
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, source, weight='weight')
            all_paths[source] = lengths
        except nx.NodeNotFound:
            print(f"  Warning: Node '{source}' not in graph. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"  Error calculating paths from '{source}': {e}", file=sys.stderr)
            
    return all_paths

def plot_network(graph, draw_edge_labels=False):
    """Plots the directed graph using matplotlib."""
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty, skipping plot.")
        return

    print("\nAttempting to plot the network graph...")
    print("This may take a moment and might be unreadable if the graph is large.")
    
    plt.figure(figsize=(25, 25))
    
    try:
        pos = nx.spring_layout(graph, k=0.6, iterations=50)
    except Exception as e:
        print(f"  Could not compute spring layout, falling back to random layout. Error: {e}", file=sys.stderr)
        pos = nx.random_layout(graph)

    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        node_size=80,
        font_size=6,
        font_color='black',
        node_color='skyblue',
        edge_color='gray',
        alpha=0.8,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10
    )
    
    if draw_edge_labels:
        try:
            edge_labels = {(u, v): f"{d['weight']:.1f}" 
                           for u, v, d in graph.edges(data=True) if 'weight' in d}
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                font_size=5,
                font_color='red'
            )
        except Exception as e:
            print(f"  Warning: Could not draw edge labels. {e}", file=sys.stderr)
    
    plt.title("Network Visualization", size=20)
    plt.axis('off')
    
    try:
        plt.show()
    except Exception as e:
        print(f"Error: Could not display plot. {e}", file=sys.stderr)
        print("Continuing without plot...")

def remove_saved_graph(graph_filename):
    """Removes the specified .gpickle file if it exists."""
    if os.path.exists(graph_filename):
        try:
            os.remove(graph_filename)
            print(f"Successfully removed saved graph: '{graph_filename}'")
            print("The graph will be rebuilt from CSV on the next run.")
        except Exception as e:
            print(f"Error removing file '{graph_filename}': {e}", file=sys.stderr)
    else:
        print(f"No saved graph file to remove ('{graph_filename}').")

def load_or_build_graph_data(df_hyperlinks, graph_filename):
    """
    Loads a pre-computed graph and shortest paths from a .gpickle file
    or builds them from the hyperlink DataFrame if the file doesn't exist.
    Returns: (graph, source_subreddits, all_shortest_paths)
    """
    graph, source_subreddits, all_shortest_paths = None, None, None

    if os.path.exists(graph_filename):
        try:
            print(f"Loading pre-computed graph and paths from '{graph_filename}'...")
            with open(graph_filename, 'rb') as f:
                (graph, source_subreddits, all_shortest_paths) = pickle.load(f)
            print(f"Graph and shortest paths loaded successfully.")
            print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error loading data from '{graph_filename}': {e}", file=sys.stderr)
            print("Will attempt to rebuild from DataFrame.")
            graph, source_subreddits, all_shortest_paths = None, None, None

    if not graph or not source_subreddits or not all_shortest_paths:
        print(f"No pre-computed data found or loading failed. Building from DataFrame...")
        
        graph, source_subreddits = build_weighted_graph_from_df(df_hyperlinks)
        
        if graph and source_subreddits:
            all_shortest_paths = find_all_shortest_paths(graph, source_subreddits)
            print("Shortest path analysis complete.")
            
            try:
                print(f"Saving new graph and paths to '{graph_filename}'...")
                with open(graph_filename, 'wb') as f:
                    pickle.dump((graph, source_subreddits, all_shortest_paths), f, pickle.HIGHEST_PROTOCOL)
                print("Graph and shortest paths saved successfully.")
            except Exception as e:
                print(f"Error saving data to '{graph_filename}': {e}", file=sys.stderr)
        else:
            print("Graph building failed. Cannot proceed.", file=sys.stderr)
            return None, None, None

    if all_shortest_paths:
        print("\nShortest path data is ready.")
    else:
        print("\nCould not load or compute shortest paths.", file=sys.stderr)
    
    return graph, source_subreddits, all_shortest_paths

def plot_country_graph(graph, country_csv_path):
    """Builds and plots an aggregated country-level graph."""
    if not graph:
        print("No graph provided. Cannot plot country graph.")
        return

    print(f"\nLoading country subreddit list from '{country_csv_path}'...")
    try:
        df_approved_countries = pd.read_csv(country_csv_path)
        
        if 'subreddit' not in df_approved_countries.columns or 'predicted_country' not in df_approved_countries.columns:
            print(f"Error: Country CSV '{country_csv_path}' must contain 'subreddit' and 'predicted_country' columns.", file=sys.stderr)
            print("Skipping plot.")
            return

        print("Building aggregated country-level graph...")
        
        sub_to_country = pd.Series(df_approved_countries.predicted_country.values, 
                                   index=df_approved_countries.subreddit).to_dict()

        country_edge_links = {}

        for u, v, data in graph.edges(data=True):
            country_u = sub_to_country.get(u)
            country_v = sub_to_country.get(v)
            
            if country_u and country_v:
                try:
                    original_hyperlink_count = 100.0 / data['weight']
                except ZeroDivisionError:
                    continue
                
                edge_pair = (country_u, country_v)
                country_edge_links[edge_pair] = country_edge_links.get(edge_pair, 0) + original_hyperlink_count

        if not country_edge_links:
            print("No links found between subreddits that are both in the country list. Skipping plot.")
        else:
            G_countries = nx.DiGraph()
            all_country_nodes = set([pair[0] for pair in country_edge_links.keys()]).union(
                                set([pair[1] for pair in country_edge_links.keys()]))
            G_countries.add_nodes_from(all_country_nodes)
            
            for (country_u, country_v), total_links in country_edge_links.items():
                weight = 100.0 / total_links
                G_countries.add_edge(country_u, country_v, weight=weight)
                
            print(f"Built country graph with {G_countries.number_of_nodes()} nodes (countries) and {G_countries.number_of_edges()} edges.")
            
            # Call the plotting function
            plot_network(G_countries, draw_edge_labels=True)
            
    except FileNotFoundError:
        print(f"Error: The country CSV file '{country_csv_path}' was not found.", file=sys.stderr)
        print("Cannot plot country graph.")
    except Exception as e:
        print(f"Error loading or processing country CSV: {e}", file=sys.stderr)
        print("Skipping plot.")

def calculate_country_shortest_paths(all_shortest_paths, country_csv_path, output_csv_path):
    """
    Calculates the minimum shortest paths between countries based on subreddit paths
    and saves the results to a CSV file.
    """
    if not all_shortest_paths:
        print("No shortest paths were provided. Cannot calculate country paths.")
        return

    print("\nLoading country subreddit list...")
    try:
        df_approved_countries = pd.read_csv(country_csv_path)
        sub_to_country = pd.Series(df_approved_countries.predicted_country.values, 
                                   index=df_approved_countries.subreddit).to_dict()
        print(f"Loaded {len(sub_to_country)} subreddit-to-country mappings.")
    except FileNotFoundError:
        print(f"Error: Country CSV file '{country_csv_path}' was not found.", file=sys.stderr)
        print("Cannot filter for country paths. Aborting.")
        return
    except Exception as e:
        print(f"Error loading country file: {e}. Cannot filter for country paths.", file=sys.stderr)
        return

    print(f"\nFinding minimum shortest paths between countries...")
    
    country_shortest_paths = {} 
    total_sources = len(all_shortest_paths)
    
    for i, (source_sub, paths) in enumerate(all_shortest_paths.items()):
        
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  ... processing source subreddit {i+1}/{total_sources} ('{source_sub}')")

        country_u = sub_to_country.get(source_sub)
        
        if not country_u:
            continue
            
        for target_sub, length in paths.items():
            if source_sub == target_sub:
                continue
                
            country_v = sub_to_country.get(target_sub)
            
            if country_v:
                pair = (country_u, country_v)
                current_min_length = country_shortest_paths.get(pair, float('inf'))
                if length < current_min_length:
                    country_shortest_paths[pair] = length

    print("...Processing complete.")

    if country_shortest_paths:
        print(f"\nConverting {len(country_shortest_paths)} country-to-country paths to DataFrame...")
        
        path_list = [(source, target, length) for (source, target), length in country_shortest_paths.items()]
        
        df_countries = pd.DataFrame(path_list, columns=['source_country', 'target_country', 'shortest_path_length'])
        df_countries.sort_values(by='shortest_path_length', inplace=True)
        
        try:
            df_countries.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved all *country* shortest paths to '{output_csv_path}'")
            print("\n---\nTo inspect the file, run this in a new cell:")
            print(f"df_country_paths = pd.read_csv('{output_csv_path}')")
            print("display(df_country_paths.head())")
        except Exception as e:
            print(f"\nError saving country DataFrame to CSV: {e}", file=sys.stderr)
    
    else:
        print("\nNo shortest paths found between any two mapped countries.")

# --- Constants for network analysis ---
# Path for the saved/loaded pre-computed graph
PATH = "data/"
FILENAME_TITLES = 'soc-redditHyperlinks-title.tsv'
FILENAME_BODIES = 'soc-redditHyperlinks-body.tsv'
GRAPH_FILENAME = os.path.join(PATH, "subreddit_graph.gpickle")

# Path for the country mapping file
COUNTRY_CSV_FILENAME = os.path.join(PATH, "country_matches_map_exp.csv")

# Path for the final output file
COUNTRY_PATHS_OUTPUT = os.path.join(PATH, "country_shortest_paths_output.csv")

df_title, df_body = load_dataframes(PATH, FILENAME_TITLES, FILENAME_BODIES)

if df_title is not None and df_body is not None:
    print("--- Titles DataFrame (Raw) ---")
    display(df_title.head())
    print("\n--- Bodies DataFrame (Raw) ---")
    display(df_body.head())

#na.remove_saved_graph(GRAPH_FILENAME)

graph, source_subreddits, all_shortest_paths = None, None, None
if df_title is not None:
    graph, source_subreddits, all_shortest_paths = load_or_build_graph_data(df_title, GRAPH_FILENAME)
else:
    print("df_title not loaded, cannot build graph.")

if graph:
    plot_country_graph(graph, COUNTRY_CSV_FILENAME)
else:
    print("Graph not loaded, cannot plot country graph.")

# --- Calculate and save country-to-country shortest paths ---

if all_shortest_paths:
    calculate_country_shortest_paths(all_shortest_paths, COUNTRY_CSV_FILENAME, COUNTRY_PATHS_OUTPUT)
else:
    print("Shortest paths not available, cannot calculate country paths.")