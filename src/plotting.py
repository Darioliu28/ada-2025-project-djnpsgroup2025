# plotting.py
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np 

# Set default renderer for notebooks
pio.renderers.default = "vscode" 

# -- Functions for cluster with embedding analysis --

def plot_kmeans_elbow(elbow_df):
    """
    Plots the results of the K-Means elbow analysis using Matplotlib.
    
    Args:
        elbow_df (pd.DataFrame): DataFrame from calculate_kmeans_elbow.
    """
    print("Plotting K-Means elbow graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(elbow_df['k'], elbow_df['inertia'], marker='o')
    plt.title('K-Means Elbow Plot')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

def plot_interactive_cluster_map(tsne_df):
    """
    Creates an interactive Plotly scatter plot of the t-SNE results.
    
    Args:
        tsne_df (pd.DataFrame): The DataFrame from run_clustering_and_tsne.
    """
    print("Generating interactive cluster map...")
    
    fig = px.scatter(
        tsne_df,
        x='tsne_x',
        y='tsne_y',
        color='cluster',
        hover_name='subreddit',
        title=f"t-SNE Visualization of {len(tsne_df)} Subreddit Clusters",
        template='plotly_dark'
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        xaxis=dict(title='t-SNE Component 1', showticklabels=False),
        yaxis=dict(title='t-SNE Component 2', showticklabels=False),
        legend_title_text='Cluster'
    )
    fig.show()

def plot_labeled_cluster_map(tsne_df, label_map):
    """
    Creates an interactive Plotly scatter plot of the t-SNE results,
    but uses a user-provided dictionary to label the clusters.
    
    Args:
        tsne_df (pd.DataFrame): The DataFrame from run_clustering_and_tsne.
        label_map (dict): A dictionary mapping cluster IDs (as str) 
                          to new string labels (e.g., {'5': 'Sports'}).
    """
    print("Generating labeled interactive cluster map...")
    
    # Make a copy to avoid changing the original dataframe
    plot_df = tsne_df.copy()
    
    # Map the new labels. Use .get() to avoid errors
    # if a cluster is in the df but not the map
    plot_df['Topic'] = plot_df['cluster'].apply(
        lambda x: label_map.get(x, f"Cluster {x} (Unlabeled)")
    )
    
    fig = px.scatter(
        plot_df,
        x='tsne_x',
        y='tsne_y',
        color='Topic',  # Color by the new 'Topic' column
        hover_name='subreddit',
        title="t-SNE Visualization of Subreddit Topics",
        template='plotly_dark'
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        xaxis=dict(title='t-SNE Component 1', showticklabels=False),
        yaxis=dict(title='t-SNE Component 2', showticklabels=False),
        legend_title_text='Subreddit Topic'
    )
    fig.show()

# -- Functions for factions with positive posts analysis --

def plot_faction_world_map(factions_df, title="World Map Colored by Reddit Factions"):
    """
    Generates an interactive Plotly choropleth map of country factions.
    Requires the 'pycountry' library to be installed.
    """
    try:
        import pycountry
    except ImportError:
        print("Error: 'pycountry' library not found. Please install it to use this function:")
        print("pip install pycountry")
        return

    def get_iso_alpha_3(country_name):
        """Helper to find ISO 3166-1 alpha-3 code for a country name."""
        try:
            # Handle specific known mismatches first
            name_map = {
                "United States": "USA", "South Korea": "KOR", "Viet Nam": "VNM",
                "Moldova, Republic of": "MDA", "Bolivia, Plurinational State of": "BOL",
                "Palestine, State of": "PSE", "Tanzania, United Republic of": "TZA",
                "Åland Islands": "ALA", "Russia": "RUS", "Turkey": "TUR", "Türkiye": "TUR"
            }
            if country_name in name_map:
                return name_map[country_name]
            
            country = pycountry.countries.get(name=country_name)
            if country: return country.alpha_3
            country = pycountry.countries.get(common_name=country_name)
            if country: return country.alpha_3
            results = pycountry.countries.search_fuzzy(country_name)
            if results: return results[0].alpha_3
            return None
        except Exception:
            return None

    print("Mapping country names to ISO codes...")
    plot_data = factions_df.copy()
    plot_data['iso_alpha'] = plot_data['country'].apply(get_iso_alpha_3)
    
    unmapped = plot_data[plot_data['iso_alpha'].isnull()]['country'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Could not find ISO codes for {len(unmapped)} countries. They will not be plotted.")
        print(unmapped)
    
    plot_data = plot_data.dropna(subset=['iso_alpha'])
    plot_data['faction_str'] = plot_data['faction'].astype(str) 

    print("Generating Plotly map...")
    fig = px.choropleth(
        plot_data,
        locations="iso_alpha",
        color="faction_str",
        hover_name="country",
        title=title
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        legend_title_text='Faction ID'
    )
    fig.show()

def plot_signed_network(mapped_posts_df, factions_df_norm, title="Signed Network of Country Interactions"):
    """
    Generates a detailed NetworkX graph plot based on the provided logic.
    - Nodes are colored by faction.
    - Edges are colored by net sentiment (Green=Positive, Red=Negative).
    - Edge width is log-scaled by total post volume.
    """
    print("\nGenerating Signed Network Plot...")
    try:
        # --- Calculate Negative Links ---
        df_negative_posts = mapped_posts_df[mapped_posts_df["LINK_SENTIMENT"] == -1].copy()
        country_negative_links = (
            df_negative_posts.groupby(["source_country", "target_country"])
            .size()
            .reset_index(name="num_negative_posts")
        )

        # --- Combine Positive and Negative Links ---
        df_positive_posts = mapped_posts_df[mapped_posts_df["LINK_SENTIMENT"] == 1].copy()
        country_positive_links_counts = (
            df_positive_posts.groupby(["source_country", "target_country"])
            .size()
            .reset_index(name="num_positive_posts")
        )

        # Combine counts using source/target pairs
        country_links_combined = pd.merge(
            country_positive_links_counts,
            country_negative_links,
            on=["source_country", "target_country"],
            how="outer"
        ).fillna(0)

        # Aggregate interactions regardless of direction
        country_links_combined['pair'] = country_links_combined.apply(lambda row: tuple(sorted((row['source_country'], row['target_country']))), axis=1)
        signed_agg = country_links_combined.groupby('pair').agg(
            num_positive=('num_positive_posts', 'sum'),
            num_negative=('num_negative_posts', 'sum')
        ).reset_index()
        signed_agg[['country1', 'country2']] = pd.DataFrame(signed_agg['pair'].tolist(), index=signed_agg.index)

        signed_agg["net_sentiment_count"] = signed_agg["num_positive"] - signed_agg["num_negative"]
        signed_agg["total_posts"] = signed_agg["num_positive"] + signed_agg["num_negative"]

        signed_agg = signed_agg[(signed_agg["country1"] != signed_agg["country2"]) & (signed_agg["total_posts"] > 0)]

        # --- Build Signed Graph ---
        G_signed = nx.Graph()
        for _, row in signed_agg.iterrows():
            c1, c2 = row['country1'], row['country2']
            net_sentiment = row['net_sentiment_count']
            total_posts = row['total_posts']
            G_signed.add_edge(c1, c2, net_sentiment=net_sentiment, total_posts=total_posts)

        # Remove isolated nodes (countries with no interactions)
        G_signed.remove_nodes_from(list(nx.isolates(G_signed)))
        if G_signed.number_of_nodes() == 0:
            print("Graph is empty after processing. No plot generated.")
            return

        # --- Prepare for Plotting ---
        edges = G_signed.edges(data=True)
        edge_colors = ['green' if data['net_sentiment'] > 0 else 'red' if data['net_sentiment'] < 0 else 'grey' for u, v, data in edges]
        edge_widths = [np.log1p(data['total_posts']) * 0.5 + 0.1 for u, v, data in edges]

        # Node colors by faction
        num_factions = factions_df_norm['faction'].nunique()
        node_color_map = plt.get_cmap('tab20', max(20, num_factions)) # Ensure enough colors
        country_to_faction = factions_df_norm.set_index('country')['faction'].to_dict()
        # Assign a default color index (e.g., -1 maps to grey) for nodes not in a faction
        node_colors = [node_color_map(country_to_faction.get(node, -1) % 20) if country_to_faction.get(node, -1) != -1 else 'lightgrey' for node in G_signed.nodes()]


        # --- Plotting ---
        fig_net, ax_net = plt.subplots(figsize=(20, 20)) # Increased size
        pos = nx.spring_layout(G_signed, k=0.6, iterations=60, seed=42) # Adjusted layout parameters

        nx.draw_networkx_nodes(G_signed, pos, node_size=60, node_color=node_colors, alpha=0.9, ax=ax_net)
        nx.draw_networkx_edges(G_signed, pos, edge_color=edge_colors, width=edge_widths, alpha=0.3, ax=ax_net)
        # Draw labels with slight adjustments
        nx.draw_networkx_labels(G_signed, pos, font_size=9, font_weight='bold', ax=ax_net)

        ax_net.set_title("Signed Network of Country Interactions (Green=Positive, Red=Negative Net Sentiment)", fontsize=16)
        ax_net.set_axis_off()
        plt.tight_layout()
        plt.savefig("signed_network_plot.png", dpi=300) # Higher resolution save
        print("Signed Network Plot saved as signed_network_plot.png")
        plt.show() # Display the plot directly
        plt.close(fig_net)

    except ImportError:
        print("Could not generate signed network plot: NetworkX or Matplotlib might not be installed correctly.")
        print("Try: pip install networkx matplotlib")
    except Exception as e:
        print(f"An error occurred in plot_signed_network: {e}")
        import traceback
        traceback.print_exc()
