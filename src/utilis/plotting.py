import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np 
import pycountry
import holoviews as hv
from holoviews import opts
import plotly.graph_objects as go
import plotly.colors as pc
from sklearn.manifold import MDS

# Set default renderer for notebooks
pio.renderers.default = "vscode" 

# -- Functions for sentiment analysis

def plot_top_5_subreddits(avg_props_by_subreddit, target_metrics):

    """
    Generates a 2x3 grid of horizontal bar charts showing the top 5 subreddits for specified metrics.

    Args:
        avg_props_by_subreddit (pd.DataFrame): DataFrame containing average metric values per subreddit.
        target_metrics (list): List of column names (metrics) to visualize (e.g., 'LIWC_Anger').

    Returns:
        str: The filename of the saved plot ('images/top_5_subreddits_metrics.png').
    """

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    axes = axes.flatten()  

    chart_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#ffb3e6']

    for i, col in enumerate(target_metrics):
        ax = axes[i]
        
        top_5 = (avg_props_by_subreddit[col].sort_values(ascending=True) .tail(5))  
        current_color = chart_colors[i % len(chart_colors)]                  
            
        top_5.plot(kind='barh', ax=ax, color=current_color, edgecolor='black', width=0.7)
            
        ax.set_title(col.replace('LIWC_', ''), fontsize=12, fontweight='bold')
        ax.set_xlabel('Average Value')
        ax.set_ylabel('') 
        ax.grid(axis='x', linestyle='--', alpha=0.5)


    plt.tight_layout()
    plt.show() 

    output_filename = 'images/top_5_subreddits_metrics.png'
    
    fig.savefig(output_filename, bbox_inches='tight', dpi=300) 
    
    plt.close(fig)

    return output_filename

def plot_countries_by_metric(avg_props_by_country, metric, title):

    """
    Generates and saves a vertical bar chart displaying the top 70 countries for a specific metric.

    Args:
        avg_props_by_country (pd.DataFrame): DataFrame containing average metric values per country.
        metric (str): The specific column name to plot (e.g., 'LIWC_Relig').
        title (str): The title of the chart.

    Returns:
        None: Displays the interactive plot and saves a static image to 'images/avg_relig_by_country.png'.
    """

    plot_df = (
        avg_props_by_country[[metric]]
        .sort_values(by=metric, ascending=False) 
        .reset_index()
        .head(70)
    )

    fig = px.bar(
        plot_df,
        x="source_country", 
        y=metric,            
        title=title,
        text_auto='.3f',
        color=metric,
        color_continuous_scale='Viridis',
        height=600
    )

    fig.update_layout(
        xaxis_title="Country",             
        yaxis_title="Average Sentiment Score", 
        xaxis=dict(
            dtick=1,
            automargin=True,
            tickangle=-45 
        ),
        margin=dict(b=150), 
        showlegend=False
    )

    output_filename = 'images/avg_relig_by_country.png'
    fig.write_image(output_filename, scale=3) 
    fig.show()
    
# -- Functions for interaction analysis

def chord_plot(df_final):

    """
    Generates and saves an interactive Chord diagram visualizing the top 50 inter-country interactions.

    Args:
        df_final (pd.DataFrame): DataFrame containing 'Country_A', 'Country_B', and 'n_interactions'.

    Returns:
        hv.Chord: The generated interactive Chord diagram object.
    """

    hv.extension('bokeh')

    top_interactions = df_final.head(50).copy()
    
    chord_data = top_interactions[['Country_A', 'Country_B', 'n_interactions']]
    chord_data.columns = ['source', 'target', 'value']

    chord = hv.Chord(chord_data)

    viz = chord.opts(
        opts.Chord(
            labels='index',           
            label_text_font_size='10pt',
            label_text_color='black',
            
            cmap='Category20',        
            edge_cmap='Category20',  
            edge_color='source',      
            node_color='index',       

            edge_line_color=None,
            
            edge_hover_line_color='black',
            node_hover_fill_color='red',
            
            width=750, 
            height=750,
            title="Inter-Country Digital Connections (Interaction Volume)"
        )
    )

    output_filename = 'images/country_chord_diagram.html'
    hv.save(viz, output_filename)

    return viz

# -- Functions for cluster with embedding analysis --

def plot_kmeans_elbow(elbow_df):
    """
    Plots the results of the K-Means elbow analysis using Matplotlib.
    
    Args:
        elbow_df (pd.DataFrame): DataFrame from calculate_kmeans_elbow.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    plt.plot(elbow_df['k'], elbow_df['inertia'], marker='o')
    plt.title('K-Means Elbow Plot')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)

    output_filename = 'images/kmeans_elbow_plot.png' 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close(fig)

def plot_labeled_cluster_map(tsne_df, label_map):
    """
    Creates an interactive Plotly scatter plot of the t-SNE results,
    but uses a user-provided dictionary to label the clusters.
    
    Args:
        tsne_df (pd.DataFrame): The DataFrame from run_clustering_and_tsne.
        label_map (dict): A dictionary mapping cluster IDs (as str) 
                          to new string labels (e.g., {'5': 'Sports'}).
    """
    
    plot_df = tsne_df.copy()
    
    plot_df['Topic'] = plot_df['cluster'].apply(
        lambda x: label_map.get(x, f"Cluster {x} (Unlabeled)")
    )
    
    fig = px.scatter(
        plot_df,
        x='tsne_x',
        y='tsne_y',
        color='Topic', 
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

    output_filename = 'images/tsne_cluster_map.html' 
    fig.write_html(output_filename)

    fig.show()

# -- Functions for factions with positive posts analysis --

def plot_faction_world_map(factions_df, title):
    """
    Generates an interactive Plotly choropleth map of country factions.
    """

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

    plot_data = factions_df.copy()
    plot_data['iso_alpha'] = plot_data['country'].apply(get_iso_alpha_3)
    
    plot_data = plot_data.dropna(subset=['iso_alpha'])
    plot_data['faction_str'] = plot_data['faction'].astype(str) 

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

    filename = 'images/faction_world_map.html'
    fig.write_html(filename)

def plot_signed_network(mapped_posts_df, factions_df_norm, title):
    """
    Generates an interactive Plotly network graph.
    - Nodes are colored by faction.
    - Edges are colored by net sentiment (Green=Positive, Red=Negative).
    - Edge width is log-scaled by total post volume.
    """
    df_negative_posts = mapped_posts_df[mapped_posts_df["LINK_SENTIMENT"] == -1].copy()
    country_negative_links = (
        df_negative_posts.groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="num_negative_posts")
    )

    df_positive_posts = mapped_posts_df[mapped_posts_df["LINK_SENTIMENT"] == 1].copy()
    country_positive_links_counts = (
        df_positive_posts.groupby(["source_country", "target_country"])
        .size()
        .reset_index(name="num_positive_posts")
    )

    country_links_combined = pd.merge(
        country_positive_links_counts,
        country_negative_links,
        on=["source_country", "target_country"],
        how="outer"
    ).fillna(0)

    country_links_combined['pair'] = country_links_combined.apply(
        lambda row: tuple(sorted((row['source_country'], row['target_country']))), axis=1
    )
    signed_agg = country_links_combined.groupby('pair').agg(
        num_positive=('num_positive_posts', 'sum'),
        num_negative=('num_negative_posts', 'sum')
    ).reset_index()
    signed_agg[['country1', 'country2']] = pd.DataFrame(signed_agg['pair'].tolist(), index=signed_agg.index)

    signed_agg["net_sentiment_count"] = signed_agg["num_positive"] - signed_agg["num_negative"]
    signed_agg["total_posts"] = signed_agg["num_positive"] + signed_agg["num_negative"]

    signed_agg = signed_agg[(signed_agg["country1"] != signed_agg["country2"]) & (signed_agg["total_posts"] > 0)]

    G_signed = nx.Graph()
    for _, row in signed_agg.iterrows():
        c1, c2 = row['country1'], row['country2']
        G_signed.add_edge(c1, c2, 
                          net_sentiment=row['net_sentiment_count'], 
                          total_posts=row['total_posts'],
                          pos_count=row['num_positive'],
                          neg_count=row['num_negative'])

    G_signed.remove_nodes_from(list(nx.isolates(G_signed)))

    pos = nx.spring_layout(G_signed, k=0.6, iterations=60, seed=42)


    edge_traces = []
    
    for u, v, data in G_signed.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        sentiment = data['net_sentiment']
        total = data['total_posts']
        
        if sentiment > 0:
            color = 'rgba(0, 128, 0, 0.6)' # Green
        elif sentiment < 0:
            color = 'rgba(255, 0, 0, 0.6)' # Red
        else:
            color = 'rgba(128, 128, 128, 0.4)' # Grey

        width = np.log1p(total) * 0.5 + 0.5

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            mode='lines',
            text=f"{u} - {v}<br>Net: {sentiment}<br>Total: {total} (Pos: {data['pos_count']}, Neg: {data['neg_count']})",
            showlegend=False
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    country_to_faction = factions_df_norm.set_index('country')['faction'].to_dict()
    unique_factions = sorted(list(set(country_to_faction.values())))
    
    colors_list = pc.qualitative.Dark24 * 2 
    faction_color_map = {f: colors_list[i % len(colors_list)] for i, f in enumerate(unique_factions)}

    for node in G_signed.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        faction_id = country_to_faction.get(node, -1)
        faction_name = f"Faction {faction_id}" if faction_id != -1 else "Unknown"
        
        node_text.append(f"<b>{node}</b><br>Faction: {faction_name}")
        
        if faction_id != -1:
            node_colors.append(faction_color_map.get(faction_id, 'lightgrey'))
        else:
            node_colors.append('lightgrey')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G_signed.nodes()], 
        textposition="top center",
        hovertext=node_text, 
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line_width=1,
            line_color='black'
        ),
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title={'text': title, 'font': {'size': 16}},
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=800,
                        width=1200
                    ))
    
    output_html = 'images/signed_network_interactive.html'
    
    fig.write_html(output_html)    
    fig.show()

def plot_faction_evolution(y_factions_norm_df):

    """
    Generates and saves an interactive Sankey diagram visualizing the flow of countries between factions over time.

    It tracks how countries transition between different factions across years (or quarters), 
    mapping these transitions as flows with colors corresponding to the source faction.

    Args:
        y_factions_norm_df (pd.DataFrame): DataFrame containing 'year', 'faction', and a list of 'countries'.

    Returns:
        None: Displays the plot and saves it as 'images/faction_evolution.html'.
    """
    
    df_factions = y_factions_norm_df.explode('countries').reset_index(drop=True)

    df_factions = df_factions.rename(columns={'countries': 'source_country'})

    df_factions = df_factions[['year', 'faction', 'source_country']]

    quarters = sorted(df_factions['year'].unique())

    palette = pc.qualitative.Plotly 
    
    all_nodes = []
    node_indices = {}
    node_colors = [] 
    
    for q in quarters:
        q_data = df_factions[df_factions['year'] == q]
        factions = sorted(q_data['faction'].unique())
        
        for f in factions:
            node_id = f"{str(q)} - Faction {f}"
            
            if node_id not in node_indices:
                node_indices[node_id] = len(all_nodes)
                all_nodes.append(node_id)
                
                f_int = int(f) 
                
                specific_color = palette[f_int % len(palette)]
                node_colors.append(specific_color)

    sources = []
    targets = []
    values = []
    link_colors = [] 
    
    for i in range(len(quarters) - 1):
        q_current = quarters[i]
        q_next = quarters[i+1]
        
        df_curr = df_factions[df_factions['year'] == q_current]
        df_next = df_factions[df_factions['year'] == q_next]
        
        transitions = pd.merge(
            df_curr[['source_country', 'faction']], 
            df_next[['source_country', 'faction']], 
            on='source_country', 
            suffixes=('_curr', '_next')
        )
        
        flow_counts = (
            transitions.groupby(['faction_curr', 'faction_next'])
            .size()
            .reset_index(name='count')
        )
        
        for _, row in flow_counts.iterrows():
            src_node = f"{str(q_current)} - Faction {row['faction_curr']}"
            tgt_node = f"{str(q_next)} - Faction {row['faction_next']}"
            
            sources.append(node_indices[src_node])
            targets.append(node_indices[tgt_node])
            values.append(row['count'])
            

            f_id = int(row['faction_curr']) if str(row['faction_curr']).isdigit() else abs(hash(row['faction_curr']))
            base_color = palette[f_id % len(palette)]
            
            if base_color.startswith('#'):
                h = base_color.lstrip('#')
                rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                link_colors.append(f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)')
            else:
                link_colors.append('rgba(150, 150, 150, 0.2)') 

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors 
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors 
        )
    )])

    fig.update_layout(
        title_text="Evolution of Factions over time",
        font_size=10,
        height=800
    )
    
    fig.show()

    filename = 'images/faction_evolution.html'
    fig.write_html(filename)

def heatmap_co_occurrence(target_countries, q_factions_norm_df):

    """
    Generates and saves a heatmap visualizing the stability of alliances between target countries.

    It calculates how often pairs of countries appear in the same faction across different time periods 
    and displays this frequency as a color-coded matrix.

    Args:
        target_countries (list): List of country names to include in the heatmap.
        q_factions_norm_df (pd.DataFrame): DataFrame containing faction membership data with a 'countries' column.

    Returns:
        None: Displays the interactive heatmap and saves it as 'images/heatmap_co_occurrence.html'.
    """

    pair_counts = {}

    for c1 in target_countries:
        for c2 in target_countries:
            pair_counts[(c1, c2)] = 0

    for _, row in q_factions_norm_df.iterrows():

        members = [c for c in row['countries'] if c in target_countries]
        
        for i in range(len(members)):
            for j in range(len(members)):
                pair_counts[(members[i], members[j])] += 1

    matrix_data = []
    for c1 in target_countries:
        row_data = []
        for c2 in target_countries:
            row_data.append(pair_counts[(c1, c2)])
        matrix_data.append(row_data)

    df_heatmap = pd.DataFrame(matrix_data, index=target_countries, columns=target_countries)

    fig = px.imshow(
        df_heatmap, 
        width=1000, height=1000,
        title="Alliance Stability: How often are countries in the same faction?",
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='linear', 
            tickangle=-90,     
            side='bottom'      
        ),
        
        margin=dict(l=50, r=50, t=50, b=200) 
    )
    # -----------------------------

    fig.show()

    filename = 'images/heatmap_co_occurrence.html'
    fig.write_html(filename)

# -- Functions for shortest path analysis

def plot_MDS_shortest_path(df_country_paths):

    """
    Visualizes the "distances" between countries using Multidimensional Scaling (MDS) based on shortest path lengths.

    It creates a symmetric distance matrix from shortest path data, handling missing paths with a penalty value,
    and projects the countries into a 2D space where closer points represent stronger connectivity.

    Args:
        df_country_paths (pd.DataFrame): DataFrame containing 'source_country', 'target_country', and 'shortest_path_length'.

    Returns:
        None: Displays the interactive scatter plot and saves it as 'images/mds_shortest_path.html'.
    """

    all_countries = np.unique(
        np.concatenate([
            df_country_paths['source_country'].unique(), 
            df_country_paths['target_country'].unique()
        ])
    )

    matrix_df = df_country_paths.pivot(
        index='source_country', 
        columns='target_country', 
        values='shortest_path_length'
    )

    matrix_df = matrix_df.reindex(index=all_countries, columns=all_countries)

    np.fill_diagonal(matrix_df.values, 0)

    matrix_df = matrix_df.replace([np.inf, -np.inf], np.nan)

    max_path = matrix_df.max().max()
    penalty_val = max_path + 2 if not pd.isna(max_path) else 10

    matrix_df = matrix_df.fillna(penalty_val)
    matrix_symmetric = (matrix_df.values + matrix_df.values.T) / 2

    embedding = MDS(
        n_components=2, 
        dissimilarity='precomputed', 
        random_state=42, 
        normalized_stress='auto'
    )
    transformed = embedding.fit_transform(matrix_symmetric)

    df_coords = pd.DataFrame(transformed, columns=['x', 'y'])
    df_coords['Country'] = all_countries 

    df_coords['color_gradient'] = df_coords['x'] + df_coords['y']

    fig = px.scatter(
        df_coords, 
        x='x', 
        y='y', 
        text='Country',
        hover_name='Country',
        color='color_gradient',
        color_continuous_scale='Turbo',
        size_max=20
    )

    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='White')),
        textposition='top center',
        textfont=dict(family="Arial", size=10, color="gray")
    )

    fig.update_layout(
        title="The 'Reddit Geography' of Nations (MDS Projection)",
        title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white',
        showlegend=False,
        coloraxis_showscale=False,
        height=800
    )

    fig.show()

    filename = 'images/mds_shortest_path.html'
    fig.write_html(filename)

# -- Functions for sport analysis

def plot_sunburst(df_top_sport_per_country):

    """
    Generates and saves an interactive Sunburst chart showing the dominant sports across different countries.

    It creates a hierarchical visualization where the inner ring represents sports and the outer ring lists 
    countries, sized by their total interaction volume.

    Args:
        df_top_sport_per_country (pd.DataFrame): DataFrame containing 'Sport', 'Country', and 'total_interactions'.

    Returns:
        None: Displays the interactive sunburst chart and saves it as 'images/sport_sunburst.html'.
    """

    fig = px.sunburst(
        df_top_sport_per_country,
        path=['Sport', 'Country'],  
        values='total_interactions', 
        color='Sport',               
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data=['total_interactions']
    )

    fig.update_layout(
        title_text="Global Sports Fandom: Top Sport by Country",
        title_x=0.5,          
        font_family="Arial",   
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,            
        margin=dict(t=50, l=0, r=0, b=0) 
    )

    fig.update_traces(
        textinfo="label+percent parent",
        insidetextorientation='radial'  
    )

    fig.show()

    filename = 'images/sport_sunburst.html'
    fig.write_html(filename)

# --Response analysis--

def funnel_graph_global(total_initiators_global, total_responses_global):

    """
    Generates and saves a funnel chart visualizing the global reciprocity rate between initiations and responses.

    It creates a two-stage funnel showing the drop-off from total initial posts to reciprocal 
    responses received within a set window, calculating the percentage conversion.

    Args:
        total_initiators_global (int): Total count of initiating posts (A->B).
        total_responses_global (int): Total count of reciprocal responses (B->A).

    Returns:
        go.Figure: The generated interactive funnel chart object.
    """

    fig = go.Figure(go.Funnel(
        y = ["Total Initiations (A->B)", "Received Responses (B->A)"],
        x = [total_initiators_global, total_responses_global],
        textposition = "inside",
        textinfo = "value+percent initial",
        opacity = 0.65, 
        marker = {"color": ["#1f77b4", "#2ca02c"]}
    ))

    fig.update_layout(title_text="Global Reciprocity Funnel")

    filename = 'images/funnel_global.html'
    fig.write_html(filename)

    return fig

def funnel_graph_intra_country(total_initiators, total_responses):

    """
    Generates and saves a funnel chart visualizing reciprocity between subreddits within the same country.

    It creates a two-stage funnel showing the drop-off from initial intra-country posts to 
    reciprocal responses, highlighting the domestic engagement rate.

    Args:
        total_initiators (int): Total count of initiating posts between domestic subreddits.
        total_responses (int): Total count of reciprocal responses received.

    Returns:
        go.Figure: The generated interactive funnel chart object.
    """

    fig = go.Figure(go.Funnel(
        y = ["Total Initiations (Sub A -> Sub B)", "Received Responses (Sub B -> Sub A)"],
        x = [total_initiators, total_responses],
        textposition = "inside",
        textinfo = "value+percent initial",
        opacity = 0.65, 
        marker = {"color": ["#1f77b4", "#FF7F0E"]} 
    ))

    fig.update_layout(title_text="Global Intra-Country Reciprocity Funnel (Subreddits)")

    filename = 'images/funnel_intra_country.html'
    fig.write_html(filename)

    return fig




