# Mapping Societal Dynamics on Reddit

<p>
  <img src="images/cartoon.png" alt="Meme" width="500"/>
</p>


## Data Story: https://julsofpet.github.io/the_world_of_reddit/


## Abstract

People in the society tend to create communities based on geography, interests, and culture. This project explores whether Reddit reflects these aggregation dynamics, questioning if it's a valid proxy for real-world interactions or just a skewed representation. The core idea is to map subreddits to entities like countries (e.g., r/italy) or topics (e.g., r/sport) to analyze their interaction patterns. The goal is twofold: first, study "country-based" groups by analyzing their engagement with thematic subreddits. Second, determining if inter-subreddit interactions have an influence on future posts. We aim to uncover if digital interactions can reveal cultural traits and intrinsic dynamics present in their real-world counterparts.

## Research Questions

Our analysis follows hierarchical questions, from data processing to high-level insights:

1.  **Subreddit-to-Topic Mapping:** How accurately can we map subreddit names to real-world entities (countries) and topics (sports) using keyword, fuzzy matching, and rule-based heuristics?
2.  **Embedding-based Clustering:** Can we identify "natural" community clusters from subreddit embeddings using unsupervised methods (K-Means, t-SNE)? How well do these align with our manual country or topic labels?
3.  **Community Network Analysis:** What is the macro-structure of the Reddit interaction network aggregated by country or topic? Can we identify factions or one-to-one relationships (e.g., between countries) by analyzing the hyperlink graph?
4.  **Mirroring Analysis:**  To what extent do reciprocal online interactions exhibit stylistic mirroring, and how does this level of similarity compare to a random baseline?
5.  **Thematic Interaction (Case Study):** What are the dominant sports-related topics for different national communities, and what does this reveal about cross-cultural engagement?

## Data Setup

To run the analysis, you must first place the original datasets in the `/data/` folder.

The following files are required:

* `soc-redditHyperlinks-body.tsv`
* `soc-redditHyperlinks-title.tsv`
* `web-redditEmbeddings-subreddits.csv`

The `src/data` contains all the files and notebook that have been used to pre-process the data or map subreddits. Their output has been saved (as csv files) in the `data` folder for a faster analysis. 
The correct way to reproduce all the findings is to simply run `results.ipynb` using the already approved and saved files

## Data Enrichment and Metadata Generation

From these datasets, we generated new metadata by classifying subreddits into categories.
1.  **Country-Subreddit Mapping:**
    * **Source:** Generated from unique subreddits.
    * **Process:** (file: `src/data/filter_countries_expanded.py`) We built a map of country names, demonyms, and codes. We applied matching rules (direct, token, fuzzy string matching with `rapidfuzz`) to map names (e.g., 'r/norge' to 'Norway'). An initial 3700-subreddit mapping was manually checked and filtered.
    * **Management:** Final mapping saved as `data/country_matches_map_expanded.csv` for lookup.

2.  **Sport-Subreddit Mapping:**
    * **Source:** Generated from unique subreddits. 
    * **Process:** (file: `src/data/filter_sports.py`) Defined a sports keyword dictionary. Used fuzzy-matching (`src/data/dataFunctions.py`) to find subreddits matching these keywords with high confidence.
    * **Management:** Produces `data/df_countries_sport.csv`, a dataset of posts with country-sport links.

3.  **Subreddit Embeddings:**
    * **Source:** Used pre-computed embeddings provided with the dataset `web-redditEmbeddings-subreddits.csv`.
    * **Process:** Used as features for clustering.

### Note on Country Representation
Country-based subreddits are used as proxies for national communities. We acknowledge that not all users in these spaces are actually from the corresponding country. Still, since our goal is to see whether the “world of Reddit” reflects real-world dynamics, we treat these interactions as meaningful indicators of perceived international relationships.

## Methods

1.  **Data Ingestion & Preparation:**
    * Load and merge hyperlink datasets.
    * Load and merge our generated mappings, tagging source and target subreddits.

2.  **Clustering & Visualization:**
    * Apply K-Means clustering (using `src/utilis/plotting.py` elbow-plot for optimal *k*) on embeddings to find "natural" clusters.
    * Visualize clusters using t-SNE.

3.  **Interaction effect**
    * Calculate 7-day conditional reciprocity probability for inter-country interactions (B-->A | A-->B).
    * Apply same logic to intra-country interactions.
    * Test for stylistic mirroring by comparing LIWC/VADER cosine similarity of reciprocal pairs against a random baseline.

4.  **Community & Interaction Analysis:**
    * Merge topic subreddits with countries to find patterns.
    * Use TIMESTAMP for timewise analysis.
    * Create a summary DataFrame of interactions.

5.  **Network Construction (NetworkX):**
    * (For `network_analysis.ipynb`) Construct a directed `networkx` graph (nodes=subreddits, edges=hyperlinks).
    * Attach post properties (sentiment, LIWC) as edge attributes.
    * Create a "signed network" from net sentiment between nodes (`src/utilis/plotting.py`).
    * Calculate shortest paths between key national communities.


## Workload division within the team

* **Julie: Data & Country Mapping Lead**
    * **Tasks:** Cleaned dataset for country-subreddit mapping. Runs/refines country mapping (`filter_country_expanded.py`). Analysis of one-to-one country interactions. Development of the website
    * **Milestone:** Delivers final `data/country_matches_map_expanded.csv`.

* **Dario: Cluster and Community analysis Lead**
    * **Tasks:** Performed K-Means and t-SNE. Faction analysis (interactions, evolution). Handles faction/network plotting. Readme. Data story script and development of the website.
    * **Milestone:** Analysis in `results.ipynb`.

* **Noemi: Chain of Interactions and Sport analysis Lead**
    * **Tasks:** Delivered reciprocity probabilities (global, intra-country). Completed statistical analysis of linguistic style mirroring. Sport subreddits filtering (`filter_sports.ipynb`). Data story script and development of the website.
    * **Milestone:** Delivers `data/df_country_sport_map.csv` and analysis in `results.ipynb`.

* **Simon: Network Analysis Lead**
    * **Tasks:** Building core `networkx` graph (`network_analysis.py`). Calculates graph properties (centrality, degree, paths). Development of the website.
    * **Milestone:** Delivers saved graph object.

* **Pietro: Data Cleaning&Labeling and Support**
    * **Tasks:** Dataset cleaning and preprocessing, with particular attention to fixing and standardizing country labels. Provided support in minor tasks across data analysis and implementation stages.
    * **Milestone:** Contributed to the cleaned dataset used in the analyses.

## Repository structure

```text
ADA-2025-PROJECT-DJNPSGROUP2025/
├── data/                                               # Data folder (contains raw Reddit data and processed CSVs)
│   ├── country_matches_map_expanded.csv                # Expanded dataset mapping country matches
│   ├── country_shortest_paths_output.csv               # Output results from shortest path analysis
│   ├── df_country_sport_map.csv                        # Mapping dataframe for countries and sports
│   ├── soc-redditHyperlinks-body.tsv                   # Reddit hyperlinks dataset (body hyperlinks)
│   ├── soc-redditHyperlinks-title.tsv                  # Reddit hyperlinks dataset (titles hyperlinks)
│   └── web-redditEmbeddings-subreddits.csv             # Subreddit embeddings data
├── images/                                             # Folder for project images and assets
├── src/                                                # Source code for data processing and analysis
│   ├── data/                                           # Scripts specific to data filtering and handling
│   │   ├── dataFunctions.py                            # Core functions for data manipulation
│   │   ├── filter_countries_expanded.py                # Script to expand and filter country data
|   |   ├── network_analysis.py                         # Functions for shortest path dataset creation
│   │   └── filter_sports.py                            # Script to filter sport-related data
│   └── utilis/                                         # Utility modules and helper functions
│       ├── plotting.py                                 # Helper functions for generating plots
│       └── utilis.py                                   # General utility functions
├── .gitignore                                          # Files and directories to be ignored by Git
├── README.md                                           # Project documentation
├── requirements.txt                                    # Required Python packages
└── results.ipynb                                       # Main Jupyter notebook containing the entire analysis
```

## How to run the code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/epfl-ada/ada-2025-project-djnpsgroup2025.git
   cd ada-2025-project-djnpsgroup2025
    ```
2. **Create and activate a virtual environment:**

    For macOS/Linux
    ```bash 
    python -m venv venv
    source venv/bin/activate
    ```
    or
   
   ```
   conda create -n "ada-project"
   conda install pip
   pip install -r requirements.txt
    ```

    For Windows
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```


4.  **Install the required packages using the requirements.txt file:**

    ```bash
    pip install -r requirements.txt
    ```
3. **Download the data:**
   - Download the dataset from the paper [Social Network: Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) 
   - Extract the files and ensure ```soc-redditHyperlinks-body.tsv```, ```soc-redditHyperlinks-title.tsv```and ```web-redditEmbeddings-subreddit.csv``` are placed inside the data/ folder.
4.  **Run the notebook:**
    ```bash
    jupyter notebook results.ipynb


