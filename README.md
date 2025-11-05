# Mapping Societal Dynamics on Reddit

## Abstract

People in the society tend to create communities based on geography, interests, and culture. This project explores whether Reddit reflects these aggregation dynamics, questioning if it's a valid proxy for real-world interactions or just a skewed representation. The core idea is to map subreddits to entities like countries (e.g., r/italy) or topics (e.g., r/politic) to analyze their interaction patterns. The goal is twofold: first, study "country-based" groups by analyzing their engagement with thematic subreddits. Second, determining if inter-subreddit interactions have an influence on future posts. We aim to uncover if digital interactions can reveal cultural traits and intrinsic dynamics present in their real-world counterparts.

## Research Questions

Our analysis follows hierarchical questions, from data processing to high-level insights:

1.  **Subreddit-to-Topic Mapping:** How accurately can we map subreddit names to real-world entities (countries) and topics (politics, sports) using keyword, fuzzy matching, and rule-based heuristics?
2.  **Embedding-based Clustering:** Can we identify "natural" community clusters from subreddit embeddings using unsupervised methods (K-Means, t-SNE)? How well do these align with our manual country or topic labels?
3.  **Community Network Analysis:** What is the macro-structure of the Reddit interaction network aggregated by country or topic? Can we identify factions or one-to-one relationships (e.g., between countries) by analyzing the hyperlink graph?
4.  **Mirroring Analysis:**  To what extent do reciprocal online interactions exhibit stylistic mirroring, and how does this level of similarity compare to a random baseline?
5.  **Political Landscape Analysis:** How do national communities interact with subreddits of different political ideologies? Can we quantify and visualize political "echo chambers" or "bridges" through sentiment and link analysis?
6.  **Thematic Interaction (Case Study):** What are the dominant sports-related topics for different national communities, and what does this reveal about cross-cultural engagement?

**Further Questions:**
* Are there "broker" subreddits (high centrality) bridging disconnected national or political communities?
* Do linguistic features (LIWC) and sentiment differ in posts linking *within* (country-to-country) versus *between* communities (country-to-politics)?
* Can 2014-2016 economic relationships be seen in national subreddit interaction patterns?

## Data Setup

To run the analysis, you must first place the original datasets in the `/data/` folder.

The following files are required:

* `soc-redditHyperlinks-body.tsv`
* `soc-redditHyperlinks-title.tsv`
* `web-redditEmbeddings-subreddits.csv`

The `src/data` contains all the files and notebook that have been used to pre-process the data or map subreddits. Their output has been saved (as csv files) in the `data` folder for a faster analysis. 
The correct way to reproduce all the findings is to simply run `results.ipynb` using the already approved and saved files

## Data Enrichment and Metadata Generation

From these datasets, we generated new metadata by classifying subreddits into categories such as countries, political ideologies, and sports.

1.  **Country-Subreddit Mapping:**
    * **Source:** Generated from unique subreddits.
    * **Process:** (Notebooks: `make_initial_subreddit_maps.ipynb`, `filter_matches.ipynb`). We built a map of country names, demonyms, and codes. We applied matching rules (direct, token, fuzzy string matching with `rapidfuzz`) to map names (e.g., 'r/norge' to 'Norway'). An initial 1700-subreddit mapping was manually checked and filtered. A much bigger country related dataset has been also produced for some analysis; please check in the section **Mapping Subreddits to Country** in `results.ipynb` for a better description.
    * **Management:** Final mapping saved as `subreddit_matches_approved.csv` for lookup.

2.  **Political-Ideology-Subreddit Mapping:**
    * **Source:** Generated from unique subreddits (`filter_politic_subreddits.ipynb`).
    * **Process:** Defined a keyword dictionary for 'left', 'right', and 'center_or_other' political ideologies. Used fuzzy-matching to find subreddits matching keywords with high confidence (score > 95).
    * **Management:** Produces `politic_subreddit.csv` to be merged with the main dataset.

3.  **Sport-Subreddit Mapping:**
    * **Source:** Generated from `df_countries_expanded` (`filter_sports.py`), looking for sport-country interactions.
    * **Process:** Defined a sports keyword dictionary. Used fuzzy-matching to find subreddits matching these keywords with high confidence.
    * **Management:** Produces `df_countries_sport.csv`, a dataset of posts with country-sport links.

4.  **Subreddit Embeddings:**
    * **Source:** Used pre-computed embeddings provided with the dataset (`web-redditEmbeddings-subreddits.csv`).
    * **Process:** Used as features for clustering.

## Methods

1.  **Data Ingestion & Preparation:**
    * Load and merge hyperlink datasets.
    * Load and merge our generated mappings, tagging source and target subreddits.

2.  **Clustering & Visualization:**
    * Apply K-Means clustering (using `plotting.py` elbow-plot for optimal *k*) on embeddings to find "natural" clusters.
    * Visualize clusters using t-SNE.
    * Generate a final network visualization (`matplotlib`/`networkx`) showing the signed, aggregated network of country/political interactions.

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
    * Create a "signed network" from net sentiment between nodes (`plotting.py`).
    * Calculate shortest paths between key national communities.

### Note on Country Representation
Country-based subreddits are used as proxies for national communities. We acknowledge that not all users in these spaces are actually from the corresponding country. Still, since our goal is to see whether the “world of Reddit” reflects real-world dynamics, we treat these interactions as meaningful indicators of perceived international relationships.

## Proposed Timeline

* **Week 10/11 - 16/11:** Data cleaning and finalization of mappings.
    * **Milestone:** Final, validated `.csv` mapping files (`country`, `politic`, `sport`).
* **Week 17/11 - 23/11:** Work on ADA course Homework.
* **Week 24/11 - 30/11:** Network construction, aggregate analysis, sentiment weighting. Finalize Homework.
    * **Milestone:** Full NetworkX graph built, saved; basic properties calculated.
* **Week 1/12 - 7/12:** Final analysis and results compilation.
    * **Milestone:** All research questions addressed. Final report/notebook drafted.
* **Week 8/12 - 14/12:** Begin final website and graphical work.
    * **Milestone:** Implemented plots to explain notebook results.
* **Week 15/12 - 17/12:** Final website development. Clean GitHub repo and code.
    * **Milestone (P3):** Project finished.

## Current workload division within the team

* **Julie: Data & Country Mapping Lead**
    * **Tasks:** Cleaned dataset for country-subreddit mapping. Runs/refines country mapping (`make_initial_subreddit_maps.ipynb`). Analysis of one-to-one country interactions.
    * **Milestone:** Delivers final `subreddit_matches_approved.csv`.

* **Dario: Cluster and Community analysis Lead**
    * **Tasks:** Political ideology filtering (`filter_politic_subreddits.ipynb`). Performed K-Means and t-SNE. Faction analysis (interactions, evolution). Handles faction/network plotting. Writes Readme.
    * **Milestone:** Delivers `politic_subreddit.csv` and analysis in `results.ipynb`.

* **Noemi: Chain of Interactions and Sport analysis Lead**
    * **Tasks:** Delivered reciprocity probabilities (global, intra-country). Completed statistical analysis of linguistic style mirroring. Sport subreddits filtering (`filter_sports.ipynb`). 
    Expanded the dataset for country-subreddit mapping (`filter_countries_expanded.py`).
    * **Milestone:** Delivers `df_countries_sport.csv` and analysis in `results.ipynb`.

* **Simon: Network Analysis Lead**
    * **Tasks:** Building core `networkx` graph (`network_analysis.ipynb`). Calculates graph properties (centrality, degree, paths).
    * **Milestone:** Delivers saved graph object and notebook with network statistics.

* **Pietro: Economics Analysis Lead**
    * **Tasks:** Economics subreddits filtering. Analysis of economics-subreddit interactions, plotting results, and stats for `economic_links_with_geo_labeled2.csv`.
    * **Milestone:** Delivers `economic_links_with_geo_labeled2.csv` and analysis in `results.ipynb`.
