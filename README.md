# Mapping Societal Dynamics on Reddit

## Abstract

One essential aspect of society is the human propensity to group people according to common geography, interests, and culture. This project is motivated by exploring whether these same aggregation dynamics are reflected in Reddit's communities. We question if this platform can serve as a valid proxy for complex real-world interactions or if it is merely a small, skewed representation of our world.
The core idea is to map subreddits to specific entities, such as countries (e.g., r/italy) or distinct topics (e.g., r/politic), to analyze their interaction patterns.
The goal is divided in 2 steps: first, to determine if intrinsic components within these digital groups (like language or common interests) highlight real-world relationships and factions. Second, to study the characteristics of these "country-based" groups by analyzing how they engage with various thematic subreddits. We aim to tell a story of how analyzing these digital interactions can uncover underlying cultural traits and intrinsic dynamics present in their real-world counterparts.

## Research Questions

Our analysis is guided by a set of hierarchical questions, starting from data processing and moving toward high-level community insights:

1.  **Subreddit-to-Topic Mapping:** How accurately can we map subreddit names to real-world entities (like countries) and thematic topics (like politics or sports) using a combination of keyword, fuzzy matching, and rule-based heuristics?
2.  **Embedding-based Clustering:** Is it possible to identify "natural" community clusters from subreddit embeddings using unsupervised methods (K-Means, t-SNE)? How well do these data-driven clusters align with our manually-defined country or topic labels?
3.  **Community Network Analysis:** What is the macro-structure of the Reddit interaction network when aggregated by country or topic? Can we explore and identify community factions or specific one-to-one relationships (e.g., between specific countries) by analyzing the hyperlink graph?
4.  **Political Landscape Analysis:** How do national communities interact with subreddits of different political ideologies (left, right, center)? Can we quantify and visualize political "echo chambers" or "bridges" through sentiment and link analysis?
5.  **Thematic Interaction (Case Study):** What are the dominant sports-related topics of interest for different national communities, and what does this reveal about cross-cultural engagement?

**Further Questions:**
* Are there specific "broker" subreddits (with high network centrality) that bridge otherwise-disconnected national or political communities?
* Do linguistic features (LIWC) and post sentiment differ significantly in posts that link *within* a community (e.g., country-to-country) versus posts that link *between* communities (e.g., country-to-politics)?
* Can we find evidence of real-world geopolitical relationships or cultural affinities from the 2014-2016 period reflected in the interaction patterns between national subreddits?

## Proposed Additional Datasets (Data Enrichment)

Our primary dataset is the `soc-redditHyperlinks-title.tsv` and `soc-redditHyperlinks-body.tsv` corpus. The core of our project lies in enriching this data by classifying the subreddits. We are not proposing to add external datasets, but rather to generate new metadata through the following processes:

1.  **Country-Subreddit Mapping:**
    * **Source:** We generated this mapping from the list of all unique subreddits in the dataset.
    * **Process:** We used the notebooks `make_initial_subreddit_maps.ipynb` and `filter_maches.ipynb`. This involves a multi-step process:
        * Generating a comprehensive map of country names, demonyms, and codes.
        * Applying a series of matching rules (direct match, token match, fuzzy string matching with `rapidfuzz`) to map subreddit names (e.g., 'r/norge') to a standardized country (e.g., 'Norway').
        * After making the first mapping of 1700 subreddits to the country they belong to, we manually checked the matched, and filtered the big csv file into the approved, rejected, unsure.
    * **Management:** The final mapping has been saved as a `subreddit_matches_approved.csv` file to be used as a lookup table in all subsequent analyses.

2.  **Political-Ideology-Subreddit Mapping:**
    * **Source:** We generated this from the list of all unique subreddits, as detailed in `filter_politic_subreddits.ipynb`.
    * **Process:**  We defined a comprehensive dictionary of keywords associated with 'left', 'right', and 'center_or_other' ideologies.
        * Using the fuzzy-matching functions in `data.py`, we searched for subreddits whose names match these keywords with a high confidence score (e.g., score > 95).
    * **Management:** This produces a `politic_subreddit.csv` file, classifying subreddits by their likely ideology, which will be merged with the main dataset.

3.  **Sport-Subreddit Mapping:**
    * **Source:** We generated this from df_countries_expanded, since we were looking for subreddits linked to a sport that interacted with a subreddit linked to a country, as detailed in `filter_sports.py`.
    * **Process:**
        * We defined a comprehensive dictionary of keywords associated with different sports
        * Using the fuzzy-matching functions, we searched for subreddits whose names match these keywords with a high confidence score
    * **Management:** This produces a `df_countries_sport.csv` file, a dataset with all the posts where one interacting subreddit is linked to a country and the other is linked to a sport.

4.  **Subreddit Embeddings:**
    * **Source:** We used the pre-computed subreddit embeddings provided as a side of the main dataset (`load_embeddings`).
    * **Process:** These embeddings will be used as features for our clustering and visualization methods.

## Methods

1.  **Data Ingestion & Preparation:**
    * Load and merge the title and body hyperlink datasets.
    * Load our generated mappings and merge them with the main data, tagging source and target subreddits with an adequate label.

2.  **Clustering & Visualization:**
    * Apply K-Means clustering (using the elbow-plot method from `plotting.py` to find optimal *k*) on the subreddit embeddings to identify "natural" community clusters.
    * Visualize these clusters using t-SNE.
    * Generate a final network visualization (using `matplotlib`/`networkx` from `plotting.py`) showing the signed, aggregated network of country and political interactions.
    
3.  **Interaction effect**
    * Calculate the 7-day conditional reciprocity probability for inter-country interactions (B-->A | A-->B), defining the initiator (A) based on the first-ever post.
    * Apply the same 7-day reciprocity logic to interactions between different subreddits within the same country.
    *  Test for stylistic mirroring by statistically comparing the LIWC/VADER cosine similarity of reciprocal pairs against a random baseline.

4.  **Community & Interaction Analysis:**
    * Merge topic related subreddits to countries to find patterns.
    * Use TIMESTAMP to enrich with timewise analysis
    * Use the `utilis.py` functions to create a summary DataFrame of interactions.

5.  **Network Construction (NetworkX):**
    * As developed in `network_analysis.ipynb`, construct a directed graph using `networkx` where nodes are subreddits and directed edges represent hyperlinks.
    * The post properties (sentiment, LIWC features) from the dataset will be attached as attributes to each edge.
    * Create a "signed network" by computing the net sentiment of interactions between any two nodes, as planned in `plotting.py`.
    * Calculate shortest paths between key national communities.



## Proposed Timeline

* **Week 10/11 - 16/11:** Data cleaning and finalization of community mappings.
    * **Milestone:** Final `country_subreddit.csv`, `politic_subreddit.csv` and `sport_subreddit.csv` files generated and validated.
* **Week 17/11 - 23/11:** Working on the Homework on ADA course
    * **Milestone:** 
* **Week 24/11 - 30/11:** Network construction and aggregate network analysis and sentiment weighting. Finalising the Homework.
    * **Milestone:** Full NetworkX graph object built, saved, and basic properties (degree, centrality) calculated.
* **Week 1/12 - 7/12:** Final analysis and results compilation.
    * **Milestone:** All research questions addressed. Final project report/notebook used to answer the questions drafted.
* **Week 8/12 - 14/12:** Start working on the final website. Graphical work.
    * **Milestone:** Implementing all the plots to explain the results of the notebook.
* **Week 15/12 - 17/12:** Final website development. Cleaning the Github repository and the code with adequate comments
    * **Milestone (P3):** Finishing the project.

## Current workload division within the team

* **Julie: Data & Country Mapping Lead**
    * **Tasks:** Cleaned the dataset to obtain a clear mapping coutry-subreddit. Runs and refines the country mapping (`make_initial_subreddit_maps.ipynb`). Analysis of one-to-one country subreddit interactions
    * **Milestone:** Delivers the final, cleaned `subreddit_matches_approved.csv`.

* **Dario: Cluster and Community analysis Lead**
    * **Tasks:**  Political ideology filtering in `filter_politic_subreddits.ipynb`. Performed K-Means clustering and t-SNE dimensionality reduction. Facion analysis using positive interaction between communities and evolution of the factions over time. Handling plotting for factions and networks illustrations. Writing of the Readme.
    * **Milestone:** Delivers the final `politic_subreddit.csv` and analysis in `result.ipynb`.

* **Noemi: Chain of Interactions and Sport analysis Lead**
    * **Tasks:** Sport subreddits filtering in `filter_sports.ipynb`. Delivered final, deterministic probabilities for global and intra-country reciprocity. Completed statistical analysis proving/disproving linguistic style mirroring. Generated the final top_sport_per_country.csv analysis.
    * **Milestone:** Delivers the final `df_countries_sport.csv` and analysis in `result.ipynb`.

* **Simon: Network Analysis Lead**
    * **Tasks:** Responsible for building the core `networkx` graph from the hyperlink data (`network_analysis.ipynb`). Calculates graph-theoretic properties (centrality, degree, paths).
    * **Milestone:** Delivers the final, saved graph object and a notebook with basic network statistics.

* **Pietro:**
    * **Tasks:** 
    * **Milestone:** 