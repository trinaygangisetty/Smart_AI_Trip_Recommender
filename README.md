# Smart AI Trip Recommender

This repository contains a comprehensive trip recommendation system built using Streamlit, BigQuery, FAISS vector search, and large language models (LLMs) via OpenAI. It offers the following features:

* **SQL‑based recommendations via BigQuery**
* **RAG (Destination Tags)** – uses embeddings + FAISS for similarity matching
* **RAG (Trip History Match)** – finds similar travelers
* **Weather‑Aware Destination Insights** – combines embeddings with weather data
* **Chat Assistant** – allows users to upload CSVs and ask questions using LLM with vector retrieval

---

## Project Structure

```
/
├── All Documentation Files/                    
│   ├── App Screen shots/                        # Visuals of different Streamlit app features
│   │   ├── Streamlit Big query Recommendations.pdf
│   │   ├── Streamlit Destination Tags.pdf
│   │   ├── Streamlit Trip History match.pdf
│   │   ├── Streamlit open AI.pdf
│   │   ├── Streamlit weather aware destination insights.pdf
│   ├── Architecture Diagram/                    # Visuals explaining app architecture and pipeline
│   │   ├── Trip Recommendation Architecture.jpg
│   ├── Data sets/
│   │   ├── Cleaned Data Sets/                   # Final preprocessed data used for loading into BigQuery and embeddings
│   │   │   ├── destinations_clean.csv
│   │   │   ├── members_clean.csv
│   │   │   ├── trips_clean.csv
│   │   │   ├── weather_clean.csv
│   │   ├── Un cleaned Data Sets/               # Raw synthetic data before cleaning and preprocessing
│   │   │   ├── destinations.csv
│   │   │   ├── members.csv
│   │   │   ├── trips.csv
│   │   │   ├── weather.csv
│   ├── Notebooks/                              # Data generation and cleaning logic written in Jupyter
│   │   ├── Trip-recommendation-project-data-generation.ipynb
│   │   ├── Trip_Data_Cleaning.ipynb
│   ├── Write ups/                              # Explanation documents prepared for interviews or collaborators
│   │   ├── Architecture Overview _ Pipeline Description.pdf
│   │   ├── Data Set Description.pdf
│   │   ├── How to Run the AI Trip Recommender Streamlit App Locally.pdf
│   │   ├── Trip Recommendation Project_ System Architecture and RAG Pipeline Write-up (1).pdf

├── Datasets/
│   ├── destinations.csv                         # Main destination metadata used in both BigQuery and embeddings
│   ├── members.csv                              # Traveler profiles with age, interests, and budget
│   ├── trips.csv                                # Historical trip records mapped to members
│   ├── weather.csv                              # Monthly weather and seasonal ratings per destination

├── Flask App/
│   ├── app.py                                   # Legacy or optional Flask entry point (replaced by Streamlit)
│   ├── export_routes.py                         # Export logic for PDF/CSV generation
│   ├── rag_chat_routes.py                       # RAG logic for CSV-based LLM Q&A
│   ├── rag_history_routes.py                    # Endpoint to get recommendations using trip history
│   ├── rag_routes.py                            # RAG-based destination match using tags

├── destination_embeddings/
│   ├── destination_embeddings.pkl               # Pre-computed destination tag embeddings (Hugging Face vectors)
│   ├── embed_destinations.py                    # Script to generate destination tag embeddings
│   ├── faiss_destinations.index                 # FAISS index for fast nearest-neighbor search on tags
│   ├── test_embeddings.py                       # Utility to test loading or searching the embeddings

├── destination_weather_embeddings/
│   ├── destination_weather_embeddings.pkl       # Embeddings combining destination tags and weather
│   ├── embed_destinations_weather.py            # Script to generate these weather-aware embeddings
│   ├── faiss_destinations_weather.index         # FAISS index for semantic weather-wise search

├── notebooks/
│   ├── Trip-recommendation-project (1).ipynb    # Possibly renamed version of main notebook
│   ├── Trip_Data_Cleaning_GCP.ipynb             # Used for cleaning data before loading to BigQuery

├── templates/                                   # Jinja HTML templates (used only in Flask, not Streamlit)
│   ├── chat.html
│   ├── index.html
│   ├── rag.html
│   ├── rag_history.html

├── trip_member_embeddings/
│   ├── embed_trip_histories.py                  # Script to embed member trip summaries
│   ├── faiss_trip_history.index                 # FAISS index for searching similar trip profiles
│   ├── trip_history_embeddings.pkl              # Embedding vectors for trip history + metadata

├── .gcloudignore                                # Files to ignore during Google App Engine deploy
├── .gitignore                                   # Files ignored by Git (e.g., .env, __pycache__)
├── Dockerfile                                   # Optional Docker setup for local deployment
├── README.md                                    # Project overview, setup, and usage instructions
├── app.yaml                                     # Google App Engine config (if deploying via gcloud CLI)
├── requirements.txt                             # All Python packages needed to run the app
└── trip_app_combined.py                         # Main Streamlit dashboard with all feature routes
                              
```

Key CSV files and resulting BigQuery tables:

| CSV File                  | Table Name            |
| ------------------------- | --------------------- |
| `destinations.csv`        | `destinations`        |
| `weather.csv`             | `weather`             |
| `destination_weather.csv` | `destination_weather` |
| `trip_histories.csv`      | `trip_histories`      |

---

##  Setup & Local Run

1. **Clone repo & create venv**

   ```bash
   git clone https://github.com/trinaygangisetty/Smart_AI_Trip_Recommender.git
   cd Smart_AI_Trip_Recommender
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create `secrets.toml`** in `.streamlit/`:
   *Include your service account JSON credentials properly on multiple lines for `private_key`.*

   ```toml
   OPENAI_API_KEY = "sk-..."

   [gcp_service_account]
   type = ...
   project_id = ...
   private_key = """-----BEGIN PRIVATE KEY-----
   ...
   -----END PRIVATE KEY-----"""
   client_email = ...
   ...
   ```

4. **Upload your CSVs to BigQuery**, preserving table names above.

5. **Run the app** locally with:

   ```bash
   streamlit run trip_app_combined.py
   ```

---

##  Deployment with Streamlit Cloud

1. Connect your GitHub repo in Streamlit Cloud.
2. Set secrets (OPENAI\_API\_KEY, GCP credentials).
3. Trigger automatic deploy; if any build errors, check version compatibility in `requirements.txt`.

---

##  Future Enhancements

1. Refactor and modularize the codebase into logical directories.
2. Explore advanced embeddings like OpenAI’s embeddings or fine-tuned transformer models.
3. Deploy on Streamlit Community Cloud or GCP App Engine and set up CI/CD pipelines.

---

##  Architecture Overview

### 1. BigQuery + SQL Insights

* **Data**: destinations + weather per month
* **Process**: run parameterized SQL queries for “Top Picks”, “Hidden Gems”, etc.
* **Output**: tabular insights filtered by age, budget, preference, month

### 2. RAG Pipelines (Destinations & History)

* **Embeddings**: using `sentence-transformers/all-MiniLM-L6-v2`
* **Index**: FAISS indices store vector representations
* **Query**: user's description is embedded, searched against FAISS, and matched

### 3. Weather‑Aware Insights

* Extends destination embeddings with weather metadata
* Provides smart hints (e.g., “perfect weather”, “low safety”, “great value”)

### 4. LLM Chat

* Users upload CSVs, which are chunked and embedded
* FAISS retrieves relevant chunks, prepends to prompt
* Sent to OpenAI `gpt-3.5-turbo` for context-aware answers

---

##  Why Hugging Face Transformers?

* Fast, lightweight embeddings with good semantic matching
* Easy integration via `sentence-transformers`
* Free and offline—not restricted like paid APIs

---

##  Data & RAG Pipeline Workflow

1. **Data prep** in Jupyter/Colab from raw CSVs
2. **Pickle embeddings** and build FAISS indices
3. **Load indices in Streamlit** using `@st.cache_resource`
4. **Apply RAG**:

   * user input → embed → FAISS search → retrieve nearest entries
   * output displayed with explanatory texts

---

##  Additional Notes

* Stores separate embeddings for destination+weather and trip history
* BigQuery handles pre-computed insights with SQL flexibility
* Streamlit UI unifies all features under one dashboard
