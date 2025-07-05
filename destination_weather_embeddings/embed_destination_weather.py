import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load destination and weather data
dest_df = pd.read_csv(r"..\Datasets\destinations.csv")
weather_df = pd.read_csv(r"..\Datasets\weather.csv")  # Should contain destination, month, weather, seasonal_rating

# Merge destination and weather (one row per destination + month)
merged = pd.merge(dest_df, weather_df, on="destination")

# Create rich semantic description
merged["text"] = merged.apply(lambda row: (
    f"{row['destination']} in {row['country']} is ideal for {row['tags']} travelers. "
    f"In {row['month']}, it typically experiences {row['weather']} with a seasonal rating of {row['seasonal_rating']}. "
    f"The average cost is ${row['avg_cost_usd']}, and the safety rating is {row['safety_rating']}."
), axis=1)

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(merged["text"].tolist(), show_progress_bar=True)

# Save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save to file
merged["embedding"] = embeddings.tolist()
merged.to_pickle("destination_weather_embeddings.pkl")
faiss.write_index(index, "faiss_destinations_weather.index")

print("Weather-aware destination embeddings saved.")
