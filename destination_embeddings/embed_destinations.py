import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("destinations.csv")  

df["text"] = df.apply(lambda row: (
    f"{row['destination']} in {row['country']} is great for {row['tags']} travelers. "
    f"Average cost: ${row['avg_cost_usd']} - Safety rating: {row['safety_rating']}"
), axis=1)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

df["embedding"] = embeddings.tolist()
df.to_pickle("destination_embeddings.pkl")
faiss.write_index(index, "faiss_destinations.index")

print("Local embeddings generated and saved.")