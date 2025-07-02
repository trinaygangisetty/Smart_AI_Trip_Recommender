import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

# Load cleaned data
members_df = pd.read_csv("members.csv")
trips_df = pd.read_csv("trips.csv")

# Join on member_id
merged_df = trips_df.merge(members_df, on="member_id")

# Aggregate trip details
def summarize_trips(group):
    member_id = group['member_id'].iloc[0]
    name = group['name'].iloc[0].title()
    age = int(group['age'].iloc[0])
    preference = group['preference'].iloc[0]
    num_trips = group.shape[0]
    destinations = group['destination'].unique().tolist()
    
    avg_cost = group['cost_usd'].replace('[\$,]', '', regex=True).astype(float).mean()
    
    activities = group['activities'].dropna().astype(str).str.split('[,;/]').sum()
    common_acts = Counter([a.strip() for a in activities if a.strip() != '']).most_common(3)
    common_acts = [act for act, _ in common_acts]
    
    return f"{name} (age {age}, prefers {preference}) has taken {num_trips} trips to {', '.join(destinations)}. Their average trip cost is ${int(avg_cost):,}. Favorite activities include: {', '.join(common_acts)}."

# Group by member and generate summaries
summaries = merged_df.groupby("member_id").apply(summarize_trips).tolist()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(summaries, show_progress_bar=True)

# Save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save
df_embed = pd.DataFrame({
    "member_id": merged_df.groupby("member_id").first().index,
    "summary": summaries,
    "embedding": embeddings.tolist()
})

df_embed.to_pickle("trip_history_embeddings.pkl")
faiss.write_index(index, "faiss_trip_history.index")

print("Trip history embeddings saved.")
