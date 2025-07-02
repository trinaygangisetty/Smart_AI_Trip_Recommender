from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

rag_bp = Blueprint("rag", __name__)

# Load data and FAISS index
df = pd.read_pickle("destination_embeddings/destination_embeddings.pkl")
index = faiss.read_index("destination_embeddings/faiss_destinations.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

@rag_bp.route("/rag", methods=["GET", "POST"])
def rag_search():
    recommendations = None
    query_text = None
    preference = None

    if request.method == "POST":
        # Form input
        age = int(request.form["age"])
        budget = int(request.form["budget"])
        preference = request.form["preference"]
        month = request.form["month"]

        # Step 1: Pre-filter dataset
        df_filtered = df[
            (df["avg_cost_usd"] <= budget) &
            (df["tags"].str.contains(preference, case=False, na=False))
        ]

        if df_filtered.empty:
            return render_template("rag.html", query="No matching destinations after filtering!", recommendations=[])

        # Step 2: Form semantic query
        query_text = (
            f"Suggest {preference} destinations for a {age}-year-old traveler "
            f"with a budget of ${budget} in {month}. Focus on safety and quality experience."
        )

        query_vector = model.encode([query_text])[0].astype("float32")

        # Step 3: Get top 5 semantic matches from filtered set
        filtered_vectors = np.array(df_filtered["embedding"].tolist()).astype("float32")
        sub_index = faiss.IndexFlatL2(query_vector.shape[0])
        sub_index.add(filtered_vectors)

        _, sub_indices = sub_index.search(np.array([query_vector]), k=min(5, len(df_filtered)))

        matched_rows = []
        for idx in sub_indices[0]:
            row = df_filtered.iloc[idx].copy()
            row["score"] = float(np.dot(query_vector, row["embedding"]) / (np.linalg.norm(query_vector) * np.linalg.norm(row["embedding"])))
            matched_rows.append(row)

        recommendations = pd.DataFrame(matched_rows)

    return render_template("rag.html", query=query_text, recommendations=recommendations, preference=preference)
