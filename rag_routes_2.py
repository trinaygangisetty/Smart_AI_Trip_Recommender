from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

rag_bp = Blueprint("rag", __name__)

# Load data and models
df = pd.read_pickle("destination_embeddings/destination_embeddings.pkl")
index = faiss.read_index("destination_embeddings/faiss_destinations.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

@rag_bp.route("/rag", methods=["GET", "POST"])
def rag_search():
    recommendations = None
    query_text = ""

    if request.method == "POST":
        age = int(request.form["age"])
        budget = int(request.form["budget"])
        preference = request.form["preference"]
        month = request.form["month"]

        query_text = (
            f"A {age}-year-old traveler with a budget of ${budget}, "
            f"preferring {preference} experiences in {month}"
        )

        query_embedding = model.encode([query_text])
        distances, indices = index.search(np.array(query_embedding).astype("float32"), k=5)

        matched_rows = df.iloc[indices[0]].copy()
        matched_rows["score"] = 1 - distances[0]  # similarity = 1 - distance
        recommendations = matched_rows.sort_values("score", ascending=False)

    return render_template("rag.html", query=query_text, recommendations=recommendations)
