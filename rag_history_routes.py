from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

rag_history_bp = Blueprint("rag_history", __name__)

# Load trip history data + FAISS index
df = pd.read_pickle("trip_member_embeddings/trip_history_embeddings.pkl")
index = faiss.read_index("trip_member_embeddings/faiss_trip_history.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

@rag_history_bp.route("/rag-history", methods=["GET", "POST"])
def rag_history():
    recommendations = None
    query_text = None

    if request.method == "POST":
        age = request.form.get("age")
        budget = request.form.get("budget")
        interest = request.form.get("interest")

        query_text = f"A {age}-year-old traveler interested in {interest} with a budget of ${budget}."
        query_vector = model.encode([query_text])
        distances, indices = index.search(np.array(query_vector).astype("float32"), 5)

        results = df.iloc[indices[0]].copy()
        results["score"] = distances[0]

        recommendations = results

    return render_template(
        "rag_history.html",
        query=query_text,
        recommendations=recommendations
    )
