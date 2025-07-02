from flask import Blueprint, render_template, request
from vertexai.generative_models import GenerativeModel
import vertexai

rag_chat_bp = Blueprint("rag_chat", __name__)

# Initialize Vertex AI (already authenticated with service account)
vertexai.init(project="trip-recommendation-project", location="us-central1")
gemini = GenerativeModel("gemini-pro")

@rag_chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    prompt = None
    response = None

    if request.method == "POST":
        prompt = request.form["prompt"]

        result = gemini.generate_content(prompt)
        response = result.text

    return render_template("chat.html", prompt=prompt, response=response)
