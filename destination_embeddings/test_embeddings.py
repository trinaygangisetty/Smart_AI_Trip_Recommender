from vertexai.language_models import TextEmbeddingModel
import vertexai

vertexai.init(project="trip-recommendation-project", location="us-west4")
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")

embeddings = model.get_embeddings(["Trip to Paris", "Luxury resort in Maldives"])
for embedding in embeddings:
    print(embedding.values[:5])  # print first 5 dims
