from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url="https://43935042-f3be-4bcb-8c25-5f9417548ccc.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

query = "NumFocus 2025"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="GSoC_Data",
    query=query_vector,
    with_payload=True,
    limit=5
).points


for r in results:
    print(r)
