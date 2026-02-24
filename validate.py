from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

query = "NumFocus 2025"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="GSoC_Data1",
    query=query_vector,
    with_payload=True,
    limit=5
).points


for r in results:
    print(r)
