from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
from sentence_transformers import SentenceTransformer

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set")

with open("merged_output.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]
collectionName = "GSoC_Data"
model = SentenceTransformer('all-MiniLM-L6-v2')


qdrant = QdrantClient(
    url = "https://43935042-f3be-4bcb-8c25-5f9417548ccc.europe-west3-0.gcp.cloud.qdrant.io",
    api_key = QDRANT_API_KEY
)
def embed(text):
    response = model.encode(text)
    response = response.tolist()
    return response

def create_collection():
    print("\n Qdrant collection...")
    qdrant.create_collection(collection_name = collectionName,vectors_config = VectorParams(
        size = 384,
        distance = Distance.COSINE
    ))

def save_chunk_to_qdrant(chunk,chunk_id):
    vector = embed(chunk)

    qdrant.upsert(
        collection_name = collectionName,
        points = [
            PointStruct(
                id = chunk_id,
                vector = vector,
                payload ={
                    "text": chunk,
                    "chunk_id": chunk_id,
                })
        ]
    )
def createEmbeds():
    create_collection()
    global_id = 1
    for i in range(0,len(texts)):
        save_chunk_to_qdrant(texts[i],global_id)
        print(f"Saved chunk {global_id}")
        global_id += 1
if __name__ ==  "__main__":
    createEmbeds()


