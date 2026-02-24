import csv
import glob
import os
import time

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


# -------------------------
# CONFIG
# -------------------------
COLLECTION_NAME = "GSoC_Data1"
QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()

# SAFETY SETTINGS
BATCH_SIZE = 40            # smaller batches prevent payload errors
EMBED_BATCH_SIZE = 32
RETRY_ATTEMPTS = 3


# -------------------------
# INIT CLIENT
# -------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120  # prevents write timeout
)


# -------------------------
# CSV → ROW PARAGRAPH
# -------------------------
def row_to_paragraph(file_name, row_index, row):
    parts = [f"{key}={value}" for key, value in row.items() if value]
    return f"{file_name} | Row {row_index}: " + " | ".join(parts)


def load_csv_rows():
    texts = []

    for file_path in glob.glob("*.csv"):
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader, start=1):
                paragraph = row_to_paragraph(file_path, i, row)
                texts.append(paragraph)

    return texts


# -------------------------
# COLLECTION
# -------------------------
def create_collection_if_not_exists():
    collections = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in collections:
        print("Creating collection...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    else:
        print("Collection already exists.")


# -------------------------
# RESUME SUPPORT
# -------------------------
def get_existing_point_count():
    try:
        info = qdrant.count(collection_name=COLLECTION_NAME, exact=True)
        print(f"Existing points in collection: {info.count}")
        return info.count
    except Exception:
        return 0


# -------------------------
# UPSERT WITH RESUME + RETRY
# -------------------------
def upload_texts(texts):
    existing_count = get_existing_point_count()

    if existing_count >= len(texts):
        print("All data already uploaded.")
        return

    print(f"Resuming from index {existing_count}")

    global_id = existing_count + 1

    for i in range(existing_count, len(texts), BATCH_SIZE):

        batch = texts[i:i + BATCH_SIZE]

        vectors = model.encode(
            batch,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False
        )

        points = []

        for text, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=global_id,
                    vector=vector.tolist(),
                    payload={
                        "text": text,
                        "chunk_id": global_id
                    }
                )
            )
            global_id += 1

        print(f"Uploading batch starting at row {i}")

        for attempt in range(RETRY_ATTEMPTS):
            try:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                break
            except Exception as e:
                print(f"Retry {attempt+1} failed: {e}")
                time.sleep(5)

        time.sleep(0.5)

    print("Upload complete.")


# -------------------------
# MAIN
# -------------------------
def main():
    create_collection_if_not_exists()

    texts = load_csv_rows()
    upload_texts(texts)


if __name__ == "__main__":
    main()
