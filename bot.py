import os
import streamlit as st

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

from rank_bm25 import BM25Okapi


# ======================================================
# SentenceTransformer → LangChain embedding wrapper
# ======================================================

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()


# ======================================================
# Environment checks
# ======================================================

if not os.getenv("QDRANT_API_KEY"):
    st.error("QDRANT_API_KEY not set")
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not set")
    st.stop()


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(page_title="GSoC Hybrid RAG Chatbot", layout="centered")
st.title("GSoC Hybrid RAG Chatbot (Dense + BM25 Rerank)")

user_query = st.text_input("Ask a GSoC-related question:")


# ======================================================
# Embeddings
# ======================================================

embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ======================================================
# Qdrant client
# ======================================================

qdrant_client = QdrantClient(
    url="https://43935042-f3be-4bcb-8c25-5f9417548ccc.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
)


# ======================================================
# Vector store
# ======================================================

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="GSoC_Data1",
    embedding=embedding_model,
    content_payload_key="text"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})


# ======================================================
# BM25 keyword search setup
# ======================================================

# Fetch all documents once (for BM25 corpus)
all_docs = qdrant_client.scroll(
    collection_name="GSoC_Data1",
    limit=10000
)[0]

corpus = [doc.payload["text"] for doc in all_docs]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


def hybrid_search(query, top_k=10):
    # Dense retrieval
    dense_docs = retriever.invoke(query)

    # Sparse retrieval (BM25)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Attach BM25 scores to docs
    bm25_results = [
        {"text": corpus[i], "score": bm25_scores[i]}
        for i in range(len(corpus))
    ]
    bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Fusion rerank: combine dense + BM25
    combined = []
    for doc in dense_docs:
        text = doc.page_content
        bm25_score = next((r["score"] for r in bm25_results if r["text"] == text), 0)
        combined.append((text, bm25_score))

    # Sort by BM25 score (rerank fusion)
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Return top reranked docs
    return [doc for doc, _ in combined_sorted[:top_k]]


# ======================================================
# Gemini LLM
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


# ======================================================
# Strict no-hallucination prompt
# ======================================================

prompt_template = """
You are a retrieval-augmented chatbot specialized in Google Summer of Code (GSoC).

CRITICAL RULE:
You must ONLY use information explicitly present in the retrieved context. 
If the answer is not fully supported by the context, you MUST say so.

Rules:
- No guessing
- No assumptions
- No external knowledge
- No filling gaps
- Always ground every part of your answer in the provided context

Response requirements:
- Provide a clear, complete, and detailed answer that includes ALL relevant information from the context.
- Organize the answer logically, covering every important point mentioned in the retrieved context.
- Use concise, professional language that is accurate and easy to follow.
- If multiple pieces of relevant information exist, synthesize them into a cohesive answer rather than listing them separately.
- Highlight key details (such as dates, eligibility criteria, processes, or examples) so the user receives a comprehensive response.
- When possible, structure the answer into sections or bullet points for readability.

Fallbacks:
- If insufficient context exists, respond EXACTLY with:
  "The provided context does not contain sufficient information to answer this question. Please try rephrasing your query with more specific details. For example: 'What were the eligibility rules for GSoC 2023?' or 'How does the student application process work in GSoC?'"
- If the question is off-topic, respond EXACTLY with:
  "This question is outside the scope of this GSoC-focused chatbot. Please ask a question related to Google Summer of Code. For example: 'What is the role of mentors in GSoC?' or 'How are organizations selected for GSoC?'"

Additionally:
- When falling back, always suggest at least two improved versions of the user’s question that would yield better results.
- Ensure suggestions are specific, contextual, and aligned with GSoC topics.

Context:
{context}

Question:
{question}

Answer:

"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)


# ======================================================
# Helper to format docs
# ======================================================

def format_docs(docs):
    return "\n\n".join(docs)


# ======================================================
# LCEL RAG chain
# ======================================================
hybrid_search_runnable = RunnableLambda(lambda q: hybrid_search(q)) 
format_docs_runnable = RunnableLambda(lambda docs: format_docs(docs))


rag_chain = (
    {
        "context": hybrid_search_runnable | format_docs_runnable,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ======================================================
# Query execution
# ======================================================

if user_query:
    with st.spinner("Retrieving grounded answer..."):
        docs = retriever.invoke(user_query)
        answer = rag_chain.invoke(user_query)

    if not docs:
        st.write(
            "The provided context does not contain sufficient information to answer this question."
        )
    else:
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context (Debug)"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)


qdrant_client.close()