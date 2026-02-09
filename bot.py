import os
import streamlit as st

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings


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

st.set_page_config(page_title="GSoC RAG Chatbot", layout="centered")
st.title("GSoC RAG Chatbot (Strict · No Hallucinations)")

user_query = st.text_input("Ask a GSoC-related question:")


# ======================================================
# Embeddings (MUST match indexing)
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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ======================================================
# Gemini LLM
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ======================================================
# Strict no-hallucination prompt
# ======================================================

prompt_template = """
You are a retrieval-augmented chatbot specialized in Google Summer of Code (GSoC).

CRITICAL RULE:
You are ONLY allowed to use information explicitly present in the retrieved context.
If the answer is not fully supported by the context, you MUST say so.

Rules:
- No guessing
- No assumptions
- No external knowledge
- No filling gaps

If insufficient context exists, respond EXACTLY with:
"The provided context does not contain sufficient information to answer this question."

If off-topic, respond EXACTLY with:
"This question is outside the scope of this GSoC-focused chatbot."

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
    return "\n\n".join(doc.page_content for doc in docs)


# ======================================================
# LCEL RAG chain
# ======================================================

rag_chain = (
    {
        "context": retriever | format_docs,
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
