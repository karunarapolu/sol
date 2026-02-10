import os
import streamlit as st
import sqlite3
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
import uuid


# ======================================================
# Initialize conversation
# ======================================================

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.title = "New Chat"


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

st.set_page_config(page_title="GSoC Hybrid RAG Chatbot", layout="wide")
st.title("GSoC Hybrid RAG Chatbot")

# ✅ FIX: define user_query
user_query = st.chat_input("Ask your GSoC question...")


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

if user_query and st.session_state.title == "New Chat":
    st.session_state.title = user_query[:50]


# ======================================================
# BM25 keyword search setup
# ======================================================

all_docs = qdrant_client.scroll(
    collection_name="GSoC_Data1",
    limit=10000
)[0]

corpus = [doc.payload["text"] for doc in all_docs]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


def hybrid_search(query, top_k=10):
    dense_docs = retriever.invoke(query)

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_results = [
        {"text": corpus[i], "score": bm25_scores[i]}
        for i in range(len(corpus))
    ]
    bm25_results = sorted(bm25_results, key=lambda x: x["score"], reverse=True)[:top_k]

    combined = []
    for doc in dense_docs:
        text = doc.page_content
        bm25_score = next((r["score"] for r in bm25_results if r["text"] == text), 0)
        combined.append((text, bm25_score))

    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

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
# Prompt
# ======================================================

prompt_template = """
You are a retrieval-augmented chatbot specialized in Google Summer of Code (GSoC).

CRITICAL RULE:
You must ONLY use information explicitly present in the retrieved context or the provided conversation history. 
If the answer is not fully supported by either, you MUST say so.

Rules:
- No guessing
- No assumptions
- No external knowledge beyond context/history
- No filling gaps
- Always ground every part of your answer in the provided context and history

Response requirements:
- Provide a clear, complete, and detailed answer that includes ALL relevant information from the context and history.
- Organize the answer logically, covering every important point mentioned.
- Use concise, professional language that is accurate and easy to follow.
- If multiple pieces of relevant information exist, synthesize them into a cohesive answer rather than listing them separately.
- Highlight key details (such as dates, eligibility criteria, processes, or examples) so the user receives a comprehensive response.
- When possible, structure the answer into sections or bullet points for readability.

Fallbacks:
- If insufficient context/history exists, respond with:
  "The provided context and history do not contain sufficient information to answer this question. Please try rephrasing your query with more specific details.
When falling back, always suggest at least two improved versions of the user’s question that would yield better results.Ensure suggestions are specific, contextual 
and make the question more broad to increase the scope of search.
- If the question is off-topic, respond EXACTLY with:
  "This question is outside the scope of this GSoC-focused chatbot. Please ask a question related to Google Summer of Code.


Conversation history:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "history"],
)


# ======================================================
# Helpers
# ======================================================

def format_docs(docs):
    return "\n\n".join(docs)


def get_recent_history(conv_id, n=10):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history WHERE conversation_id=? ORDER BY id ASC LIMIT ?",
        (conv_id, n)
    )
    rows = cursor.fetchall()
    conn.close()
    history = "\n".join([f"User: {u}\nBot: {b}" for u, b in rows])
    return history

def get_history_safe(_):
    conv_id = st.session_state.get("conversation_id", "")
    return get_recent_history(conv_id)


# ======================================================
# RAG Chain
# ======================================================

hybrid_search_runnable = RunnableLambda(lambda q: hybrid_search(q))
format_docs_runnable = RunnableLambda(lambda docs: format_docs(docs))

rag_chain = (
    {
        "context": hybrid_search_runnable | format_docs_runnable,
        "question": RunnablePassthrough(),
        # ✅ FIX: use current conversation id
        "history": RunnableLambda(get_history_safe)
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ======================================================
# Ensure Session State Exists
# ======================================================

st.session_state.setdefault("conversation_id", str(uuid.uuid4()))
st.session_state.setdefault("title", "New Chat")


# ======================================================
# Sidebar Chats
# ======================================================

st.sidebar.title("Chats")

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT conversation_id, title FROM chat_history ORDER BY timestamp DESC")
conversations = cursor.fetchall()
conn.close()

for conv_id, title in conversations:
    if st.sidebar.button(title, key=f"chat_{conv_id}"):
        st.session_state.conversation_id = conv_id
        st.session_state.title = title
        st.rerun()





# ======================================================
# Chat execution (RUN FIRST)
# ======================================================

if user_query:
    with st.spinner("Retrieving grounded answer..."):
        docs = retriever.invoke(user_query)
        answer = rag_chain.invoke(user_query)

    if st.session_state.title == "New Chat":
        st.session_state.title = user_query[:50]

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (conversation_id, title, user_message, bot_response) VALUES (?, ?, ?, ?)",
        (st.session_state.conversation_id, st.session_state.title, user_query, answer)
    )
    conn.commit()
    conn.close()

    # 🔴 CRITICAL FIX — forces rerun so history shows
    st.rerun()


# ======================================================
# Continuous Chat Rendering (RUN AFTER EXECUTION)
# ======================================================

if "conversation_id" in st.session_state:
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history WHERE conversation_id=? ORDER BY id ASC",
        (st.session_state.conversation_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        with st.chat_message("user"):
            st.markdown(row[0])
        with st.chat_message("assistant"):
            st.markdown(row[1])


# ======================================================
# Sidebar Controls
# ======================================================

if st.sidebar.button("Clear History"):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    st.sidebar.success("Chat history cleared!")
    st.rerun()

if st.sidebar.button("New Chat"):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.title = "New Chat"
    st.rerun()


# ======================================================
# Close Qdrant
# ======================================================

qdrant_client.close()
