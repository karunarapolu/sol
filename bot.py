import os
import streamlit as st
import sqlite3
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from google.api_core.exceptions import ResourceExhausted
import uuid
import re


# ======================================================
# Initialize conversation + Gemini key/token state
# ======================================================

MAX_TOKENS_DEFAULT = 100_000
GET_KEY_URL = "https://aistudio.google.com/app/apikey"
GET_KEY_LINK_MD = f"Don't have an API key? [Get one here]({GET_KEY_URL})"

_defaults = {
    "conversation_id": str(uuid.uuid4()),
    "title": "New Chat",
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "using_env_key": True,
    "limit_reached": False,
}
for _key, _value in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


def render_key_form(location, key_prefix):
    with location.form(key=f"{key_prefix}_key_form"):
        new_key = st.text_input("Enter your Gemini API key", type="password")
        st.caption(GET_KEY_LINK_MD)
        submitted = st.form_submit_button("Save & Continue")
        if submitted and new_key:
            st.session_state.gemini_api_key = new_key
            st.session_state.using_env_key = False
            st.session_state.limit_reached = False
            st.rerun()


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

# No env key AND no user-submitted key yet -> first-run gate
if not st.session_state.gemini_api_key:
    st.set_page_config(page_title="Sol", layout="wide")
    st.title("Sol - GSoC RAG Chatbot")
    _, center, _ = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            st.subheader("Enter your Gemini API key to get started")
            render_key_form(st, "welcome")
    st.stop()


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(page_title="Sol", layout="wide")
st.title("Sol - GSoC RAG Chatbot")

#FIX: define user_query
user_query = st.chat_input("Ask your GSoC question...")

# ======================================================
# Sidebar: key status, token usage, change-key control
# ======================================================

with st.sidebar:
    st.subheader("API Key")
    st.info("Using System Free Key" if st.session_state.using_env_key else "Using Custom Key")
    with st.expander("Change / Update API Key"):
        render_key_form(st, "sidebar")


# ======================================================
# Quota banner
# ======================================================

if st.session_state.limit_reached:
    st.warning("Free Tier API quota reached. Please enter your own Gemini API key to continue.")
    render_key_form(st, "banner")
    st.stop()


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
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)
qdrant_client.create_payload_index(
    collection_name="GSoC_Data1",
    field_name="text",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        scoring=True
    )
)

# ======================================================
# Vector store
# ======================================================

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="GSoC_Data1",
    embedding=embedding_model,
    content_payload_key="text",
    validate_collection_config=False
)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

if user_query and st.session_state.title == "New Chat":
    st.session_state.title = user_query[:50]



def hybrid_search(query, top_k=10):
    dense_docs = retriever.invoke(query)
    dense_ids = {str(doc.metadata["_id"]) for doc in dense_docs}

    print("\n========== DENSE RETRIEVAL ==========")

    for i, doc in enumerate(dense_docs, 1):
         print(f"\n--- DENSE CHUNK {i} ---")
         print(f"ID: {doc.metadata['_id']}")
         print(doc.page_content)
    query = re.sub(r"[^\w\s]", "", query)
    print(query)
    sparse_results = qdrant_client.query_points(
    collection_name="GSoC_Data1",
    query=models.TextQuery(text=query),
    using="text",
    limit=10,
    with_payload=True
     ).points
    q_ids = {str(doc.id) for doc in sparse_results}
    

    for rank, doc in enumerate(sparse_results, 1):
     print(f"\nRank: {rank}")
     print(f"ID: {doc.id}")
     print(f"Score: {doc.score}")
     print(f"Text: {doc.payload['text']}")

    all_ids = dense_ids | q_ids
    all_docs = [
    doc for doc in dense_docs
    if str(doc.metadata["_id"]) in all_ids
     ]

    all_docs += [
    doc for doc in sparse_results
    if str(doc.id) in all_ids
    and str(doc.id) not in dense_ids
    ]
    print("Dense IDs:", dense_ids)
    print("Sparse IDs:", q_ids)
    all_docs_ids = [
      str(doc.metadata["_id"]) if hasattr(doc, "metadata") else str(doc.id)
      for doc in all_docs
     ]
    return all_docs

# ======================================================
# Gemini LLM
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=st.session_state.gemini_api_key,
)


# ======================================================
# Prompt
# ======================================================

prompt_template = """
You are a retrieval-augmented chatbot specialized in Google Summer of Code (GSoC).

CRITICAL RULE:
You must ONLY use information explicitly present in the context or the provided conversation history. 
If the answer is not fully supported by either, you MUST say so.
The retrieved context may contain both relevant and irrelevant chunks. Review all retrieved chunks,
identify those that directly contribute to answering the user's query, and filter out unrelated or redundant information.
Rules:
- No guessing
- No assumptions
- No external knowledge beyond context/history
- No filling gaps
- Always ground every part of your answer in the provided context and history

Response requirements:
- Provide a clear, complete, and detailed answer that includes ALL relevant information from the context and history.
- pay attention to the history and refer to the history,respond with respect to the history when there is one.
- Organize the answer logically, covering every important point mentioned.
- Use concise, professional language that is accurate and easy to follow.
- When possible, structure the answer into sections or bullet points for readability.
- Structure the answer according to standard GSoC structure and include name of the contributor, mentors, technologies, project details, year, organization and every other detail given.

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
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else doc.payload["text"]
        for doc in docs
    )

def get_recent_history(conv_id, n=10):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    title TEXT,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
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
def hybrid_with_history(q):
    retrieved_docs = hybrid_search(q)
    return retrieved_docs

hybrid_search_runnable = RunnableLambda(hybrid_with_history)
format_docs_runnable = RunnableLambda(lambda docs: format_docs(docs))

rag_chain = (
    {
        "context": hybrid_search_runnable | format_docs_runnable,
        "question": RunnablePassthrough(),
        # FIX: use current conversation id
        "history": RunnableLambda(get_history_safe)
    }
    | prompt
    | llm
    | StrOutputParser()
)

def run_rag_chain(query: str):
    """Invoke the chain, catch 429/quota errors, update token counter
    from the model's own usage_metadata."""
    try:
        message = rag_chain.invoke(query)
    except ResourceExhausted:
        st.session_state.limit_reached = True
        st.rerun()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.session_state.limit_reached = True
            st.rerun()
        raise

    return message

# ======================================================
# Ensure Session State Exists
# ======================================================

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "title" not in st.session_state:
    st.session_state.title = "New Chat"


# ======================================================
# Sidebar Chats
# ======================================================

#st.sidebar.title("Chats")

#conn = sqlite3.connect("chat_history.db")
#cursor = conn.cursor()
#cursor.execute("SELECT DISTINCT conversation_id, title FROM chat_history ORDER BY timestamp DESC")
#conversations = cursor.fetchall()
#conn.close()

#for conv_id, title in conversations:
#    if st.sidebar.button(title, key=f"chat_{conv_id}"):
#        st.session_state.conversation_id = conv_id
#        st.session_state.title = title
#        st.rerun()





# ======================================================
# Chat execution (RUN FIRST)
# ======================================================

if user_query:
    with st.spinner("Retrieving grounded answer..."):
        answer = run_rag_chain(user_query)

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

    #CRITICAL FIX — forces rerun so history shows
    st.rerun()


# ======================================================
# Continuous Chat Rendering (RUN AFTER EXECUTION)
# ======================================================

if "conversation_id" in st.session_state:
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        title TEXT,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
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

#if st.sidebar.button("Clear History"):
#    conn = sqlite3.connect("chat_history.db")
#    cursor = conn.cursor()
#    cursor.execute("DELETE FROM chat_history")
#    conn.commit()
#    conn.close()
#    st.sidebar.success("Chat history cleared!")
#    st.rerun()

#if st.sidebar.button("New Chat"):
#    st.session_state.conversation_id = str(uuid.uuid4())
#    st.session_state.title = "New Chat"
#    st.rerun()


# ======================================================
# Close Qdrant
# ======================================================

qdrant_client.close()
