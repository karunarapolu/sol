import os
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI


# Environment checks

if not os.getenv("QDRANT_API_KEY"):
    st.error("QDRANT_API_KEY not set")
    st.stop()

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not set")
    st.stop()


# Streamlit UI

st.set_page_config(page_title="GSoC RAG Chatbot", layout="centered")
st.title("GSoC RAG Chatbot (Strict, No Hallucinations)")

user_query = st.text_input("Ask a GSoC-related question:")


# Embedding model

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Qdrant Client

qdrant_client = QdrantClient(
    url="https://43935042-f3be-4bcb-8c25-5f9417548ccc.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)


# Vector Store (LangChain wrapper)

vector_store = Qdrant(
    client=qdrant_client,
    collection_name="GSoC_Data",
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)


# Gemini LLM (LangChain wrapper)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# Strict No-Hallucination Prompt

prompt_template = """
You are a retrieval-augmented chatbot specialized in Google Summer of Code (GSoC).
You MUST operate in a strictly grounded mode.

CRITICAL RULE:
You are ONLY allowed to use information explicitly present in the retrieved context.
If the answer is not fully supported by the context, you MUST say so clearly.
DO NOT guess, infer, assume, extrapolate, or use prior knowledge.

Follow these rules without exception:

1. Zero Hallucination Policy:
   - Do NOT add facts, examples, explanations, or interpretations not present in the context.
   - Do NOT combine partial information to form new conclusions.
   - Do NOT fill gaps with general GSoC knowledge.

2. Allowed Scope ONLY:
   Answer questions strictly related to:
   - GSoC organizations listed in the context
   - GSoC project descriptions present in the context
   - Required skills or technologies explicitly mentioned
   - Contribution or application details explicitly stated

3. Insufficient Context Handling:
   If the retrieved context does not contain enough information to answer the question,
   respond EXACTLY with:
   "The provided context does not contain sufficient information to answer this question."

4. Off-Topic Queries:
   If the question is unrelated to GSoC or open-source projects,
   respond EXACTLY with:
   "This question is outside the scope of this GSoC-focused chatbot."

5. No External Knowledge:
   - Do NOT rely on training data, memory, or general understanding.
   - Treat the retrieved context as the ONLY source of truth.

6. Precision Over Helpfulness:
   - Accuracy is more important than completeness.

7. Response Style:
   - Use concise, factual sentences.
   - No introductions, no conclusions, no sign-offs.

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

custom_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# RetrievalQA Chain

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)


# Query Execution

if user_query:
    with st.spinner("Retrieving grounded answer..."):
        result = rag_chain({"query": user_query})

    if not result["source_documents"]:
        st.write(
            "The provided context does not contain sufficient information to answer this question."
        )
    else:
        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("Retrieved Context (Debug)"):
            for i, doc in enumerate(result["source_documents"], start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)
