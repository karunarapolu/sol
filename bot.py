#import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from langchain.prompts import PromptTemplate
#from langchain.chains import RetrievalQA
from google.genai import Client
import warnings, os, sys


model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url="https://43935042-f3be-4bcb-8c25-5f9417548ccc.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

gemini_client = Client(api_key="")
response = gemini_client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Analyze this text and summarize it: ...'
)
print(response.text)