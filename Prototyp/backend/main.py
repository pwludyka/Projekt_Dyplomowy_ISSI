from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import ollama
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema

LLM_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "nomic-embed-text"

VECTOR_LENGTH = 768

host = "localhost"
port = "19530"

COLLECTION_NAME = "csr_demographics_single_record_demo"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend działa"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from backend"}




