from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.load_data import CSV_PATH, read_csv, create_summary_table
from backend.RAG import setup_milvus_collection, index_summary_table, retrieve_context
from backend.generate_narrative import generate_narrative

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = None
summary_df = None
milvus_client = None


@app.get("/")
def root():
    return {"message": "CSR MVP Backend działa"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/demographics/load-embed")
def load_embed_demographics():
    global df, summary_df, milvus_client

    df = read_csv(CSV_PATH)
    summary_df = create_summary_table(df)

    milvus_client = setup_milvus_collection()
    index_summary_table(milvus_client, summary_df)

    return {
        "message": "Dane załadowane i embeddingi utworzone",
        "rows_loaded": len(df),
        "summary_records": len(summary_df),
        "table_preview": df.head(20).to_dict(orient="records")
    }

@app.post("/api/demographics/generate-narrative")
def generate_demographics_narrative():
    query = "Generate CSR demographics narrative."
    context = retrieve_context(milvus_client, query)
    narrative = generate_narrative(context)

    return {
        "query": query,
        "context": context,
        "narrative": narrative
    }


