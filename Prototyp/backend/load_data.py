import ollama
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import pandas as pd
import json
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

SECTION = "Demographics"
TABLE_ID = "14.1.1"
COLLECTION_NAME = "csr_demographics_summary"

EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_LENGTH = 768

host = "localhost"
port = "19530"

ID_columns = ["patient_id"]
Categorical_columns = ["sex", "race", "ethnicity"]
Numerical_columns = ["age", "weight_kg", "height_cm", "bmi", "systolic_bp_mmhg", "diastolic_bp_mmhg",
                      "heart_rate_bpm", "lvef_percent", "disease_duration_years"]


@dataclass
class SummaryRecord:
    cell_id: str
    section: str
    table_id: str
    variable: str
    arm: str
    statistic_type: str
    text: str
    value_json: str


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "Data_Files" / "patients.csv"

def read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
    

def cell_id(table_id: str, variable: str, arm: str, statistic_type: str, index: int) -> str:
    return f"{table_id}_{variable}_{arm}_{statistic_type}_{index}"

def create_numeric_summary(df: pd.DataFrame) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    row_index = 1

    for arm, arm_df in df.groupby("arm"):
        arm_label = str(arm)
        for variable in Numerical_columns:
            series = pd.to_numeric(arm_df[variable])
            values = {
                "n": int(series.count()),
                "mean": round(series.mean(),2),
                "sd": round(series.std(ddof=1), 2),
                "median": round(series.median(), 2),
                "min": round(series.min(), 2),
                "max": round(series.max(), 2),
                "q1": round(series.quantile(0.25), 2),
                "q3": round(series.quantile(0.75), 2),
            }
            text = (
                f"{variable} in {arm_label} arm: "
                f"n={values['n']}, mean={values['mean']}, SD={values['sd']}, median={values['median']}, "
                f"Q1={values['q1']}, Q3={values['q3']}, range={values['min']}-{values['max']}."
            )

            records.append(
                SummaryRecord(
                    cell_id=cell_id(TABLE_ID, variable, arm_label, "numeric_summary", row_index),
                    section=SECTION,
                    table_id=TABLE_ID,
                    variable=variable,
                    arm=arm_label,
                    statistic_type="numeric_summary",
                    text=text,
                    value_json=json.dumps(values),
                )
            )
            row_index += 1

    return records

def create_categorical_summary(df: pd.DataFrame) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    row_index = 1

    for arm, arm_df in df.groupby("arm"):
        arm_label = str(arm)
        arm_n = len(arm_df)

        for variable in Categorical_columns:
            counts = arm_df[variable].value_counts()
            values = {level: {"n": int(count), "percent": round((count/arm_n)*100, 2)}
                      for level, count in counts.items() }
            
            levels_text = "; ".join(
                f"{level}: n={data['n']} ({data['percent']}%)" for level, data in values.items())
            
            text = f"{variable} distribution in {arm_label} arm: total n={arm_n}; {levels_text}."
            
            records.append(
                SummaryRecord(
                    cell_id=cell_id(TABLE_ID, variable, arm_label, "categorical_summary", row_index),
                    section=SECTION,
                    table_id=TABLE_ID,
                    variable=variable,
                    arm=arm_label,
                    statistic_type="categorical_summary",
                    text=text,
                    value_json=json.dumps(values),
                )
            )
            row_index += 1

    return records


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    records = create_numeric_summary(df) + create_categorical_summary(df)
    return pd.DataFrame([record.__dict__ for record in records])


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def setup_milvus_collection() -> MilvusClient:
    milvus_client = MilvusClient(host=host, port=port)

    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)

    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_LENGTH)
    section_field = FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=128)
    table_id_field = FieldSchema(name="table_id", dtype=DataType.VARCHAR, max_length=64)
    variable_field = FieldSchema(name="variable", dtype=DataType.VARCHAR, max_length=128)
    arm_field = FieldSchema(name="arm", dtype=DataType.VARCHAR, max_length=128)
    statistic_type_field = FieldSchema(name="statistic_type", dtype=DataType.VARCHAR, max_length=128)
    cell_id_field = FieldSchema(name="cell_id", dtype=DataType.VARCHAR, max_length=128)
    value_json_field = FieldSchema(name="value_json", dtype=DataType.VARCHAR, max_length=8192)

    fields = [id_field, text_field, embedding_field, table_id_field, section_field, variable_field, 
              arm_field, statistic_type_field, cell_id_field, value_json_field]
    
    schema = CollectionSchema(fields=fields, auto_id=False, enable_dynamic_field=True)

    milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="L2", params={"M": 4, "efConstruction": 64})

    milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

    milvus_client.load_collection(collection_name=COLLECTION_NAME)

    return milvus_client


def index_summary_table(
        milvus_client: MilvusClient,
        summary_df: pd.DataFrame) -> None:
    
    entities: list[dict[str, Any]] = []
    
    for id, row in summary_df.iterrows():
        text = str(row["text"])
        cell_id = str(row["cell_id"])
        entities.append(
            {
                "id": cell_id,
                "text": text,
                "embedding": embed_text(text),
                "table_id": str(row["table_id"]),
                "section": str(row["section"]),
                "variable": str(row["variable"]),
                "arm": str(row["arm"]),
                "statistic_type": str(row["statistic_type"]),
                "cell_id": str(row["cell_id"]),
                "value_json": str(row["value_json"]),
            }
        )

    milvus_client.insert(collection_name=COLLECTION_NAME, data=entities)

    milvus_client.flush(collection_name=COLLECTION_NAME)



def retrieve_context(
        milvus_client: MilvusClient,
        query: str,
        collection_name: str = COLLECTION_NAME,
        limit: int = 5) -> str:

    query_embedding = embed_text(query)

    results = milvus_client.search(
        collection_name=collection_name,
        data=[query_embedding],
        anns_field="embedding",
        limit=limit,
        filter='section == "Demographics" and table_id == "14.1.1"',
        output_fields=["text", "table_id", "section", "variable", "arm", "statistic_type", "cell_id", "value_json"],
        search_params={"metric_type": "L2", "params": {"ef": 32}}
    )

    context_lines = []
    for hit in results[0]:
        entity = hit["entity"]
        context_lines.append(
            "\n".join(
                [
                    f"Source cell_id: {entity['cell_id']}",
                    f"Table: {entity['table_id']}",
                    f"Section: {entity['section']}",
                    f"Variable: {entity['variable']}",
                    f"Arm: {entity['arm']}",
                    f"Statistic type: {entity['statistic_type']}",
                    f"Verified source data: {entity['text']}",
                ]
            )
        )

    return "\n\n---\n\n".join(context_lines)

def main():
    df = read_csv(CSV_PATH)
    summary_df = create_summary_table(df)
    summary_df.to_csv("demographics_summary.csv", index=False)

    milvus_client = setup_milvus_collection()

    index_summary_table(milvus_client, summary_df)

    print(f"Wczytano rekordów wejściowych: {len(df)}")
    print(f"Utworzono rekordów podsumowań: {len(summary_df)}")


if __name__ == "__main__":
    main()