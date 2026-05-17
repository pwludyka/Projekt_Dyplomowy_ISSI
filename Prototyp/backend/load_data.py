import ollama
from pathlib import Path
import pandas as pd
import json
import numpy as np
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
CSV_PATH = BASE_DIR / "Data_Files" / "patients.csv"

def read_csv(csv_path: Path) -> pd.DataFrame:
    with pd.read_csv(csv_path) as df:
        return df
    

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
            text = {
                f"{variable} in {arm_label} arm: "
                f"n={values['n']}, mean={values['mean']}, SD={values['sd']}, median={values['median']}, "
                f"Q1={values['q1']}, Q3={values['q3']}, range={values['min']}-{values['max']}."
            }

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