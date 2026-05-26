import ollama

LLM_MODEL = "gemma2:2b"


def generate_narrative(context: str) -> str:
    prompt = f"""
    You are a deterministic clinical study report medical writing engine.

    Your task:
    Generate the CSR Demographics narrative using ONLY the provided structured source records.

    MANDATORY OUTPUT RULES:

    1. Use ONLY information explicitly present in SOURCE RECORDS.
    2. NEVER invent numbers.
    3. NEVER estimate, summarize from memory, round, recalculate, or infer missing values.
    4. EVERY numeric value MUST be followed immediately by its exact source citation.
    5. Citation format MUST be exactly:

    [cell_id]

    Example:
    58.01 [14.1.1_age_DrugX_numeric_summary_1]

    Example:
    9.59 [14.1.1_age_DrugX_numeric_summary_1]

    6. If multiple numbers come from the same source, repeat citation after EACH number.
    7. DO NOT omit citations.
    8. DO NOT change citation format.
    9. DO NOT use parentheses for citations.
    10. Output narrative prose only.
    11. No bullet points.
    12. One paragraph per treatment arm.
    13. Formal CSR style.

    SOURCE RECORDS:
    {context}
    """.strip()

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate factual and traceable CSR text using only retrieved source context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]