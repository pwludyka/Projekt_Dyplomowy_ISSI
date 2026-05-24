import ollama

LLM_MODEL = "gemma2:2b"


def generate_narrative(context: str) -> str:
    prompt = f"""
    You are a clinical study report medical writing assistant.

    Your task: Generate the complete CSR narrative for the chapter "Demographics".

    Critical rules:
    - Use only the retrieved context below.
    - Do not invent, estimate, round, recalculate or modify any number.
    - Do not use bullet points.
    - Do not repeat the same variable.
    - Write oen short paragraph per treatment arm.
    - For categorical variables, summarize the main distribution.
    - For numeric variables, report n, mean, SD, median and range.
    - For every numerical statement used in narrative, immediately after the number add a cell_id as in the context retrieved. Format: "mean age of 58.1 [[...]]".
    - Include cell_id references in parentheses after factual numerical claims.
    - If a value is not present in the context, do not mention it.
    - Write in formal, concise CSR style.
    - Return only the final narrative.

    Retrieved context:
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