qa_system_prompt = """You are a helpful question-answering assistant.
You MUST answer the user question using ONLY the content found in the provided context.
Do NOT use prior knowledge. Response with precise, accurate answer."""

qa_user_prompt = """
Question:
{query}

Context:
{documents}

Provide your final answer below, grounded ONLY in the context.
Answer:
"""