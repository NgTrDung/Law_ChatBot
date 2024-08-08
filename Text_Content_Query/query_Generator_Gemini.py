import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

MODEL_GEMINI = os.getenv('MODEL_GEMINI')

def query_generator(original_query: str, key_manager) -> list[str]:
    """Generate queries from original query"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that generates multiple search queries based on a single query."),
            ("human", f"Generate multiple search queries related to: {original_query}. When creating queries, please refine or add closely related contextual information without significantly altering the original query's meaning. OUTPUT (3 queries)."),
            ("human", "Generate (3 queries)"),
        ]
    )
    model = ChatGoogleGenerativeAI(
        google_api_key=key_manager.get_next_key(),
        model=MODEL_GEMINI,
        temperature=0
    )
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )
    queries = query_generator_chain.invoke({})
    return queries