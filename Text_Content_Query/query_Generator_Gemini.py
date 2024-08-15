import os
import sys
import importlib

from operator import itemgetter
from langchain.load import dumps, loads
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
# from Utils import response_custom
from qdrant_client import QdrantClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
response_custom = importlib.import_module("Utils.response_Custome")

load_dotenv()

MODEL_GEMINI = os.getenv("MODEL_GEMINI")
URL_QDRANT = os.getenv("URL_QDRANT")
API_QDRANT = os.getenv("API_QDRANT")
COLLECTION_NAME_COMMON_DB_TINH_HUONG = os.getenv("COLLECTION_NAME_COMMON_DB_TINH_HUONG")
CONTENT_PAYLOAD_KEY = os.getenv("CONTENT_PAYLOAD_KEY")
METADATA_PAYLOAD_KEY = os.getenv("METADATA_PAYLOAD_KEY")

TOP_K = 5
MAX_DOCS_FOR_CONTEXT = 8

my_template_prompt = """
    Please answer the [question] using only the following [information]. 
    If there is no [information] available to answer the question. 

    Information: {context}
    Question: {question}
    Final answer:
"""

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a Qdrant collection exists"""
    collections = client.get_collections().collections
    return any(col.name == collection_name for col in collections)

from langchain_community.embeddings import GPT4AllEmbeddings

def existing_collection(collection_name: str) -> Qdrant:
    """Create vector retriever"""
    client = QdrantClient(url = URL_QDRANT, api_key = API_QDRANT)
    if not collection_exists(client, collection_name):
        return None

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}

    gpt4all_embedding = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
        )

    doc_store = Qdrant.from_existing_collection(
        url=URL_QDRANT,
        embedding=gpt4all_embedding,
        collection_name=collection_name,    
        content_payload_key=CONTENT_PAYLOAD_KEY,
        metadata_payload_key=METADATA_PAYLOAD_KEY,
        api_key=API_QDRANT
    )
    return doc_store

def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (reciprocal rank fusion)"""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]

# def query_generator(original_query: dict, key_manager) -> list[str]:
#     """Generate queries from original query"""
#     query = original_query.get("query")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant that generates multiple search queries based on a single query."),
#             ("human", "Generate 3 search queries related to: {original_query}. Provide each query on a new line, and ensure that only the queries are returned, with no additional explanations or text."),
#         ]
#     )

#     model = ChatGoogleGenerativeAI(
#         google_api_key=key_manager.get_next_key(),
#         model=MODEL_GEMINI,
#         temperature=0
#     )

#     query_generator_chain = (
#         prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
#     )

#     queries = query_generator_chain.invoke({"original_query": query})
#     return queries

def query_generator(original_query: str, key_manager) -> list[str]:
    """Generate queries from original query"""
    # Sử dụng trực tiếp original_query như là câu truy vấn
    query = original_query
    
    # Cập nhật prompt để yêu cầu rõ ràng chỉ trả về 3 câu truy vấn
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that generates multiple search queries based on a single query."),
            ("human", "Generate 3 search queries related to: {original_query}. Provide each query on a new line, and ensure that only the queries are returned, with no additional explanations or text."),
        ]
    )
    
    model = ChatGoogleGenerativeAI(
        google_api_key=key_manager.get_next_key(),
        model=MODEL_GEMINI,
        temperature=0
    )
    
    query_generator_chain = (
        prompt | model | StrOutputParser()
    )
    
    # Kết quả sẽ là một chuỗi các câu truy vấn cách nhau bằng dấu xuống dòng
    result = query_generator_chain.invoke({"original_query": query})
    
    # Tách kết quả thành danh sách các câu truy vấn
    queries = result.strip().split('\n')
    
    return queries

# def similarity_search(para: dict) -> list[Document]:
#     """RRF retriever"""
#     common_doc_store = existing_collection(COLLECTION_NAME_COMMON_DB_TINH_HUONG)
#     # user_doc_store = existing_collection(para["user_id"])
#     queries = query_generator({"query": para["query"]}, para["key_manager"])
    
#     all_results = []
#     for q in queries:
#         if common_doc_store:
#             common_results = common_doc_store.similarity_search_with_score(q, k=TOP_K)
#             all_results.append(common_results)
#         # if user_doc_store:
#         #     user_results = user_doc_store.similarity_search_with_score(q, k=TOP_K_GOV_CONTENT)
#         #     all_results.append(user_results)
    
#     fused_results = reciprocal_rank_fusion(all_results)
#     print(fused_results)
#     return fused_results

def similarity_search(original_query: str, key_manager) -> list[Document]:
    """RRF retriever"""
    common_doc_store = existing_collection(COLLECTION_NAME_COMMON_DB_TINH_HUONG)
    # user_doc_store = existing_collection(para["user_id"])
    
    # Gọi hàm query_generator với chuỗi truy vấn
    queries = query_generator(original_query, key_manager)
    
    all_results = []
    for q in queries:
        if common_doc_store:
            common_results = common_doc_store.similarity_search_with_score(q, k=TOP_K)
            all_results.append(common_results)
        # if user_doc_store:
        #     user_results = user_doc_store.similarity_search_with_score(q, k=TOP_K_GOV_CONTENT)
        #     all_results.append(user_results)
    
    fused_results = reciprocal_rank_fusion(all_results)
    print(fused_results)
    return fused_results

# def query(query: str, key_manager):
#     """Query with vector db"""
#     # filter_json = af_GEMINI.get_filter_json(query, key_manager)
#     # filter_json = af_GPT.get_filter_json(query)
#     filter_json = {}
#     # print(filter_json)
#     model = ChatGoogleGenerativeAI(
#         google_api_key=key_manager.get_next_key(),
#         model=MODEL_GEMINI,
#         temperature=0
#     )

#     prompt = PromptTemplate(
#         template=my_template_prompt,
#         input_variables=["context", "question"]
#     )

#     ss = RunnableLambda(similarity_search)
#     chain = (
#         {
#             "context": RunnableLambda(lambda ip: ss.invoke({"query": ip['query'], "filter": ip['filter'], "key_manager": ip['key_manager']})), 
#             "question": itemgetter("query")
#         }
#         | RunnablePassthrough.assign(
#             context=itemgetter("context")
#         )
#         | {
#             "response": prompt | model | StrOutputParser(), 
#             "context": itemgetter("context")
#         }
#     )

#     result = chain.invoke({"query": query, "filter": filter_json, "key_manager": key_manager})
#     return result

def query(original_query: str, key_manager):
    """Query with vector db"""
    filter_json = {}  # Giữ nguyên hoặc thay đổi tùy thuộc vào yêu cầu

    model = ChatGoogleGenerativeAI(
        google_api_key=key_manager.get_next_key(),
        model=MODEL_GEMINI,
        temperature=0
    )

    prompt = PromptTemplate(
        template=my_template_prompt,
        input_variables=["context", "question"]
    )

    # Thay đổi cách gọi hàm similarity_search để phù hợp với đầu vào mới
    ss = RunnableLambda(lambda ip: similarity_search(original_query, key_manager))

    chain = (
        {
            "context": ss,  # Truyền context là kết quả từ similarity_search
            "question": itemgetter("query")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt | model | StrOutputParser(),
            "context": itemgetter("context")
        }
    )

    result = chain.invoke({"query": original_query, "filter": filter_json, "key_manager": key_manager})
    return result

def get_documents(user_query, key_manager):
    """Get documents function"""

    answer = query(user_query, key_manager)
    if answer:
        answers = [answer['response']]
        return response_custom.response_success(answers)

    return response_custom.response_failed()