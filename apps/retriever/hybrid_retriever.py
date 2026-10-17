from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from ingestion.loader import load_document
from chunking.splitter import text_splitter
from dotenv import load_dotenv
from embeddings.embedder import embedder
from vector_store.store import vector_store
from llm.ollama_model import qwen
import os


load_dotenv()
path='ROOT/ai-document-intelligence/data/document/hdfc_ergo.pdf'
documents=load_document(path)
documents=text_splitter(documents)
bm25=BM25Retriever.from_documents(documents=documents, k=3)
db=vector_store(embedder)

dense_retriever=MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs={"k":10}),
    llm=qwen
)
ensemble=EnsembleRetriever(
    retrievers=[bm25,dense_retriever],
    weights=[0.5,0.5]
)

def hybrid_search(query:str):
    return ensemble.invoke(query)

# query='What is the sum insured opted under my Optima Secure policy?'
# response=hybrid_search(query)
# for doc in dense_retriever.invoke(query)[:5]:
#     print(doc.page_content)
#     print('='*80)
# print("-"*80)
# for doc in documents[:5]:
#     print(doc.page_content)
#     print('='*80)

