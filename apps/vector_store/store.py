from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import os
load_dotenv()
def vector_store_with_documents(documents:List[Document],embedder):
    vec_str=Chroma(
        collection_name='vec_embeddings',
        embedding_function=embedder,
        persist_directory=os.getenv('VEC_DB_PATH')
    )
    vec_str.add_documents(documents)
    return vec_str

def vector_store(embedder):
    return Chroma(
        collection_name='vec_embeddings',
        embedding_function=embedder,
        persist_directory=os.getenv('VEC_DB_PATH')
    )