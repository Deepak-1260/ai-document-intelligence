from llm.ollama_model import qwen
from embeddings.embedder import embedder
from vector_store.store import vector_store
from rag.chain import ollama_chain
from retriever.hybrid_retriever import hybrid_search
from reranking.reranker import rerank_documents
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
print("Please enter your query:")
question=input()

# db=vector_store(embedder)
compressor=LLMChainExtractor.from_llm(
    llm=qwen
)

docs=hybrid_search(query=question)
ranked_docs=rerank_documents(documents=docs, query=question, top_n=3)
only_docs=[doc[0] for doc in ranked_docs]
compress_doc=compressor.compress_documents(documents=only_docs,query=question)
context='\n'.join([doc.page_content for doc in compress_doc])
print(f"context:{context} \n question:{question}")
response= ollama_chain.invoke({"topic":f"context:{context} \n question:{question}"})
print(response.content)




