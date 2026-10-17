from langchain_pymupdf4llm import PyMuPDF4LLMLoader
def load_document(file):
    loader=PyMuPDF4LLMLoader(file_path=file,
                             mode='page')
    return loader.load()

