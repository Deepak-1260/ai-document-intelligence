from langchain_core.prompts import ChatPromptTemplate
from llm.ollama_model import qwen
from llm.model import model
system_message="You are a help assistant, who answer all my questions from provided context. If don't have enough knowledge about  it,you need to reply I don't have enough info"
human_message="{topic}"

pmpt=ChatPromptTemplate(
    [
        ("system",system_message),
        ("human",human_message)
    ]
)
ollama_chain=pmpt|qwen
google_chain=pmpt|model



