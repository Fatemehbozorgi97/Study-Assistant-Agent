from langchain_community.llms import Ollama

llm = Ollama(
    model="hermes3:8b",
    base_url="http://localhost:11435"
)