from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    base_url="http://127.0.0.1:11434"

)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)