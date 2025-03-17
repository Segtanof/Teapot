from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    base_url="http://10.0.3.228:11434"

)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)