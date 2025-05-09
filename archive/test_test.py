from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()

    model = ChatOllama(model="mistral", base_url=f"http://127.0.0.1:{args.port}")
    # model = ChatOllama(model=args.model, base_url=f"http://127.0.0.1:{args.port}")

    query = "hello there"
    response = model.invoke(query)
    print(response)

if __name__ == "__main__":
    main()