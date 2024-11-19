#from phi.agent import Agent
#from phi.tools.arxiv_toolkit import ArxivToolkit
#from phi.model.groq import Groq

"""agent = Agent(
    name = "search_agent",
    model=Groq(id="llama-3.1-70b-versatile", 
               api_key= "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"), 
    tools=[ArxivToolkit()], 
    show_tool_calls=True)

agent.print_response("Search arxiv for 'LLM agent roleplay'", markdown=True)
"""
import ollama
response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])