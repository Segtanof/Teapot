from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.groq import Groq

agent = Agent(
    llm=Groq(model="mixtral-8x7b-32768", api_key="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"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
)
agent.print_response("Whats happening in France?", markdown=True, stream=False)