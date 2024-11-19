from phi.agent import Agent
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app
from fastapi import FastAPI
from phi.model.groq import Groq

web_agent = Agent(
    name="Web Agent",
    model = Groq(id="llama3-8b-8192", api_key="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"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model = Groq(id="llama3-8b-8192", api_key="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"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

app = Playground(agents=[finance_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)