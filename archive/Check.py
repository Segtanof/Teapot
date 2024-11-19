from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.groq import Groq

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama3-8b-8192", api_key= "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"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama3-8b-8192", api_key= "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", name="Groq", provider="Groq"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

team = Agent(team=[web_agent, finance_agent], show_tool_calls=True, markdown=True)

finance_agent.print_response("give me a list of 10 stocks with analyst recommendationn of do not buy or sell")