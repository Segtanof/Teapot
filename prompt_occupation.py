import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq
import random

#import occupation description
df_occupation = pd.read_csv('occupation.txt', sep='\t')
df_occupation = df_occupation.dropna(subset=["occupation_description"])
df_occupation = df_occupation[["occupation_code", "occupation_name","occupation_description"]]
df_occupation = df_occupation.rename(columns={"occupation_code": "OCC_CODE"})
#print(df_occupation.head())

#import wage stat
df_wage = pd.read_excel("testocc.xlsx")
df_wage = df_wage[["OCC_CODE", "A_MEAN", "O_GROUP"]]
df_wage['OCC_CODE'] = df_wage['OCC_CODE'].str.replace('-', '').astype(int)
df_wage = df_wage[df_wage['O_GROUP'] == "detailed"]
#print(df_wage.head())

#join both dfs together
merged_df = pd.merge(df_occupation, df_wage, on='OCC_CODE', how='inner').drop(columns=["O_GROUP"])
print(merged_df.head())

#testing the idea
test_df = merged_df.head(5)

l1 = list(merged_df["occupation_name"])

aaa = random.choice(l1)
bbb = random.choice(l1)
ccc = random.choice(l1)

#pass that to agent
#for i in range(len(test_df)):
agent_role1 = Agent(
    name= aaa,
    role= aaa,
    model=Groq(id="mixtral-8x7b-32768", api_key= "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"),
    #tools=[DuckDuckGo()],
    instructions=[f"This is who you are: {test_df.iloc[1]['occupation_description']}"],
    show_tool_calls=True,
    markdown=True,)

agent_role2 = Agent(
    name= test_df.iloc[2]['occupation_name'],
    role= test_df.iloc[2]['occupation_name'],
    model=Groq(id="mixtral-8x7b-32768", api_key= "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"),
    #tools=[DuckDuckGo()],
    instructions=[f"This is who you are: {test_df.iloc[2]['occupation_description']}"],
    show_tool_calls=True,
    markdown=True,)

agent_role3 = Agent(
    name= test_df.iloc[3]['occupation_name'],
    role= test_df.iloc[3]['occupation_name'],
    model=Groq(id="mixtral-8x7b-32768", api_key = "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"),
    #tools=[DuckDuckGo()],
    instructions=[f"This is who you are: {test_df.iloc[3]['occupation_description']}"],
    show_tool_calls=True,
    markdown=True,)
    
agent_team = Agent(
    team=[agent_role1, agent_role2, agent_role3],
    model= Groq(id="llama-3.1-70b-versatile", api_key = "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"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

print(agent_team.print_response("""youre a team. please show discussion among you all on this topic: "How should we resolve conflicts?"""))

#get the result
