import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo

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
test_df = merged_df.head(10)

#pass that to agent
for i in range(len(test_df)):
    agent_role = Agent(
        name= "random",
        role= test_df.iloc[i]['occupation_name'],
        model=Groq(id="mixtral-8x7b-32768", api_key= "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"),
        #tools=[DuckDuckGo()],
        instructions=[f"This is who you are: {test_df.iloc[i]['occupation_description']}"],
        show_tool_calls=True,
        markdown=True,)
    
    print(f"{test_df.iloc[i]["occupation_name"]}: {agent_role.print_response("tell me a standard operating procedure of one of your tasks. list it in point form as an instruction.")}")


    
#get the result
