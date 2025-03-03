# %%
#for loading data
import pandas as pd
import json
import requests

#for llm
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

#counting
from tqdm import tqdm

#logging
import pickle
import regex as re
import os
from datetime import datetime


# Generate the folder name with current date and time
folder_name = 'results/job_match_'+datetime.now().strftime("%d%m_%H%M")

# Create the folder if it does not exist
os.makedirs(folder_name, exist_ok=True)

# set up occupation data
occupations = pd.read_excel('datasets/occupation_data.xlsx').dropna()

occupations.columns = occupations.columns.str.lower()
#rename the column 
occupations = occupations.rename(columns={'o*net-soc code':'code'})   
#drop rows with "all other" in the content
occupations = occupations[~occupations['title'].str.contains("All Other")]
#change data type
occupations['code'] = occupations['code'].astype(str)
occupations['title'] = occupations['title'].astype(str)
occupations['description'] = occupations['description'].astype(str)

occupations["ind"] = occupations["code"].str[:2]
sampled_occupation = occupations.groupby('ind', group_keys=False).sample(frac=0.05, random_state=1) #47 samples

test_sample = sampled_occupation.sample(5, random_state=1)

occupation_dict = {sampled_occupation["title"].iloc[x]: sampled_occupation["code"].iloc[x]for x in range(len(sampled_occupation))}
test_dict = {test_sample["title"].iloc[x]: test_sample["code"].iloc[x]for x in range(len(test_sample))}

# %%
json.dump(test_dict, open(folder_name + '/test_dict.json', 'w'))
# %%
#get the questions into a list
with open("datasets/60qs.json") as f:
    qs = json.load(f)
test = qs["questions"]["question"]
df = pd.DataFrame(test)[['text', 'area', '_index']]
df.columns = ['question', 'area', 'index']
qlist = list(df["question"])


# %% [markdown]
# ### Set up related occupation data
#whole dataset
related = pd.read_excel('datasets/related_occupations.xlsx').astype(str)
related.columns = related.columns.str.lower().str.replace(" ","_").str.replace("o*net-soc_", "")
related = related[related["relatedness_tier"] == "Primary-Short"]



# %% [markdown]
# ### API call score function

# %%
#access the api to get the job titles
def get_career(answer):

    url = 'https://services.onetcenter.org/ws/mnm/interestprofiler/careers?answers='+answer+'&start=1&end=1000'
    cookies = {
        'developer_login': 'dW5pX21hbm5oZWltX2RlMTowMDU1ODEyOTFiYzRjYTYxNGE5YmJlM2E4ZjgyNjk2NWQxNzFiY2Y0',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        'Authorization': 'Basic dW5pX21hbm5oZWltX2RlMTo3MzM5Y3R1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Priority': 'u=0, i',
    }

    params = {
        'start': '1',
        'end': '60',
    }

    response = requests.get(
        url,
        params=params,
        cookies=cookies,
        headers=headers,
    )
    #search for the the target occupation in the response
    data = json.loads(response.text)["career"]
    #select only title and fit
    career = pd.DataFrame(data).drop(["href", "tags"], axis=1)
    return career 

# %%
#match suggested career to the related career
def matching_score(code, career):
    #get direct match
    best_fit = career[career['fit'] == 'Best']
    if sum(code == best_fit["code"]) == 1:
        print(1)
    else:
        #get related match
        related_occ = related.query('code == @code & (relatedness_tier == "Primary-Short" | relatedness_tier == "Primary-Long")')
        related_occ = related_occ.merge(career,how="inner", left_on= "related_code", right_on="code")
        print (len(related_occ)/10)
        return related_occ




# %%
def get_rating(title, model, code, system = None ):
    class rating(BaseModel):
            do_you_like_it: str
            provide_thoughts: str
            your_rating: int

    chat_history = []
    to_parse=[]
    query = "think about your persona and rate the statement with a number between 1 to 5 depending on your interest. 1 is strongly dislike and 5 is strongly like."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "you are an experienced {name}. follow the instruction given to you, but think and feel like a {name}'",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{input}"
            )
        ]
    )

    structured_llm = model.with_structured_output(schema=rating.model_json_schema())
    for q in tqdm(qlist):
        prompt = prompt_template.invoke({"name": title, "chat_history" : chat_history[-10:], "input": query + q})
        response = structured_llm.invoke(prompt)
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(str(response)))
        to_parse.append(response["your_rating"])
    
    
    with open(folder_name + '/' + code + '.pkl', 'wb') as f:
        pickle.dump(chat_history, f)
    answer = ''.join(map(str, to_parse))
    return answer


# %%
test_model= ChatOllama(model="llama3.1", temperature=1)

for title, code in test_dict.items():
    answer = get_rating(title, test_model, code)
    career = get_career(answer)

    with open(folder_name + '/' + code + '.json', 'w') as f:
        f.write(career.to_json(index=True))

