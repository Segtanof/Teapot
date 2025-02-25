import pandas as pd
import json

#for llm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

#similarity
import regex as re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

#visualization
#plot matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#counting
from tqdm import tqdm

# %%
job_statements = pd.read_excel("datasets/task_statements.xlsx")
job_statements.columns = job_statements.columns.str.lower()
job_statements = job_statements.drop(labels=["incumbents responding","date","domain source"], axis=1).rename(columns={"o*net-soc code":"code", "task type":"type", "task id": "id", "task":"ref_task"})
job_statements = job_statements[~job_statements["type"].str.contains("Supplemental", case=False, na=True)]
job_statements["ind"] = job_statements["code"].str[:2]
job_statements = job_statements.groupby("title").agg({"ref_task":list, "ind": "first"}).reset_index().sort_values("ind")
sampled_occupation = job_statements.groupby('ind', group_keys=False).sample(frac=0.05, random_state=1) #43 samples
sampled_occupation

# %%
#for trial
trial_df = sampled_occupation
sampled_list =[trial_df.iloc[x]["title"] for x in range(len(trial_df))]

# %% [markdown]
# ### Set up functions

# %%
#get reference description
def get_des (title):
    task_list = sampled_occupation.query("title == @title")["ref_task"].iloc[0]
    return task_list

# %%
def task_gen(title,model, system = None):
    class occupation(BaseModel):
        occupation: str
        tasks: list[str]

    #initialize model
    model= model

    query = "Generate "+str(len(get_des(title)))+" task statements that "+ title +" would perform at work."

    if system == None:
        prompt_template = ChatPromptTemplate([
            ("human","{input}")
            ]
        )
    else:
        prompt_template = ChatPromptTemplate([
            ("system", system),
            ("human","{input}")
            ]
        )

    structured_llm = model.with_structured_output(schema=occupation.model_json_schema())

    prompt = prompt_template.invoke({"input": query, "title": title})
    response = structured_llm.invoke(prompt)

    return response

model = ChatOllama(model="llama3.2", temperature=1, seed= 1)

#ask user to input and save it as the variable system
system = input("")

# %%
for title in tqdm(sampled_list):
    generated_statements = task_gen(title, model, system)
    trial_df.loc[trial_df["title"] == title, "gen_task"] = [generated_statements]

trial_df