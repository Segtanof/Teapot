# %%
#for loading data
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

from datetime import datetime
import os

# Generate the folder name with current date and time
folder_name = 'results/task_match_'+datetime.now().strftime("%d%m_%H%M")

# Create the folder if it does not exist
os.makedirs(folder_name, exist_ok=True)

# %%
job_statements = pd.read_excel("datasets/task_statements.xlsx")
job_statements.columns = job_statements.columns.str.lower()
job_statements = job_statements.drop(labels=["incumbents responding","date","domain source"], axis=1).rename(columns={"o*net-soc code":"code", "task type":"type", "task id": "id", "task":"ref_task"})
job_statements = job_statements[~job_statements["type"].str.contains("Supplemental", case=False, na=True)]
job_statements["ind"] = job_statements["code"].str[:2]
job_statements = job_statements.groupby("title").agg({"ref_task":list, "ind": "first"}).reset_index().sort_values("ind")
sampled_occupation = job_statements.groupby('ind', group_keys=False).sample(frac=0.01, random_state=1) #43 samples
sampled_occupation

# %%
#for trial
trial_df = sampled_occupation
sampled_list =[trial_df.iloc[x]["title"] for x in range(len(trial_df))]
sampled_list


# %%
#get reference description
def get_des (title):
    task_list = sampled_occupation.query("title == @title")["ref_task"].iloc[0]
    return task_list

# %%
def task_gen(title,model, system = None):
    class occupation(BaseModel):
        '''Name the occupation and list out the tasks that the occupation would perform at work.'''
        occupation: str
        tasks: list[str]

    #initialize model
    model= model

    query = "List out exactly "+str(len(get_des(title)))+" tasks that the occupation \""+ title +"\" would perform at work. Make sure each statement is unique and different from one another."

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

    structured_llm = model.with_structured_output(schema=occupation, method='json_schema')

    prompt = prompt_template.invoke({"input": query, "title": title})
    # keep running until the number of parsed tasks is equal to the number of reference tasks
    while True:
        response = structured_llm.invoke(prompt)
        #parse response
        try:
            parsed = json.loads(response["tasks"])
        except:
            print('not json')
            try:
               parsed = response["tasks"]

            except:
                print('not string')
                continue
        try:
            if len(parsed) >= 0.8 * len(get_des(title)):
                return parsed
            else:
                print('not equal, parsed:', len(parsed), 'ref:', len(get_des(title)))
                continue
        except Exception as e:
            print(e)
            continue
    


# %%
#pre process text
def preProcessText(text=list):
	processed = []
	for doc in text:
		doc = re.sub(r"\\n", "", doc)
		doc = re.sub(r"\W", " ", doc) #remove non words char
		doc = re.sub(r"\d"," ", doc) #remove digits char
		doc = re.sub(r'\s+[a-z]\s+', " ", doc) # remove a single char
		doc = re.sub(r'^[a-z]\s+', "", doc) #remove a single character at the start of a document
		doc = re.sub(r'\s+', " ", doc)  #replace an extra space with a single space
		doc = re.sub(r'^\s', "", doc) # remove space at the start of a doc
		doc = re.sub(r'\s$', "", doc) # remove space at the end of a document
		processed.append(doc.lower())
	return processed

# %%
#get similarity score
def sbert(ref, gen):
    sim_model = SentenceTransformer("all-mpnet-base-v2", similarity_fn_name="cosine")

    # Compute embeddings for both lists
    embeddings_ref = sim_model.encode(ref)
    embeddings_gen = sim_model.encode(gen)


    # Compute cosine similarities
    similarities = sim_model.similarity(embeddings_ref, embeddings_gen).numpy()
    return similarities


# %%
def match(ref, gen):
    try:
        ref_clean = preProcessText(ref)
        gen_clean = preProcessText(gen)
        matrix = sbert(ref_clean, gen_clean)
        row_ind, col_ind = linear_sum_assignment(1 - matrix)  # Minimize cost (1 - similarity)
        assigned_similarities = matrix[row_ind, col_ind]
        return np.mean(assigned_similarities), matrix, row_ind.tolist(), col_ind.tolist()
    except:
        return np.nan


# %%
model = ChatOllama(model="granite3.2", temperature=1)

#ask user to input and save it as the variable system
system = "your occupation is {title}. Respond with the knowledge of the occupation."

# %%
for title in tqdm(sampled_list):
    generated_statements = task_gen(title, model)
    trial_df.loc[trial_df["title"] == title, "gen_task"] = pd.Series([generated_statements]).values
trial_df

# %%
result_df = trial_df.reset_index(drop=True)


# %%
result_df[["score", "matrix", "ref_order", "gen_order"]] = result_df.apply(lambda row: match(row["ref_task"], row["gen_task"]), axis=1).apply(pd.Series)
result_df

# %%
with open(folder_name + '/sys_prompt1.json', 'w') as f:
    f.write(result_df.to_json(index=True))

with open(folder_name + '/sys_prompts.txt', 'w') as f:
    f.write(system)

mean = result_df["score"].mean()
with open(folder_name + '/score.json', 'w') as f:
    f.write(mean)

