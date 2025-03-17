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
import numpy as np

#counting
from tqdm import tqdm

from datetime import datetime
import os

# %%
# Generate the folder name with current date and time
folder_name = 'results/task_match_'+datetime.now().strftime("%d%m_%H%M")+"/"


# Create the folder if it does not exist
os.makedirs(folder_name, exist_ok=True)

# %% [markdown]
# ### Preprocess data and sampling

# %%
# read dataset and drop columns
job_statements = pd.read_excel("datasets/task_statements.xlsx")
job_statements.columns = job_statements.columns.str.lower()
job_statements = job_statements.drop(labels=["incumbents responding","date","domain source"], axis=1).rename(columns={"o*net-soc code":"code", "task type":"type", "task id": "id", "task":"ref_task"})
job_statements = job_statements[~job_statements["type"].str.contains("Supplemental", case=False, na=True)]
job_statements["ind"] = job_statements["code"].str[:2]
job_statements = job_statements.groupby("title").agg({"ref_task":list, "ind": "first"}).reset_index().sort_values("ind")
sampled_occupation = job_statements.groupby('ind', group_keys=False).sample(frac=0.05, random_state=1) #43 samples


# %%
#for trial
trial_df = sampled_occupation.sample(3, random_state= 1)
test_sample_list =[trial_df.iloc[x]["title"] for x in range(3)]
test_sample_list

# %% [markdown]
# ### Set up functions

# %%
#get reference description
def get_des (title):
    task_list = sampled_occupation.query("title == @title")["ref_task"].iloc[0]
    return task_list

# %%
#invoke llm to generate tasks
def task_gen(title, model, system = None):
    json_schema = {
        "type": "object",
        "properties": {
            "occupation": {
                "type": "string"
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": len(get_des(title)),
                "maxItems": len(get_des(title))
            }
        },
        "required": ["occupation", "tasks"]
    }

    #initialize model

    query = "List out exactly "+str(len(get_des(title)))+" task statements that the occupation \""+ title +"\" would perform at work.Make sure each statement is unique and different from one another."

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

    llm = model.with_structured_output(schema=json_schema, method="json_schema")

    prompt = prompt_template.invoke({"input": query, "title": title})
    # keep running until the number of parsed tasks is equal to the number of reference tasks
    for i in range (3):
        response = llm.invoke(prompt)
        #parse response
        try:
            parsed = json.loads(response["tasks"])
            print('parsed json')
        except:
            print('not json')
            try:
               parsed = response["tasks"]
               print('parsed string')
            except:
                print('not string')
                continue
        try:
            if len(parsed) == len(get_des(title)):
                return parsed
            else:
                print('not equal, parsed:', len(parsed), 'ref:', len(get_des(title)))
        except Exception as e:
            #try 3 more times, and if it still fails, return the parsed
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
#correlation matrix and reorder them based on the hungarian algorithm
def match(ref, gen):
    try:
        ref_clean = preProcessText(ref)
        gen_clean = preProcessText(gen)
        matrix = sbert(ref_clean, gen_clean)
        row_ind, col_ind = linear_sum_assignment(1 - matrix)  # Minimize cost (1 - similarity)
        assigned_similarities = matrix[row_ind, col_ind]
        return np.mean(assigned_similarities), matrix, row_ind.tolist(), col_ind.tolist()
    except:
        print('error in matching' + ref[0])
        return np.nan


# %%
# start the process
model = ChatOllama(model="llama3.2", temperature=1, base_url="http://10.0.3.228:11434")

prompts = {"no_prompt": None, 
           "prompt1": "You are an expert of this occupation: \"{title}\". Your task is to generate clear and concise task descriptions that reflect common responsibilities in this profession. Each description should be specific, action-oriented, and use professional language. Avoid unnecessary details—focus on the core action and purpose of the task.", 
           "prompt2": "You are a specialist of this job: \"{title}\". Your assignment is to create precise and brief job task descriptions that capture typical duties in this occupation. Each description should be particular, action-focused, and employ professional terminology. Skip superfluous details—concentrate on the essential action and objective of the task.", 
           "prompt3": "As an expert of this job: \"{title}\", you generate precise and professional task descriptions that accurately represent duties in this field. Each task description should begin with a strong action verb, clearly state the responsibility, and, where applicable, include the purpose or outcome of the task. Keep descriptions brief but informative.",
           "prompt4": "As a specialist of this occupation: \"{title}\", you create exact and professional task descriptions that correctly represent responsibilities in this domain. Each task description should start with a powerful action verb, explicitly state the duty, and, when relevant, incorporate the goal or result of the task. Keep descriptions concise but enlightening."}


# %%
# run 5 times each for each prompt
for name, prompt in prompts.items():
    if prompt != None:
            with open(folder_name + '/sys_prompt.txt', 'a') as f:
                f.write(prompt + '\n')

    for i in range (5):
        # invoke llm for each title
        for title in tqdm(test_sample_list):
            generated_statements = task_gen(title, model, prompt)
            trial_df.loc[trial_df["title"] == title, "gen_task"] = pd.Series([generated_statements]).values
        result_df = trial_df.reset_index(drop=True)
        result_df[["score", "matrix", "ref_order", "gen_order"]] = result_df.apply(lambda row: match(row["ref_task"], row["gen_task"]), axis=1).apply(pd.Series)

        with open(folder_name + '/' + name + '_'+str(i)+'_result.json', 'w') as f:
            f.write(result_df.to_json(index=True))
    
        

