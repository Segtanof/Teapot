from multiprocessing import Pool
import pandas as pd
import json
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import logging
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
import os
import argparse

# Setup output folder
folder_name = f'results/mistask_match_other{datetime.now().strftime("%d%m_%H%M")}/'
os.makedirs(folder_name, exist_ok=True)
print("folder created")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.FileHandler("execution_log.log"), logging.StreamHandler()])

job_statements = pd.read_excel("datasets/task_statements.xlsx")
job_statements.columns = job_statements.columns.str.lower()
job_statements = job_statements.drop(labels=["incumbents responding","date","domain source"], axis=1).rename(columns={"o*net-soc code":"code", "task type":"type", "task id": "id", "task":"ref_task"})
job_statements = job_statements[job_statements["type"].notna()]
# job_statements["ind"] = job_statements["code"].str[:2]
job_statements = job_statements.groupby("title").agg({"ref_task":list}).reset_index()#, "ind": "first"}).reset_index().sort_values("ind")
# sampled_occupation = job_statements.groupby('ind', group_keys=False).sample(frac=0.05, random_state=3) #43 samples
occupations = (
    pd.read_excel("datasets/occupation_data.xlsx")
    .dropna()
    .rename(columns=lambda x: x.lower())  # Convert column names to lowercase
    .rename(columns={"o*net-soc code": "code"})  # Rename specific column
)
sampled_occupation = job_statements.merge(occupations, how="left", on="title")
# sampled_occupation = sampled_occupation.iloc[501:600]
sampled_occupation = sampled_occupation.iloc[[60, 100] + list(range(200, 401)) + list(range(600, 601))].reset_index(drop=True)

#for trial
# trial_df = sampled_occupation#.sample(3, random_state= 1)
# test_sample_list =[trial_df.iloc[x]["title"] for x in range(len(trial_df))]

#get reference description
def get_des (title):
    task_list = sampled_occupation.query("title == @title")["ref_task"].iloc[0]
    return task_list

def task_gen(title, model, description, system=None):  
    ref_task_count = len(get_des(title))
    json_schema = {"type": "object", "properties": {"occupation": {"type": "string"}, "tasks": {"type": "array", "items": {"type": "string"}, "minItems": ref_task_count, "maxItems": ref_task_count}}, "required": ["occupation", "tasks"]}
    query = "List exactly "+ str(ref_task_count) +" unique task statements that the occupation " + title + "would perform at work. Here is the description of the occupation: " + description
    prompt_template = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")] if system else [("human", "{input}")])
    llm = model.with_structured_output(schema=json_schema, method="json_schema")
    prompt = prompt_template.invoke({"input": query, "title": title})
    while True:
        try:
            response = llm.invoke(prompt)
            tasks = response["tasks"]
            if len(tasks) == ref_task_count and len(set(tasks)) == len(tasks):
                break
            else:
                logging.warning(f"Task issues for {title}: count {len(tasks)}/{ref_task_count}, uniques {len(set(tasks))}. Retrying...")
        except Exception as e:
            logging.error(f"Failed for {title}: {e}. Retrying...")
    return tasks

def process_title(args):
    title, model_config, description, prompt = args
    model = ChatOllama(**model_config)
    start_time = datetime.now()
    tasks = task_gen(title, model, description, system=prompt)
    logging.info(f"Single inference for {title}, duration: {datetime.now() - start_time}")
    return title, tasks

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=11434)  # Dynamic port
args = parser.parse_args()

model_configs = [
    # {"model": "llama3.3", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}","num_predict": 2048},
    {"model": "mistral", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}", "num_predict": 1024},
    # {"model": "deepseek-r1", "temperature": 1, "base_url": "http://127.0.0.1:11434", "num_predict": 512, "num_ctx": 16384}
]
prompts = {
    # "no_prompt": None,
    "prompt1": "You are an expert of this occupation: \"{title}\". Your task is to generate clear, concise and relevant task descriptions associated with this occupation. Each description should be specific, action-oriented, distinct from one another, and use professional language. Avoid unnecessary details—focus on the core action and purpose of the task. "
}

logging.info("Script started")
for model_config in model_configs:
    model_name = model_config["model"]
    logging.info(f"Processing model: {model_name}")
    model = ChatOllama(**model_config)
    model.invoke("Warm-up prompt")

    for name, prompt in prompts.items():
        if prompt:
            start_time = datetime.now()
            with open(f"{folder_name}/sys_prompt.txt", "a") as f:
                f.write(prompt + "\n")
            logging.info(f"Wrote prompt {name}, duration: {datetime.now() - start_time}")

        all_results_df = sampled_occupation.copy()
        all_results_df["gen_task"] = [None] * len(all_results_df)
        all_results_df["iteration"] = None

        for i in range(10):
            start_time = datetime.now()
            with Pool(processes=8) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_title, [(row['title'], model_config, row['description'], prompt) for _, row in sampled_occupation[['title', 'description']].iterrows()]),
                    total=len(sampled_occupation), desc=f"{model_name}-{name}-{i}"
                ))
            logging.info(f"Multiprocessing for {model_name}-{name}-{i}, duration: {datetime.now() - start_time}")

            temp_df = sampled_occupation.copy()
            for title, tasks in results:
                temp_df.loc[temp_df["title"] == title, "gen_task"] = pd.Series([tasks]).values
            temp_df["iteration"] = i
            all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

        start_time = datetime.now()
        with open(f"{folder_name}/{model_name}_{name}_results.json", "w") as f:
            f.write(all_results_df.to_json(index=True))
        logging.info(f"Wrote results JSON for {model_name}-{name}, duration: {datetime.now() - start_time}")

logging.info("Script completed")