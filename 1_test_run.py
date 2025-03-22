# %%
#for loading data
import pandas as pd
import json

#for llm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#counting
from tqdm import tqdm

#logging
import regex as re
import os
from datetime import datetime
import logging
from multiprocessing import Pool


# Setup output folder
folder_name = f'results/job_match_{datetime.now().strftime("%d%m_%H%M")}/'
os.makedirs(folder_name, exist_ok=True)
print("folder created")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    handlers=[logging.FileHandler("execution_log.log"), logging.StreamHandler()])

# Load and preprocess occupation data
occupations = (
    pd.read_excel("datasets/occupation_data.xlsx")
    .dropna()
    .rename(columns=lambda x: x.lower())  # Convert column names to lowercase
    .rename(columns={"o*net-soc code": "code"})  # Rename specific column
)

# Filter out rows containing "All Other" in the 'title' column
occupations = occupations[~occupations["title"].str.contains("All Other", na=False)]

# Ensure correct data types
occupations = occupations.astype({"code": str, "title": str, "description": str})

# Extract industry code
occupations["ind"] = occupations["code"].str[:2]

# Sample 5% of occupations per industry
sampled_occupation = occupations.groupby("ind", group_keys=False).sample(frac=0.05, random_state=1)

# get a list of sampled occupations
test_sample_list = list(sampled_occupation["title"])

#get the questions into a list
with open("datasets/60qs.json") as f:
    qs = json.load(f)
    test = qs["questions"]["question"]
    df = pd.DataFrame(test)[['text', 'area', '_index']]
    df.columns = ['question', 'area', 'index']
    qlist = list(df["question"])

def get_rating(title, model, system=None):
   json_schema = {"type":"object","properties":{"reason":{"type":"string"},"rating":{"type":"integer","minimum":1,"maximum":5},"items":{"type":"string"}},"required":["reason","rating"]}
   query = "Rate the statement with a number 1, 2, 3, 4, or 5 base on the interest of the occupation \"" + title + "\". 1 is strongly dislike, 2 is dislike, 3 is neutral, 4 is like and 5 is strongly like. Provide your reasons. Here is the statement: "
   prompt_template = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")] if system else [("human", "{input}")])
   llm = model.with_structured_output(schema=json_schema, method="json_schema")

   # pass question to the llm
   rating_list =[]
   reason_list = []
   for q in qlist:
        prompt = prompt_template.invoke({"input": query + q + ".", "title": title})
        while True:
            try:
                response = llm.invoke(prompt)
                rating = response["rating"]
                reason = response["reason"]
                if rating in [1, 2, 3, 4, 5]:
                    rating_list.append(rating)
                    reason_list.append(reason)
                    break
                else:
                    logging.warning(f"Rating issue. Retrying...")
            except Exception as e:
                logging.error(f"Failed for {title}: {e}. Retrying...")
   rating_list = "".join(map(str, rating_list))
   return rating_list, reason_list

def process_title(args):
    title, model_config, prompt = args
    model = ChatOllama(**model_config)
    start_time = datetime.now()
    rating, reason = get_rating(title, model, system=prompt)
    logging.info(f"Single inference for {title}, duration: {datetime.now() - start_time}")
    return title, rating, reason


model_configs = [
    {"model": "llama3.2", "temperature": 1, "base_url": "http://127.0.0.1:11434", "num_predict": 512, "num_ctx": 16384},
    # {"model": "mistral", "temperature": 1, "base_url": "http://127.0.0.1:11434", "num_predict": 512, "num_ctx": 16384},
    # {"model": "deepseek-r1", "temperature": 1, "base_url": "http://127.0.0.1:11434", "num_predict": 512, "num_ctx": 16384}
]
prompts = {
    "no_prompt": None,
    "prompt1": "You are an expert of this occupation: \"{title}\". Your task is to rate the statement according to your professional interest and occupation relevance."
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

        # create a df to store the results
        all_results_df = sampled_occupation.copy()
        all_results_df["rating"] = [None] * len(all_results_df)
        all_results_df["reason"] = [None]

        for i in range(10):
            start_time = datetime.now()
            with Pool(processes=8) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_title, [(title, model_config, prompt) for title in test_sample_list]),
                    total=len(test_sample_list), desc=f"{model_name}-{name}-{i}"
                ))
            logging.info(f"Multiprocessing for {model_name}-{name}-{i}, duration: {datetime.now() - start_time}")

            temp_df = sampled_occupation.copy()
            for title, rating, reason in results:
                temp_df.loc[temp_df["title"] == title, "rating"] = pd.Series([rating]).values
                temp_df.loc[temp_df["title"] == title, "reason"] = pd.Series([reason]).values
            temp_df["iteration"] = i
            all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

        start_time = datetime.now()
        with open(f"{folder_name}/{model_name}_{name}_results.json", "w") as f:
            f.write(all_results_df.to_json(index=True))
        logging.info(f"Wrote results JSON for {model_name}-{name}, duration: {datetime.now() - start_time}")

logging.info("Script completed")



