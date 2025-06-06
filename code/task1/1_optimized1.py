import logging
import argparse
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import time
import torch
import os

# SLURM environment setup
# os.environ["OMP_NUM_THREADS"] = "4"  # Half of 8 cores for threading within processes
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU
folder_name = f'results/test1ajob_match_{datetime.now().strftime("%d%m_%H%M")}/'
os.makedirs(folder_name, exist_ok=True)
print("folder created")

logging.basicConfig(level=logging.INFO , format="%(asctime)s - %(levelname)s - %(message)s")

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

# discard rows with ind = 55
occupations = occupations[occupations['ind'] != '55'].reset_index(drop=True)

occupations = occupations.iloc[360:480]

first = occupations.index[0]
last = occupations.index[-1]

# Sample 5% of occupations per industry
# sampled_occupation = occupations.groupby('ind', group_keys=False).sample(frac=0.05, random_state=1)

# get a list of sampled occupations
# dsampled_occupation = sampled_occupation[:]
# test_sample_list = list(dsampled_occupation["title"])
# test_description_list = list(dsampled_occupation["description"])

#get the questions into a list
with open("datasets/60qs.json") as f:
    qs = json.load(f)
    test = qs["questions"]["question"]
    df = pd.DataFrame(test)[['text', 'area', '_index']]
    df.columns = ['question', 'area', 'index']
    rqlist = list(df["question"])
    qlist = rqlist
  
def clean_text(text):
    if isinstance(text, str):
        # Replace invalid UTF-8 characters with a replacement character or empty string
        return text.encode('utf-8', errors='replace').decode('utf-8')
    elif isinstance(text, list):
        # Handle lists (e.g., 'reason' column)
        return [clean_text(item) if isinstance(item, str) else item for item in text]
    return text

def get_rating(title, model, description, system_prompt=None, batch_size=60):
    json_schema = {"type":"object","properties":{"reason":{"type":"string"},"rating":{"type":"integer","minimum":1,"maximum":5},"items":{"type":"string"}},"required":["reason","rating"]}
    query = "Rate the statement with a number either 1, 2, 3, 4, or 5 base on the interest of the occupation \"" + title + "\". This occupation "+ description +" Your options for the ratings are the following: 1 is strongly dislike, 2 is dislike, 3 is neutral, 4 is like and 5 is strongly like. Provide your reasons. Return your response strictly as a JSON object matching this schema: "+ str(json_schema) +". Here is the statement: "
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")] if system_prompt else [("human", "{input}")])

    llm = model.with_structured_output(schema=json_schema, method="json_schema")
    
    ratings, reasons = [], []
    for i in range(0, len(qlist), batch_size):
        batch = qlist[i:i + batch_size]
        prompts = [prompt_template.invoke({"input": query + q + ".", "title": title}) for q in batch]
        
        attempt = 0
        while True:  # Infinite loop until all ratings are valid
            attempt += 1
            start_time = time.time()
            logging.info(f"Starting batch {i} (questions {i}-{i+len(batch)-1}), attempt {attempt}")
            try:
                with torch.cuda.device(0):
                    logging.debug(f"Sending batch {i} to LLM: {prompts}")
                    responses = llm.batch(prompts, config={"num_threads": 2})
                
                elapsed = time.time() - start_time
                logging.info(f"Batch {i} response received after {elapsed:.2f} seconds")
                logging.debug(f"Raw responses for batch {i}: {responses}")

                temp_ratings = []
                temp_reasons = []
                all_valid = True
                
                for j, resp in enumerate(responses):
                    rating = resp.get("rating")
                    reason = resp.get("reason", "No reason provided")
                    
                    if not isinstance(rating, int) or rating not in [1, 2, 3, 4, 5]:
                        logging.warning(f"Attempt {attempt}: Invalid rating {rating} for question {batch[j]}. Retrying batch...")
                        all_valid = False
                        break
                    
                    temp_ratings.append(str(rating))
                    temp_reasons.append(reason)
                
                if all_valid:
                    ratings.extend(temp_ratings)
                    reasons.extend(temp_reasons)
                    logging.info(f"Batch starting at question {i} succeeded after {attempt} attempts")
                    break  # Move to next batch if all ratings are valid
                
            except Exception as e:
                logging.error(f"Attempt {attempt}: Batch failed for {title} starting at question {i}: {e}. Retrying...")
            time.sleep(0.5)

    return "".join(ratings), reasons


def process_chunk(chunk_args):
    model_config, prompt, titles, descriptions = chunk_args
    model = ChatOllama(**model_config)
    results = []
    for title, desc in zip(titles, descriptions):
        ratings, reasons = get_rating(title, model, desc, prompt)
        if ratings:  # Only append if successful
            results.append((title, ratings, reasons))
    return results
def initializer():
    """Initialize each process"""
    logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    
    model_configs = [
        # {"model": "mistral", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}", 
        #  "num_predict": 1024, "num_ctx": 8192},
        # {"model": "deepseek-r1", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}", 
        #  "num_predict": 1024, "num_ctx": 8192},
        # {"model": "llama3.3", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}", 
        #  "num_predict": 1024, "num_ctx": 8192},
        {"model": "llama3.2", "temperature": 1, "base_url": f"http://127.0.0.1:{args.port}", 
         "num_predict": 1024, "num_ctx": 8192}
    ]
    
    prompts = {
        # "no_prompt": None,
        "prompt1": "You are an expert of this occupation: \"{title}\". Your task is to rate the statement according to your professional interest and occupation relevance."
    }
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Script started")
    
    num_processes = 4
    
    for model_config in model_configs:
        model_name = model_config["model"]
        logging.info(f"Processing model: {model_name}")
        
        warmup_model = ChatOllama(**model_config)
        warmup_model.invoke("Warm-up prompt")
        
        for name, prompt in prompts.items():
            if prompt:
                with open(f"{folder_name}/sys_prompt.txt", "a") as f:
                    f.write(prompt + "\n")
            
            all_results_df = occupations.copy()
            all_results_df["rating"] = pd.Series([None] * len(all_results_df))
            all_results_df["reason"] = [None] * len(all_results_df)
            all_results_df = all_results_df.astype({"rating": str})
            
            for i in range(10):  # Reduced iterations
                start_time = datetime.now()
                
                chunk_size = len(occupations) // num_processes
                chunks = [
                    (
                        model_config,
                        prompt,
                        occupations['title'][j:j+chunk_size].tolist(),
                        occupations['description'][j:j+chunk_size].tolist()
                    )
                    for j in range(0, len(occupations), chunk_size)
                ]
                
                with Pool(processes=num_processes, initializer=initializer) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(process_chunk, chunks),
                        total=len(chunks),
                        desc=f"{model_name}-{name}-{i}"
                    ))
                
                results = [item for sublist in results for item in sublist]
                
                temp_df = occupations.copy()
                for title, rating, reason in results:
                    temp_df.loc[temp_df["title"] == title, "rating"] = pd.Series([rating], dtype="string").values
                    temp_df.loc[temp_df["title"] == title, "reason"] = pd.Series([reason]).values
                temp_df["iteration"] = i
                all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)
                
                logging.info(f"Completed {model_name}-{name}-{i}, duration: {datetime.now() - start_time}")
            
            # Clean text columns before saving
            for col in ['title', 'description', 'reason']:
                all_results_df[col] = all_results_df[col].apply(clean_text)

            all_results_df.to_json(f"{folder_name}/{model_name}_{name}_results{first}-{last}.json", orient="records", lines=True)

    logging.info("Script completed")

if __name__ == "__main__":
    main()