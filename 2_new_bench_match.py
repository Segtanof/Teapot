# %%
#for loading data
import pandas as pd
import json

#for llm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


#similarity
import regex as re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
import numpy as np

#counting
from tqdm import tqdm

from datetime import datetime
import os

# computation
from concurrent.futures import ThreadPoolExecutor
import logging  # For logging to file and console

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("execution_log.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

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
job_statements = job_statements[job_statements["type"].notna()]
job_statements["ind"] = job_statements["code"].str[:2]
job_statements = job_statements.groupby("title").agg({"ref_task":list, "ind": "first"}).reset_index().sort_values("ind")
sampled_occupation = job_statements.groupby('ind', group_keys=False).sample(frac=0.05, random_state=1) #43 samples


# %%
#for trial
trial_df = sampled_occupation#.sample(3, random_state= 1)
test_sample_list =[trial_df.iloc[x]["title"] for x in range(len(trial_df))]
test_sample_list

# %% [markdown]
def process_title(args):
    title, model, prompt = args
    start_time = datetime.now()
    tasks = task_gen(title, model, system=prompt)  # prompt as system
    logging.info(f"Single inference for {title}, duration: {datetime.now() - start_time}")
    return title, tasks  # Returns list of tasks

# %%
#get reference description
def get_des (title):
    task_list = sampled_occupation.query("title == @title")["ref_task"].iloc[0]
    return task_list

# %%
#invoke llm to generate tasks
def task_gen(title, model, system=None):
    """Generate exactly the required number of unique task statements for a title."""
    # Get reference task count (assumed function)
    ref_task_count = len(get_des(title))
    
    # Define JSON schema
    json_schema = {
        "type": "object",
        "properties": {
            "occupation": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": ref_task_count,
                "maxItems": ref_task_count
            }
        },
        "required": ["occupation", "tasks"]
    }

    # Construct prompt
    query = (
        f"List exactly {ref_task_count} unique task statements that the occupation '{title}' "
        "would perform at work. Ensure each statement is distinct, concise, and relevant."
    )
    
    # Use system prompt if provided, otherwise just human query
    if system:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}")
        ])
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", "{input}")
        ])

    # Configure LLM with structured output
    llm = model.with_structured_output(schema=json_schema, method="json_schema")
    prompt = prompt_template.invoke({"input": query})

    # Invoke and parse response
    try:
        response = llm.invoke(prompt)
        # Expecting dict from with_structured_output
        tasks = response["tasks"]
        
        # Validate task count and uniqueness
        if len(tasks) != ref_task_count:
            logging.warning(f"Task count mismatch for {title}: got {len(tasks)}, expected {ref_task_count}")
            return tasks  # Return anyway, handle downstream
        if len(set(tasks)) < len(tasks):
            logging.warning(f"Duplicate tasks detected for {title}")
        
        return tasks
    except Exception as e:
        logging.error(f"Failed to generate tasks for {title}: {e}")
        # Return dummy tasks to avoid breaking pipeline
        return [f"Error: Task {i+1} for {title}" for i in range(ref_task_count)]
        
    
def preProcessText(text):
    """Preprocess a list of text strings."""
    processed = []
    for doc in text:
        if not isinstance(doc, str):  # Handle non-string (e.g., list or NaN)
            doc = str(doc)
        doc = re.sub(r"\\n", "", doc)
        doc = re.sub(r"\W", " ", doc)
        doc = re.sub(r"\d", " ", doc)
        doc = re.sub(r'\s+[a-z]\s+', " ", doc)
        doc = re.sub(r'^[a-z]\s+', "", doc)
        doc = re.sub(r'\s+', " ", doc)
        doc = re.sub(r'^\s', "", doc)
        doc = re.sub(r'\s$', "", doc)
        processed.append(doc.lower())
    return processed

def sbert_batch(ref_list, gen_list):
    """Compute similarity scores for all ref and gen texts in one batch."""
    sim_model = SentenceTransformer("all-mpnet-base-v2", similarity_fn_name="cosine")
    embeddings_ref = sim_model.encode(ref_list, batch_size=32)  # Batch embeddings
    embeddings_gen = sim_model.encode(gen_list, batch_size=32)
    similarities = sim_model.similarity(embeddings_ref, embeddings_gen).numpy()
    return similarities

def match_batch(ref_list, gen_list):
    """Batch process matching for multiple ref-gen pairs."""
    ref_clean = preProcessText(ref_list)
    gen_clean = preProcessText(gen_list)
    matrix = sbert_batch(ref_clean, gen_clean)
    
    # Process Hungarian algorithm per pair
    results = []
    for i in range(len(ref_list)):
        row_matrix = matrix[i:i+1, i:i+1] if len(ref_list) == 1 else matrix[i, i].reshape(1, 1)  # Handle single pair
        row_ind, col_ind = linear_sum_assignment(1 - row_matrix)  # Minimize cost
        score = matrix[i, i]  # Diagonal score for single pair
        results.append((score, row_matrix.tolist(), row_ind.tolist(), col_ind.tolist()))
    return results

def apply_match_batch(df):
    """Apply batched matching to the entire DataFrame."""
    ref_list = df["ref_task"].tolist()
    gen_list = df["gen_task"].tolist()
    results = match_batch(ref_list, gen_list)
    scores, matrices, ref_orders, gen_orders = zip(*results)
    df["score"] = scores
    df["matrix"] = matrices
    df["ref_order"] = ref_orders
    df["gen_order"] = gen_orders
    return df

# %%
# start the process
llama = ChatOllama(model="llama3.2", temperature=1, base_url="http://127.0.0.1:11434")
mistral = ChatOllama(model="mistral", temperature=1, base_url="http://127.0.0.1:11434")
deepseek = ChatOllama(model="deepseek-r1", temperature=1, base_url="http://127.0.0.1:11434")
model_list = [llama, mistral, deepseek]

prompts = {"no_prompt": None, 
           "prompt1": "You are an expert of this occupation: \"{title}\". Your task is to generate clear and concise task descriptions that reflect common responsibilities in this profession. Each description should be specific, action-oriented, and use professional language. Avoid unnecessary details—focus on the core action and purpose of the task.",}

logging.info("Script started")
for model in model_list:
    model_name = model.model
    logging.info(f"Processing model: {model_name}")
    model.invoke("Warm-up prompt")

    for name, prompt in prompts.items():
        if prompt:
            start_time = datetime.now()
            with open(f"{folder_name}/sys_prompt.txt", "a") as f:
                f.write(prompt + "\n")
            logging.info(f"Wrote prompt {name}, duration: {datetime.now() - start_time}")

        all_results_df = trial_df.copy()
        all_results_df["gen_task"] = None  # Now a list of tasks
        all_results_df["iteration"] = None

        for i in range(10):
            start_time = datetime.now()
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(tqdm(
                    executor.map(process_title, [(title, model, prompt) for title in test_sample_list]),
                    total=len(test_sample_list),
                    desc=f"{model_name}-{name}-{i}"
                ))
            logging.info(f"ThreadPoolExecutor for {model_name}-{name}-{i}, duration: {datetime.now() - start_time}")

            temp_df = trial_df.copy()
            for title, tasks in results:
                temp_df.loc[temp_df["title"] == title, "gen_task"] = pd.Series([tasks]).values  # Store as list
            temp_df["iteration"] = i
            all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

        # Batch match (adjust for list of tasks)
        start_time = datetime.now()
        # Flatten gen_task lists to strings for matching (assuming one task per ref_task for simplicity)
        all_results_df["gen_task_str"] = all_results_df["gen_task"].apply(lambda x: x[0] if x and len(x) > 0 else "")
        all_results_df = apply_match_batch(all_results_df.rename(columns={"gen_task_str": "gen_task"}))
        logging.info(f"Batch matching for {model_name}-{name}, duration: {datetime.now() - start_time}")

        # Save once
        start_time = datetime.now()
        all_results_df = all_results_df.reset_index(drop=True)
        with open(f"{folder_name}/{model_name}_{name}_results.json", "w") as f:
            f.write(all_results_df.to_json(index=True))
        logging.info(f"Wrote results JSON for {model_name}-{name}, duration: {datetime.now() - start_time}")

logging.info("Script completed")