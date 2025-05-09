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

#get the questions into a list
with open("datasets/60qs.json") as f:
    qs = json.load(f)
    test = qs["questions"]["question"]
    df = pd.DataFrame(test)[['text', 'area', '_index']]
    df.columns = ['question', 'area', 'index']
    rqlist = list(df["question"])
    qlist = rqlist

def clean_text(text):
    return text.strip().replace("\n", " ")

def get_multi_rating(titles_descriptions, model, system_prompt=None):
    json_schema = {
        "type": "object",
        "patternProperties": {
            ".*": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"}
                    },
                    "required": ["question", "rating", "reason"]
                }
            }
        }
    }

    instructions = (
        "You are evaluating interest ratings for multiple occupations. For each occupation below, "
        "rate the following 60 questions using integers 1–5 and give a short reason. "
        "Respond in JSON format where each key is the occupation title and the value is a list of objects "
        "with 'question', 'rating', and 'reason'."
    )

    occupation_blocks = "\n\n".join([
        f"Occupation: \"{title}\"\nDescription: {desc}" for title, desc in titles_descriptions
    ])
    question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(qlist)])
    full_prompt = f"{instructions}\n\n{occupation_blocks}\n\nQuestions:\n{question_block}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")] if system_prompt else [("human", "{input}")]
    )
    prompt = prompt_template.invoke({"input": full_prompt})
    llm = model.with_structured_output(schema=json_schema, method="json_schema")

    for attempt in range(5):
        try:
            with torch.cuda.device(0):
                return llm.invoke(prompt)
        except Exception as e:
            logging.error(f"Batch failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    return {}

def process_group(args):
    group, model_config, prompt = args
    model = ChatOllama(**model_config)
    return get_multi_rating(group, model, prompt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Script started")

    # Load your occupations dataframe
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

    occupations = occupations.iloc[:100]

    first = occupations.index[0]
    last = occupations.index[-1]

    model_configs = [{
        "model": "deepseek-r1",
        "temperature": 1,
        "base_url": f"http://127.0.0.1:{args.port}",
        "num_predict": 1024,
        "num_ctx": 8192,
    }]

    prompts = {"no_prompt": None}

    group_size = 4
    occupation_rows = occupations[['title', 'description']].values.tolist()
    occupation_groups = [occupation_rows[i:i + group_size] for i in range(0, len(occupation_rows), group_size)]

    for model_config in model_configs:
        model_name = model_config["model"]
        logging.info(f"Processing model: {model_name}")

        warmup_model = ChatOllama(**model_config)
        warmup_model.invoke("Warm-up prompt")

        for prompt_name, prompt in prompts.items():
            results = []
            args_list = [(group, model_config, prompt) for group in occupation_groups]

            with Pool(processes=2) as pool:
                for result in tqdm(pool.imap_unordered(process_group, args_list), total=len(args_list), desc=f"{model_name}-{prompt_name}"):
                    if result:
                        results.append(result)

            # Flatten and save with error checking
            records = []
            for res in results:
                for title, qa_list in res.items():
                    for item in qa_list:
                        try:
                            question = clean_text(item["question"])
                            rating = item["rating"]
                            reason = clean_text(item["reason"])
                            records.append({
                                "title": title,
                                "question": question,
                                "rating": rating,
                                "reason": reason,
                            })
                        except KeyError as e:
                            logging.warning(f"Missing key {e} in response for title '{title}': {item}")
                        except Exception as e:
                            logging.error(f"Unexpected error parsing response for title '{title}': {item} | {e}")


            result_df = pd.DataFrame(records)
            result_df["iteration"] = 0

            folder_name = f'results/{model_name}_job_match_{datetime.now().strftime("%d%m_%H%M")}/'
            os.makedirs(folder_name, exist_ok=True)
            result_df.to_json(f"{folder_name}/{model_name}_{prompt_name}_results.json", orient="records")

    logging.info("Script completed")

if __name__ == "__main__":
    main()
