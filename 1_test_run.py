# %%
#for loading data
import pandas as pd
import json

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

#get the questions into a list
with open("datasets/60qs.json") as f:
    qs = json.load(f)
test = qs["questions"]["question"]
df = pd.DataFrame(test)[['text', 'area', '_index']]
df.columns = ['question', 'area', 'index']
qlist = list(df["question"])

#whole dataset
related = pd.read_excel('datasets/related_occupations.xlsx').astype(str)
related.columns = related.columns.str.lower().str.replace(" ","_").str.replace("o*net-soc_", "")
related = related[related["relatedness_tier"] == "Primary-Short"]

def get_rating()
