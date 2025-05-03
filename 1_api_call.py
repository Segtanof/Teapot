import pandas as pd
import os
import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Load related occupations
related = pd.read_excel('datasets/related_occupations.xlsx').astype(str)
related.columns = related.columns.str.lower().str.replace(" ", "_").str.replace("o*net-soc_", "")
related = related[related["relatedness_tier"].isin(["Primary-Short", "Primary-Long"])]

# Directory to save career data
CAREER_CACHE_DIR = "career_cache"
os.makedirs(CAREER_CACHE_DIR, exist_ok=True)

# API call to get career data
def get_career(answer):
    # Generate a unique filename based on the rating (hash to avoid special chars)
    hash_key = hashlib.md5(answer.encode()).hexdigest()
    cache_file = f"{CAREER_CACHE_DIR}/career_{hash_key}.json"
    
    # Check if cached file exists
    if os.path.exists(cache_file):
        return pd.read_json(cache_file)
    
    # API request
    url = 'https://services.onetcenter.org/ws/mnm/interestprofiler/careers?answers=' + answer + '&start=1&end=1000'
    cookies = {'developer_login': 'dW5pX21hbm5oZWltX2RlMTowMDU1ODEyOTFiYzRjYTYxNGE5YmJlM2E4ZjgyNjk2NWQxNzFiY2Y0'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        'Authorization': 'Basic dW5pX21hbm5oZWltX2RlMTo3MzM5Y3R1',
        'Connection': 'keep-alive',
    }
    params = {'start': '1', 'end': '60'}
    
    response = requests.get(url, params=params, cookies=cookies, headers=headers)
    data = json.loads(response.text)["career"]
    career = pd.DataFrame(data).drop(["href", "tags"], axis=1)
    
    # Save to cache
    career.to_json(cache_file, orient='records')
    time.sleep(1.1)  # Respect API rate limit
    return career

# Matching function
def match_score(rating, title):
    # Load pre-fetched career data
    onet_career = get_career(rating)
    
    #create an empty dataframe to store the results
    result_df = pd.DataFrame(columns=[])

    # Filter careers by fit categories
    best_fit = onet_career[onet_career["fit"] == "Best"]
    great_fit = onet_career[onet_career["fit"] == "Great"]
    good_fit = onet_career[onet_career["fit"] == "Good"]
    
    # Calculate hit rate @n
    best_fit_hits = 1 if title in best_fit["title"].values else 0
    best_and_great_fit_hits = 1 if title in pd.concat([best_fit, great_fit])["title"].values else 0
    all_fit_hits = 1 if title in onet_career["title"].values else 0
    
    # Calculate precision
    best_fit_precision = best_fit_hits / len(best_fit) if len(best_fit) > 0 else 0
    great_fit_precision = best_and_great_fit_hits / len(pd.concat([best_fit, great_fit])) if len(pd.concat([best_fit, great_fit])) > 0 else 0
    all_fit_precision = all_fit_hits / len(onet_career) if len(onet_career) > 0 else 0

    # Calculate hit rate @n in related jobs
    related_job = related[related["title"] == title]
    related_best_fit_hits = 1 if related_job["related_title"].isin(best_fit["title"]).any() else 0
    related_best_and_great_fit_hits = 1 if related_job["related_title"].isin(pd.concat([best_fit, great_fit])["title"]).any() else 0
    related_all_fit_hits = 1 if related_job["related_title"].isin(onet_career["title"]).any() else 0

    
    # Calculate recall @n
    related_job = related[related["title"] == title]
    related_in_best = pd.merge(
        right=best_fit, 
        left=related_job, 
        right_on="title", 
        left_on="related_title", 
        how="inner"
    )
    related_in_best_and_great = pd.merge(
        right=pd.concat([best_fit, great_fit]),
        left=related_job, 
        right_on="title", 
        left_on="related_title", 
        how="inner"
    )
    related_in_all = pd.merge(
        right=onet_career, 
        left=related_job, 
        right_on="title", 
        left_on="related_title", 
        how="inner"
    )

    # Calculate recall
    best_fit_recall = len(related_in_best) / len(related_job) if len(related_job) > 0 else 0
    best_and_great_fit_recall = len(related_in_best_and_great) / len(related_job) if len(related_job) > 0 else 0
    all_fit_recall = len(related_in_all) / len(related_job) if len(related_job) > 0 else 0
    result_df = pd.DataFrame({
        "title": [title],
        "best_fit": [len(best_fit)],
        "great_fit": [len(great_fit)],
        "good_fit": [len(good_fit)],
        "all_fit": [len(onet_career)],
        "c_best_fit_hits": [best_fit_hits],
        "c_best_fit_precision": [best_fit_precision],
        "r_best_fit_hits": [related_best_fit_hits],
        "r_best_fit_recall": [best_fit_recall],
        "c_best_and_great_fit_hits": [best_and_great_fit_hits],
        "c_best_and_great_fit_precision": [great_fit_precision],
        "r_best_and_great_fit_hits": [related_best_and_great_fit_hits],
        "r_best_and_great_fit_recall": [best_and_great_fit_recall],
        "c_all_fit_hits": [all_fit_hits],
        "c_all_fit_precision": [all_fit_precision],
        "r_all_fit_hits": [related_all_fit_hits],
        "r_all_fit_recall": [all_fit_recall]

    })

    
    return result_df

# Parallel processing helper
def process_single_pair(args):
    rating, title = args
    return match_score(rating, title)

def process_rating(generated_df, num_workers=5):
    rating_list = generated_df["rating"].tolist()
    title_list = generated_df["title"].tolist()
    pairs = list(zip(rating_list, title_list))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result_df= list(tqdm(executor.map(process_single_pair, pairs), total=len(pairs)))
    
    return result_df


# Main execution
folder_name = "results/task1/j_mistral"
json_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]
for file in json_files:
    generated_df = pd.read_json(f"{folder_name}/{file}", dtype={"rating": "object"}).dropna()
    generated_df = generated_df[["title", "ind", "rating", "iteration"]]
    # generated_df = generated_df[generated_df["iteration"]]
    result_df = process_rating(generated_df)
    # Concatenate the list of DataFrames into a single DataFrame
    result_df_combined = pd.concat(result_df, ignore_index=True)
    result_df_combined.to_json(f"{folder_name}/{file[:-5]}_processed.json", orient='records')