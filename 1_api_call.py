import pandas as pd
import os
import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

#get related occupation, filtered by primary-short (most relevant)
related = pd.read_excel('datasets/related_occupations.xlsx').astype(str)
related.columns = related.columns.str.lower().str.replace(" ","_").str.replace("o*net-soc_", "")
related = related[related["relatedness_tier"].isin(["Primary-Short", "Primary-Long"])]

#access the api to get the job titles
def get_career(answer):

    url = 'https://services.onetcenter.org/ws/mnm/interestprofiler/careers?answers='+answer+'&start=1&end=1000'
    cookies = {
        'developer_login': 'dW5pX21hbm5oZWltX2RlMTowMDU1ODEyOTFiYzRjYTYxNGE5YmJlM2E4ZjgyNjk2NWQxNzFiY2Y0',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        'Authorization': 'Basic dW5pX21hbm5oZWltX2RlMTo3MzM5Y3R1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Priority': 'u=0, i',
    }

    params = {
        'start': '1',
        'end': '60',
    }

    response = requests.get(
        url,
        params=params,
        cookies=cookies,
        headers=headers,
    )
    #search for the the target occupation in the response
    data = json.loads(response.text)["career"]
    #select only title and fit
    career = pd.DataFrame(data).drop(["href", "tags"], axis=1)
    return career 

def match_score(rating, title):
    # Get career possibilities based on rating
    onet_career = get_career(rating)
    time.sleep(1.1)
    
    # Check for perfect match (returns 1 if match found, 0 if not)
    perfect_match = 1 if title in onet_career["title"].values else 0
    
    # Get best fit careers
    best_fit = onet_career[onet_career["fit"] == "Best"]
    
    # Get related jobs for the given title
    related_job = related[related["title"] == title]
    
    # Check if any related titles match best fit careers
    related_match = len(pd.merge(
        right=best_fit, 
        left=related_job, 
        right_on="title", 
        left_on="related_title", 
        how="inner"
    ))
    
    return perfect_match, related_match

# Parallel processing helper function
def process_single_pair(args):
    rating, title = args
    return match_score(rating, title)

def process_rating(generated_df, num_workers=5):
    """
    Process ratings with parallel execution while respecting API rate limits
    """
    rating_list = generated_df["rating"].tolist()
    title_list = generated_df["title"].tolist()
    pairs = list(zip(rating_list, title_list))
    
    # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # tqdm for progress bar
        results = list(tqdm(executor.map(process_single_pair, pairs), total=len(pairs)))
    
    # Unzip results
    perfect_match, related_match = zip(*results)
    
    # Create new DataFrame efficiently
    result_df = generated_df.copy()
    result_df["perfect_match"] = perfect_match
    result_df["related"] = related_match
    
    return result_df

folder_name = "results/_prompt1"
#access the folder, get file name ends with .json
json_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]

for file in json_files:
    generated_df = pd.read_json(folder_name + '/' + file, dtype={"rating": "object"}).dropna()
    generated_df = generated_df[["title", "ind", "rating","iteration"]]
    generated_df[["perfect_match", "related"]] = None
    result_df = process_rating(generated_df)
    result_df.to_json(folder_name+"/"+file[:-5]+"_processed.json", orient='records')