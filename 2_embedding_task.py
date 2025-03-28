import os
import logging
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preProcessText(text):
    processed = []
    for doc in text:
        if not isinstance(doc, str): doc = str(doc)
        doc = re.sub(r"\\n|\W|\d", " ", doc)
        doc = re.sub(r'\s+[a-z]\s+|^[a-z]\s+|\s+', " ", doc)
        doc = re.sub(r'^\s|\s$', "", doc)
        processed.append(doc.lower())
    return processed

def sbert_batch(ref_list, gen_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim_model = SentenceTransformer("all-mpnet-base-v2", similarity_fn_name="cosine", device=device)
    embeddings_ref = sim_model.encode(ref_list, batch_size=16, convert_to_tensor=True)
    embeddings_gen = sim_model.encode(gen_list, batch_size=16, convert_to_tensor=True)
    similarity = sim_model.similarity(embeddings_ref, embeddings_gen).cpu().numpy()
    del embeddings_ref, embeddings_gen  # Free tensors
    torch.cuda.empty_cache()  # Clear GPU memory
    return similarity

def match_batch(ref_lists, gen_lists):
    results = []
    for ref_tasks, gen_tasks in zip(ref_lists, gen_lists):
        ref_clean = preProcessText(ref_tasks)
        gen_clean = preProcessText(gen_tasks)
        matrix = sbert_batch(ref_clean, gen_clean)
        row_ind, col_ind = linear_sum_assignment(1 - matrix)
        avg_score = np.mean(matrix)
        results.append((avg_score, matrix.tolist(), row_ind.tolist(), col_ind.tolist()))
    return results

def process_chunk(chunk):
    refs, gens = chunk
    return match_batch(refs, gens)

def match_batch_parallel(ref_lists, gen_lists, num_processes=None):
    if num_processes is None:
        num_processes = min(cpu_count(), 2)  # Cap at 8 for SLURM setup
    
    chunk_size = max(1, len(ref_lists) // num_processes)
    chunks = [(ref_lists[i:i + chunk_size], gen_lists[i:i + chunk_size]) 
             for i in range(0, len(ref_lists), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    results = []
    for chunk in chunk_results:
        results.extend(chunk)
    return results

def apply_match_batch(df):
    try:
        ref_lists = df["ref_task"].tolist()
        gen_lists = df["gen_task"].tolist()
        results = match_batch_parallel(ref_lists, gen_lists, num_processes=2)
        scores, matrices, ref_orders, gen_orders = zip(*results)
        df["score"] = scores
        df["matrix"] = matrices
        df["ref_order"] = ref_orders
        df["gen_order"] = gen_orders
        return df
    except Exception as e:
        logging.error(f"Error in apply_match_batch: {str(e)}")
        raise

def process_files(folder_name):
    logging.info("Script started")
    
    if not os.path.exists(folder_name):
        logging.error(f"Folder {folder_name} does not exist")
        return
    
    all_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]
    logging.info(f"Found {len(all_files)} JSON files in {folder_name}")
    
    for file in all_files:
        input_path = os.path.join(folder_name, file)
        output_path = os.path.join(folder_name, "sim" + file)
        
        logging.info(f"Processing file: {file}")
        try:
            all_results_df = pd.read_json(input_path).dropna()
            logging.info(f"Loaded {len(all_results_df)} rows from {file}")
            
            all_results_df = apply_match_batch(all_results_df)
            logging.info(f"Computed similarities for {file}")
            
            all_results_df.to_json(output_path, orient='records', lines=True)
            logging.info(f"Saved results to {output_path}")
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Failed to process {file}: {str(e)}")
            continue
    
    logging.info("Script completed")

if __name__ == "__main__":
    folder_name = 'results/task_match_2103_1957'
    process_files(folder_name)