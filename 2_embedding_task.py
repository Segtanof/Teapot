import os
import logging
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
import math

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

def embed_sentences(sentences, model=None):
    if model is None:
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    embeddings = model.encode(sentences, batch_size=16, convert_to_numpy=True)
    return embeddings.tolist()

def process_chunk(chunk, model=None):
    try:
        # Preprocess and embed ref_task and gen_task for the chunk
        chunk['ref_clean'] = preProcessText(chunk['ref_task'].tolist())
        chunk['gen_clean'] = preProcessText(chunk['gen_task'].tolist())
        chunk['ref_embeddings'] = embed_sentences(chunk['ref_clean'].tolist(), model)
        chunk['gen_embeddings'] = embed_sentences(chunk['gen_clean'].tolist(), model)
        return chunk
    except Exception as e:
        logging.error(f"Error in process_chunk: {str(e)}")
        return None

def process_file(df, num_processes=8):
    try:
        # Initialize the model once to share across processes
        model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        # Split DataFrame into chunks
        chunk_size = math.ceil(len(df) / num_processes)
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, [(chunk, model) for chunk in chunks])

        # Combine results
        processed_chunks = [chunk for chunk in results if chunk is not None]
        if not processed_chunks:
            raise ValueError("No chunks processed successfully")
        
        result_df = pd.concat(processed_chunks, ignore_index=True)
        return result_df
    except Exception as e:
        logging.error(f"Error in process_file: {str(e)}")
        raise

def process_files(folder_name, num_processes=8):
    logging.info("Script started")
    
    if not os.path.exists(folder_name):
        logging.error(f"Folder {folder_name} does not exist")
        return
    
    all_files = [f for f in os.listdir(folder_name) if f.endswith('.json')]
    logging.info(f"Found {len(all_files)} JSON files in {folder_name}")
    
    for file in all_files:
        input_path = os.path.join(folder_name, file)
        output_path = os.path.join(folder_name, "emb_" + file)
        
        logging.info(f"Processing file: {file}")
        try:
            df = pd.read_json(input_path, lines=True).dropna()
            logging.info(f"Loaded {len(df)} rows from {file}")
            
            df = process_file(df, num_processes)
            logging.info(f"Computed embeddings for {file}")
            
            df.to_json(output_path, orient='records', lines=True)
            logging.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to process {file}: {str(e)}")
            continue
    
    logging.info("Script completed")

if __name__ == "__main__":
    folder_name = 'results/test'
    process_files(folder_name, num_processes=8)