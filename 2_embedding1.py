import os
import logging
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
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

def embed_sentences(sentences):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    embeddings = model.encode(sentences, batch_size=16, convert_to_numpy=True)
    torch.cuda.empty_cache()  # Clear GPU memory
    return embeddings.tolist()

def process_file(df):
    try:
        # Preprocess and embed ref_task and gen_task
        df['ref_clean'] = preProcessText(df['ref_task'].tolist())
        df['gen_clean'] = preProcessText(df['gen_task'].tolist())
        df['ref_embeddings'] = embed_sentences(df['ref_clean'].tolist())
        df['gen_embeddings'] = embed_sentences(df['gen_clean'].tolist())
        return df
    except Exception as e:
        logging.error(f"Error in process_file: {str(e)}")
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
        output_path = os.path.join(folder_name, "emb_" + file)
        
        logging.info(f"Processing file: {file}")
        try:
            df = pd.read_json(input_path).dropna()
            logging.info(f"Loaded {len(df)} rows from {file}")
            
            df = process_file(df)
            logging.info(f"Computed embeddings for {file}")
            
            df.to_json(output_path, orient='records', lines=True)
            logging.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to process {file}: {str(e)}")
            continue
    
    logging.info("Script completed")

if __name__ == "__main__":
    folder_name = 'results/tm_llama3b_n'
    process_files(folder_name)