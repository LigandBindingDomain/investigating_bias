import os
import json
import pandas as pd
import numpy as np
import requests
import tqdm
import logging
import psutil
from requests.exceptions import RequestException
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.mpnn.model import residue_constants

import plotly.express as px
from scipy.special import softmax
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time
from colabdesign.af import mk_af_model
from colabdesign.shared.protein import pdb_to_string

AA_TO_INDEX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}


# Ensure colabdesign is installed
def install_colabdesign():
    try:
        import colabdesign
    except ImportError:
        print("colabdesign not found, installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1"])
        import colabdesign
    print("colabdesign installed successfully!")
install_colabdesign()

# Constants
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def read_and_select_proteins(file_path, protein_families=None, subset_size=None):
    # Load the data
    proteins_data = pd.read_csv(file_path)

    if subset_size:
        subset_accessions = np.random.choice(proteins_data['Entry'].tolist(), size=min(subset_size, len(proteins_data)), replace=False)
        print(f"Analysing a random subset of {len(subset_accessions)} proteins")
    elif protein_families:
        subset_df = proteins_data[proteins_data['first_protein_name'].isin(protein_families)]
        subset_accessions = subset_df['Entry'].tolist()
        print(f"Analysing {len(subset_accessions)} proteins from the specified families")
    else:
        subset_accessions = proteins_data['Entry'].tolist()
        print(f"Analysing all {len(subset_accessions)} proteins")

    return proteins_data, subset_accessions

# Ensure cache directory exists
RESULTS_CACHE_DIR = "results_cache_new"
if not os.path.exists(RESULTS_CACHE_DIR):
    os.makedirs(RESULTS_CACHE_DIR)

def cache_results(accession, results_data):
    cache_path = os.path.join(RESULTS_CACHE_DIR, f"{accession}.pkl")
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(results_data, cache_file)

def load_cached_results(accession):
    cache_path = os.path.join(RESULTS_CACHE_DIR, f"{accession}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            logging.info(f"Loading cached results for {accession}")
            return pickle.load(cache_file)
    return None

# Define the CHECKPOINT_FILE constant
#CHECKPOINT_FILE = "checkpoint.json"

def convert_float32_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            #print the number of proteins already in the results file of the checkpoint
            print()
            try:
                checkpoint = json.load(f)
                logging.info("Checkpoint loaded successfully.")
                return checkpoint
            except json.JSONDecodeError:
                logging.warning(f"Checkpoint file {CHECKPOINT_FILE} is empty or corrupted. Initializing a new checkpoint.")
                return {'processed': [], 'results': {}}
    else:
        logging.info("Checkpoint file not found. Initializing a new checkpoint.")
        return {'processed': [], 'results': {}}

def save_checkpoint(checkpoint):
    checkpoint = convert_float32_to_float(checkpoint)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    logging.info("Checkpoint saved successfully.")


def get_pdb(accession, output_dir="pdb_files", max_retries=3, retry_delay=5):
    base_url = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
    pdb_path = os.path.join(output_dir, f"{accession}.pdb")

    if os.path.exists(pdb_path):
        logging.info(f"PDB file for {accession} already exists locally.")
        return pdb_path, False

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url.format(accession), timeout=30)
            if response.status_code == 200:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(pdb_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Downloaded {accession}.pdb from AlphaFold database.")
                return pdb_path, False
            else:
                logging.warning(f"Attempt {attempt + 1}: Failed to download {accession}.pdb. Status code: {response.status_code}")
        except RequestException as e:
            logging.warning(f"Attempt {attempt + 1}: Error downloading {accession}.pdb: {str(e)}")

        if attempt < max_retries - 1:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    logging.warning(f"All download attempts failed for {accession}. Skipping this protein.")
    return None, False

def calculate_amino_acid_logits(mpnn_model, mode="unconditional"):
    L = sum(mpnn_model._lengths)
    ar_mask = np.zeros((L, L)) if mode == "unconditional" else 1 - np.eye(L)
    logits = mpnn_model.score(ar_mask=ar_mask)["logits"]
    return logits

def calculate_sequence_score(logits, sequence):
    score = 0
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            aa_index = AA_TO_INDEX[aa]
            position_logits = logits[i]
            score += position_logits[aa_index] - np.log(np.sum(np.exp(position_logits)))
        else:
            print(f"Warning: Unknown amino acid '{aa}' at position {i}")
        # Normalize the score by the length of the sequence
        if len(sequence) > 0:
            score /= len(sequence)

        return score

def main_analysis_pipeline(proteins_data, subset_accessions, pdb_dir):
    mpnn_model = mk_mpnn_model("v_48_020")
    results = {}

    for accession in tqdm.tqdm(subset_accessions, desc="Analysing proteins"):
        try:
            wt_seq = proteins_data.loc[proteins_data['Entry'] == accession, 'sequence'].values[0]
            print(f"Processing {accession}, sequence: {wt_seq[:50]}...")

            pdb_path, _ = get_pdb(accession, pdb_dir)
            if pdb_path is None:
                logging.warning(f"Failed to obtain structure for {accession}. Skipping this protein.")
                continue

            mpnn_model.prep_inputs(pdb_filename=pdb_path)
            mpnn_out = mpnn_model.sample(num=5, temperature=0.1)

            top_index = np.argmax(mpnn_out["score"])
            pmpnn_seq = mpnn_out["seq"][top_index]
            pmpnn_org_score = mpnn_out["score"][top_index]

            logits = calculate_amino_acid_logits(mpnn_model)

            wt_score = calculate_sequence_score(logits, wt_seq)
            pmpnn_score = calculate_sequence_score(logits, pmpnn_seq)

            results[accession] = {
                "Entry": accession,
                "WT_seq": wt_seq,
                "pMPNN_seq": pmpnn_seq,
                "WT_logit_summary": wt_score,
                "pMPNN_logit_summary": pmpnn_score,
                "pMPNN_org_score": pmpnn_org_score,
                "pdb_path": pdb_path,
            }

            del mpnn_out, logits

        except Exception as e:
            logging.error(f"Error processing accession {accession}: {str(e)}", exc_info=True)

    results_df = pd.DataFrame(list(results.values()))
    merged_df = pd.merge(proteins_data, results_df, on="Entry", how="right")
    merged_df.to_csv("common_proteins_progen_esm_loglikelihood_part_4_results.csv", index=False)
    return merged_df

# Usage
file_path = 'common_proteins_progen_esm_loglikelihood_part_4.csv'
subset_size = None  # Set this if you want a random subset
protein_families = None  # Set this if you want to use protein families
proteins_data, subset_accessions = read_and_select_proteins(file_path, protein_families, subset_size)
pdb_dir = '/Users/lauradillon/Downloads/pdb_files'
results_df = main_analysis_pipeline(proteins_data, subset_accessions, pdb_dir)
