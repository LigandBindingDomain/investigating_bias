import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import seaborn as sns  # Seaborn for the heatmap
import matplotlib.pyplot as plt  # Matplotlib for saving the figure
from scipy.stats import spearmanr

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# Disable Plotly's default rendering in notebooks
pio.renderers.default = None


def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("The dataset is empty.")
        # Sanity check the first few rows
        print(data.head())
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

    wt_sequence = "TEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM"
    return data, wt_sequence


def identify_mutations(wt_sequence, mutant_sequence):
    mutations = []
    for i, (wt, mt) in enumerate(zip(wt_sequence, mutant_sequence)):
        if wt != mt and mt != '*':  # Skip deletions
            mutations.append(f"{wt}{i + 1}{mt}")
    return mutations


def generate_predictions(protein_sequence, mutations, model_name="facebook/esm2_t6_8M_UR50D"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise

    predictions = {}
    for mutation in tqdm(mutations, desc="Generating predictions", disable=True):
        try:
            wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
            position = pos - 1  # Adjust for 0-based indexing

            masked_sequence = list(protein_sequence)
            masked_sequence[position] = tokenizer.mask_token
            masked_sequence = " ".join(masked_sequence)

            inputs = tokenizer(masked_sequence, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits

            probabilities = torch.nn.functional.softmax(logits[0, position + 1], dim=0)
            log_probabilities = torch.log(probabilities + 1e-8)  # Avoid log(0)

            log_prob_wt = log_probabilities[tokenizer.convert_tokens_to_ids(wt)].item()
            log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(mt)].item()
            llr = log_prob_mt - log_prob_wt

            predictions[mutation] = llr
        except Exception as e:
            print(f"Error processing mutation {mutation}: {e}")
            continue

    return predictions


def generate_predictions_with_cache(protein_sequence, mutations, cache_file='predictions_cache.json'):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_predictions = json.load(f)
    else:
        cached_predictions = {}

    new_mutations = [m for m in mutations if m not in cached_predictions]

    if new_mutations:
        new_predictions = generate_predictions(protein_sequence, new_mutations)
        cached_predictions.update(new_predictions)

        with open(cache_file, 'w') as f:
            json.dump(cached_predictions, f)

    return {m: cached_predictions[m] for m in mutations}


def calculate_esm2_scores(data, wt_sequence):
    data['mutations'] = data.apply(lambda row: identify_mutations(wt_sequence, row['aa_seq']), axis=1)
    all_mutations = list(set([mut for muts in data['mutations'] for mut in muts]))
    all_predictions = generate_predictions_with_cache(wt_sequence, all_mutations)
    data['esm2_score'] = data['mutations'].apply(lambda muts: sum(all_predictions.get(mut, 0) for mut in muts))
    return data


def calculate_correlations(data):
    # Aggregate duplicate entries by averaging the fitness values
    aggregated_data = data.groupby(['aa_seq', 'assay']).agg({'fitness': 'mean'}).reset_index()

    # Pivot the data to create a DataFrame where each column is an assay
    pivot_data = aggregated_data.pivot(index='aa_seq', columns='assay', values='fitness')

    # Calculate the correlation matrix
    correlation_matrix = pivot_data.corr(method='spearman')

    return correlation_matrix


def visualize_correlation_matrix(correlation_matrix, output_dir='output_plots'):
    # Replace NaN values with 0 for the heatmap
    correlation_matrix = correlation_matrix.fillna(0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Spearman Correlation Matrix of Experimental Fitness Between Assay Types')

    # Save the figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()


def plot_separate_groups(data, output_dir='output_plots'):
    # Separate by assay types
    for assay in data['assay'].unique():
        subset = data[data['assay'] == assay]
        fig = px.scatter(subset, x='fitness', y='esm2_score', title=f'ESM2 Score vs Fitness for {assay}')
        fig.write_html(f"{output_dir}/{assay}_fitness_vs_esm2.html")

    # Separate by Hamming distances
    for ham in sorted(data['Nham_aa'].unique()):
        subset = data[data['Nham_aa'] == ham]
        fig = px.scatter(subset, x='fitness', y='esm2_score', title=f'ESM2 Score vs Fitness for Hamming {ham}')
        fig.write_html(f"{output_dir}/hamming_{ham}_fitness_vs_esm2.html")

    # Combine all binding data and all abundance data
    binding_data = data[data['assay'].str.contains('BindingPCA')]
    fig = px.scatter(binding_data, x='fitness', y='esm2_score', title='ESM2 Score vs Fitness for Binding Data')
    fig.write_html(f"{output_dir}/binding_data_fitness_vs_esm2.html")

    abundance_data = data[data['assay'].str.contains('AbundancePCA')]
    fig = px.scatter(abundance_data, x='fitness', y='esm2_score', title='ESM2 Score vs Fitness for Abundance Data')
    fig.write_html(f"{output_dir}/abundance_data_fitness_vs_esm2.html")


def main():
    # Load and preprocess data
    data, wt_sequence = load_and_preprocess_data('KRAS_data.csv')

    # Calculate ESM2 scores
    print("Calculating ESM2 scores...")
    data = calculate_esm2_scores(data, wt_sequence)

    # Analyze assay type and Hamming distance effects
    print("Analyzing assay type and Hamming distance effects...")
    correlation_matrix = calculate_correlations(data)
    visualize_correlation_matrix(correlation_matrix)

    # Generate separate plots
    print("Generating separate plots...")
    plot_separate_groups(data)

    print("Analysis complete. Check output files for results.")


if __name__ == "__main__":
    main()
