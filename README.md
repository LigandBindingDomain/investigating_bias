
# Species Bias in Protein Design Models: Replication and Extension

This repository contains the code and data used for the replication of the species bias findings in protein language models (PLMs) and the extension of these analyses to structure-based models (ProteinMPNN). The main focus is on quantifying and analysing species bias in ESM-2 and ProteinMPNN models, and comparing model predictions with experimental fitness data.

## Project Structure

This repository is organised as follows:

- **`whole_seq_to_pppl.py`**:  
  Script for calculating pseudo-log-likelihood (PLL) scores for protein sequences using the ESM-2 model. It computes the PLL of each sequence by masking each amino acid, predicting it, and accumulating the log probabilities across the sequence. This script implements the method used to replicate the findings from Ding and Steinhardt (2024) regarding species bias in protein language models. there is also **`pppL_ESM2_650M.ipynb`**: A Jupyter Notebook that demonstrates the process of calculating the pseudo-log-likelihood (PLL) for protein sequences using the ESM-2 650M model. It includes the implementation of masked marginal scoring and visualisations of the results for species bias analysis.

- **`bias_analysis_stats.py`**:  
  This script conducts the statistical analysis of species bias in protein language models. It runs the linear regressions to quantify the variance in ESM-2 and ProteinMPNN predictions explained by species and protein type, calculating R-squared values and testing significance using t-tests.

- **`KRAS_fitness_ESM2.py`**:  
  This script compares the predictions of ESM-2 with experimentally measured fitness values from a KRAS protein deep mutational scanning dataset. It calculates the likelihoods for each variant and evaluates the correlation between these predictions and experimental fitness values across different types of mutations and assays.

- **`ProteinMPNN_bias_analysis.py`**:  
  This script extends the species bias analysis to ProteinMPNN. It compares the likelihoods of wild-type sequences and predicted sequences based on structural data and performs statistical analyses similar to those applied to the ESM-2 model.


## Requirements

To run the scripts, you'll need the following dependencies:

- Python 3.8+
- `torch` (for handling models such as ESM-2 and ProteinMPNN)
- `biopython` (for sequence analysis)
- `pandas` (for data manipulation)
- `matplotlib` and `seaborn` (for plotting)
- `scikit-learn` (for statistical analysis)
- `requests` (for accessing the UniProt and AlphaFold APIs)

You can install the required packages using pip:

```bash
pip install torch biopython pandas matplotlib seaborn scikit-learn requests
```

## Usage

1. **Replication of Species Bias Findings in ESM-2**:  
   To replicate the findings on species bias in ESM-2, run the script `whole_seq_to_pppl.py`. This script will calculate the PLL scores for the sequences in the dataset. Follow this with the script `bias_analysis_stats.py`, which will analyse the variance explained by species and protein type. THe exact database of common proteins is yet to be shared here. 

   ```bash
   python whole_seq_to_pppl.py
   python bias_analysis_stats.py
   ```

2. **Extension to ProteinMPNN**:  
   Run the script `ProteinMPNN_bias_analysis.py` to analyse species bias in the ProteinMPNN model. This script computes the likelihoods for both wild-type and predicted sequences and performs the necessary statistical analysis.

   ```bash
   python ProteinMPNN_bias_analysis.py
   ```

3. **Comparison to Experimental Fitness**:  
   The script `KRAS_fitness_ESM2.py` evaluates the ESM-2 predictions against experimental fitness data. This script processes the KRAS dataset and computes the correlation between ESM-2 predictions and experimentally validated fitness values.

   ```bash
   python KRAS_fitness_ESM2.py
   ```

## Datasets

- **Ding and Steinhardt Dataset**:  
  (yet to be shared) The dataset used for replicating the species bias findings in ESM-2 is based on the dataset from Ding and Steinhardt (2024). The sequences are organised based on species and protein types, filtered for redundancy and taxonomic representation.
  
- **KRAS Fitness Dataset**:  
  availble from the original publication "The energetic and allosteric landscape for KRAS inhibition" 10.1038/s41586-023-06954-0 as supplementary table 4. zzThis dataset includes 26,000 variants of the KRAS protein, which were subjected to two primary assays (AbundancePCA and BindingPCA). Experimental fitness values were compared to ESM-2 predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact `lauar.dillon@dtc.ox.ac.uk`.

## Acknowledgements

This work is based on previous research by Ding & Steinhardt (2024) in their paper "Protein language models are biased by unequal sequence sampling across the tree of life" 10.1101/2024.03.07.584001 and extends their findings to structure-based models like ProteinMPNN. Special thanks to the authors of the original work for their assistance in replicating their results.
