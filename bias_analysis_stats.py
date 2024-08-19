import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logging.basicConfig(level=logging.INFO)


def clean_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['species'] = df['genus_species']
    df['protein_family'] = df['first_protein_name']

    # Convert columns to numeric, replacing any non-convertible values with NaN
    df['pMPNN_logit_summary'] = pd.to_numeric(df['pMPNN_logit_summary'], errors='coerce')
    df['WT_logit_summary'] = pd.to_numeric(df['WT_logit_summary'], errors='coerce')

    # Replace infinity values with NaN and drop rows with NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Entry Name', 'score_difference_norm'])

    return df


def predict_secondary_structure(sequence):
    analyzer = ProteinAnalysis(sequence)
    secondary_structure = analyzer.secondary_structure_fraction()
    return secondary_structure


def calculate_hydrophobicity(sequence):
    analyzer = ProteinAnalysis(sequence)
    return analyzer.gravy()


def add_secondary_structure_features(df):
    df['helix_fraction'] = df['sequence'].apply(lambda seq: predict_secondary_structure(seq)[0])
    df['sheet_fraction'] = df['sequence'].apply(lambda seq: predict_secondary_structure(seq)[1])
    df['coil_fraction'] = df['sequence'].apply(lambda seq: predict_secondary_structure(seq)[2])
    df['hydrophobicity'] = df['sequence'].apply(calculate_hydrophobicity)
    return df


def calculate_r_squared(df, target_column):
    # Model 1: Species only
    model_species = ols(f'{target_column} ~ C(species)', data=df).fit()
    r2_species = model_species.rsquared

    # Model 2: Protein family only
    model_protein = ols(f'{target_column} ~ C(protein_family)', data=df).fit()
    r2_protein = model_protein.rsquared

    # Model 3: Both species and protein family
    model_both = ols(f'{target_column} ~ C(species) + C(protein_family)', data=df).fit()
    r2_both = model_both.rsquared

    # Model 4: Secondary structure features
    model_hydrophobicity = ols(f'{target_column} ~ hydrophobicity', data=df).fit()
    model_helix = ols(f'{target_column} ~ helix_fraction', data=df).fit()
    model_sheet = ols(f'{target_column} ~ sheet_fraction', data=df).fit()
    model_coil = ols(f'{target_column} ~ coil_fraction', data=df).fit()
    model_secondary = ols(f'{target_column} ~ helix_fraction + sheet_fraction + coil_fraction + hydrophobicity',
                          data=df).fit()
    r2_helix = model_helix.rsquared
    r2_sheet = model_sheet.rsquared
    r2_coil = model_coil.rsquared
    r2_hydrophobicity = model_hydrophobicity.rsquared
    r2_secondary = model_secondary.rsquared

    # Partial R-squared for species given protein family
    ss_residual_full = np.sum(model_both.resid ** 2)
    ss_residual_reduced = np.sum(model_protein.resid ** 2)
    partial_r2_species_given_protein = (ss_residual_reduced - ss_residual_full) / ss_residual_reduced

    # Explained variance
    explained_variance = 1 - ss_residual_full / np.sum((df[target_column] - df[target_column].mean()) ** 2)

    return r2_species, r2_protein, r2_both, r2_secondary, partial_r2_species_given_protein, explained_variance, r2_helix, r2_sheet, r2_coil, r2_hydrophobicity


def perform_variance_analysis(df, score_columns):
    for col in score_columns:
        logging.info(f"\nAnalysing {col}:")
        r2_species, r2_protein, r2_both, r2_secondary, partial_r2_species_given_protein, explained_variance, r2_helix, r2_sheet, r2_coil, r2_hydrophobicity = calculate_r_squared(
            df, col)

        logging.info(f"R-squared (Species only): {r2_species}")
        logging.info(f"R-squared (Protein family only): {r2_protein}")
        logging.info(f"R-squared (Both species and protein family): {r2_both}")
        logging.info(f"R-squared (Hydrophobicity): {r2_hydrophobicity}")
        logging.info(f"R-squared (Helix fraction): {r2_helix}")
        logging.info(f"R-squared (Sheet fraction): {r2_sheet}")
        logging.info(f"R-squared (Coil fraction): {r2_coil}")
        logging.info(f"R-squared (Secondary structure features): {r2_secondary}")
        logging.info(f"Partial R-squared (Species given Protein family): {partial_r2_species_given_protein}")
        logging.info(f"Species and Protein Explained variance: {explained_variance}")


def perform_t_tests(df):
    t_stat, p_value = stats.ttest_rel(df['pMPNN_logit_summary_normalized'], df['WT_logit_summary_normalized'])
    logging.info(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}")
    t_stat, p_value = stats.ttest_rel(df['logit_summary_difference'], df['score_difference_norm'])
    logging.info(f"Paired t-test for normalised and non norm scores: t-statistic = {t_stat}, p-value = {p_value}")


def simplified_variance_analysis(df, target_column, factors):
    try:
        formula = f"{target_column} ~ " + " + ".join([f"C({factor})" for factor in factors])
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        logging.info(f"ANOVA results for {target_column}:")
        logging.info(anova_table)
    except Exception as e:
        logging.error(f"Error in variance analysis for {target_column}: {str(e)}")


def plot_histograms(df, columns):
    for col in columns:
        sns.histplot(df[col], kde=True)
        plt.title(col.replace('_', ' ').title())
        plt.show()


def plot_scatter(df, x_col, y_col):
    try:
        # Fit a linear regression model
        X = sm.add_constant(df[x_col])
        model = sm.OLS(df[y_col], X).fit()
        df['trendline'] = model.predict(X)

        # Create scatter plot with trendline
        fig = px.scatter(df, x=x_col, y=y_col, color='species', hover_data=['Entry', 'protein_family'])
        fig.add_traces(list(px.line(df, x=x_col, y='trendline').data))
        fig.update_layout(title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}')
        fig.show()
    except BrokenPipeError:
        logging.error("Client disconnected before the data could be fully transmitted.")

# Main script
file_path = 'combined_analysis_results_new_normalised_updated.csv'
df = clean_and_prepare_data(file_path)

# Add secondary structure features
df = add_secondary_structure_features(df)

columns_to_check = ['ESM2_650M_pppl', 'ESM2_15B_pppl', 'ESM2_3B_pppl', 'logit_summary_difference',
                    'WT_logit_summary_normalized', 'pMPNN_logit_summary_normalized', 'pMPNN_org_score']

# Perform t-tests
perform_t_tests(df)

# Perform variance analysis
perform_variance_analysis(df, columns_to_check)

# Simplified variance analysis
factors = ['species', 'protein_family']
for col in columns_to_check:
    simplified_variance_analysis(df, col, factors)

# Plot histograms
plot_histograms(df, columns_to_check)

# Plot scatter plots with a trend line


plot_scatter(df, 'WT_logit_summary', 'pMPNN_logit_summary')
plot_scatter(df, 'WT_logit_summary_normalized', 'pMPNN_logit_summary_normalized')
plot_scatter(df, 'pMPNN_logit_summary', 'pMPNN_org_score')
plot_scatter(df, 'logit_summary_difference', 'pMPNN_org_score')


# Save the updated DataFrame
df.to_csv('combined_analysis_results_new_stats.csv', index=False)
