import pandas as pd
import numpy as np
import os
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, globalenv
from rpy2.robjects.packages import importr
pandas2ri.activate()

# Print versions for debugging
import pymer4
print(f"pymer4 version: {pymer4.__version__}")

# Set R_HOME and PATH for pymer4
r_path = r'C:\Program Files\R\R-4.5.0'
os.environ['R_HOME'] = r_path
os.environ['PATH'] = r_path + r'\bin;' + os.environ.get('PATH', '')

lme4 = importr('lme4')
print(f"lme4 version: {robjects.r('packageVersion(\"lme4\")')[0]}")

# 1. Load and preprocess JSON data
json_path = r'd:\OneDrive - Universität Mannheim\MMM\Master Thesis\Teapot\results\task1\task1_data.json'
try:
    df = pd.read_json(json_path, lines=True)
except FileNotFoundError:
    print(f"Error: File {json_path} not found. Check path.")
    raise

# Filter for model and preprocess
llm = "deepseek-r1"
df['model'] = df['model'].replace('llama3.2', 'llama3b').replace('llama3.3', 'llama70b')
df['prompt'] = df['prompt'].replace('no', 'benchmark').replace('prompt1', 'persona')
df['prompt'] = df['prompt'].astype('category')
df = df[df['model'] == llm]
df.drop(columns=['all_fit'], inplace=True)
df.rename(columns={'c_all_fit_hits': 'correct_match', 'r_all_fit_hits': 'related_match', 'r_all_fit_recall': 'related_recall'}, inplace=True)

# Load and merge occupation and zone data
code_path = r"d:\OneDrive - Universität Mannheim\MMM\Master Thesis\Teapot\datasets\occupation.xlsx"
zone_path = r"d:\OneDrive - Universität Mannheim\MMM\Master Thesis\Teapot\datasets\job_zone.xlsx"
try:
    occupation = pd.read_excel(code_path)
    occupation.drop(columns=['Description'], inplace=True)
    occupation.columns = ['code', 'title']
    zone = pd.read_excel(zone_path)
    zone = zone[['Title', 'Job Zone']]
    zone.columns = ['title', 'zone']
except FileNotFoundError as e:
    print(f"Error: Excel file not found: {e}")
    raise

df = df.merge(occupation, on='title', how='left')
df = df.merge(zone, on='title', how='left')
df['industry'] = df['code'].apply(lambda x: x.split('-')[0] if pd.notnull(x) else np.nan).fillna('Unknown')
df['industry'] = pd.Categorical(df['industry'])
df['zone'] = df['zone'].astype(str)
df['zone'] = pd.Categorical(df['zone'])

# Convert to R DataFrame
globalenv['r_df'] = pandas2ri.py2rpy(df)

# R script with scalable outlier exclusion
r_script = '''
library(lme4)

# Ensure industry is a factor with '43' as reference
if ("43" %in% levels(r_df$industry)) {
    r_df$industry <- factor(r_df$industry, levels = c("43", setdiff(levels(r_df$industry), "43")))
} else {
    message("Warning: '43' not found in industry levels. Using default order.")
}

# Fit initial GLMM
initial_model <- glmer(
    correct_match ~ prompt * industry + (1 | title),
    data = r_df,
    family = binomial(link = "logit"),
    control = glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 50000),
        calc.derivs = FALSE,
        check.conv.grad = .makeCC("warning", tol = 0.001, relTol = NULL)
    )
)

# Extract residuals
resids <- resid(initial_model, type = "pearson")
r_df$resid <- resids

# Determine exclusion threshold based on outlier rate
initial_outliers <- r_df[abs(r_df$resid) > 3, ]
initial_outlier_rate <- nrow(initial_outliers) / nrow(r_df) * 100
exclusion_threshold <- ifelse(initial_outlier_rate < 2, 5, ifelse(initial_outlier_rate <= 5, 4, NA))

if (is.na(exclusion_threshold)) {
    clean_model <- initial_model  # No exclusion, use robust method if needed
    message("High outlier rate (> 5%), consider robust method (e.g., brglm2).")
} else {
    # Exclude based on threshold
    r_df_clean <- r_df[abs(r_df$resid) <= exclusion_threshold, ]
    clean_model <- glmer(
        correct_match ~ prompt * industry + (1 | title),
        data = r_df_clean,
        family = binomial(link = "logit"),
        control = glmerControl(
            optimizer = "bobyqa",
            optCtrl = list(maxfun = 50000),
            calc.derivs = FALSE,
            check.conv.grad = .makeCC("warning", tol = 0.001, relTol = NULL)
        )
    )
}

# Extract summary and diagnostics
initial_summary <- summary(initial_model)
if (!is.na(exclusion_threshold)) {
    clean_summary <- summary(clean_model)
    clean_resids <- resid(clean_model, type = "pearson")
    r_df_clean$resid <- clean_resids
}

# Outlier summary from initial model
initial_outlier_summary <- table(initial_outliers$industry, initial_outliers$prompt)
initial_total_outliers <- nrow(initial_outliers)
initial_total_data <- nrow(r_df)
initial_outlier_percentage <- initial_total_outliers / initial_total_data * 100

# Top residuals from initial model
initial_top_residuals <- head(r_df[order(-abs(r_df$resid)), c("title", "industry", "prompt", "resid")], 10)

# Save results
sink("glmm_summary_scalable.txt")
cat("Initial Model Summary:\n")
print(initial_summary,correlation=TRUE)
if (!is.na(exclusion_threshold)) {
    cat("\nCleaned Model Summary:\n")
    print(clean_summary,correlation=TRUE)
    cat("\nCleaned Total Observations:", nrow(r_df_clean), "\n")
    cat("Cleaned Outliers (> |3|):", sum(abs(r_df_clean$resid) > 3), "\n")
}
cat("\nInitial Total Observations:", initial_total_data, "\n")
cat("Initial Outliers (> |3|):", initial_total_outliers, " (", round(initial_outlier_percentage, 2), "%)\n")
cat("Initial Outlier Summary by Industry and Prompt:\n")
print(initial_outlier_summary)
cat("\nInitial Top 10 Residuals:\n")
print(initial_top_residuals)
sink()

# Return results for Python
list(
    initial_summary = initial_summary,
    clean_summary = if (exists("clean_summary")) clean_summary else NULL,
    initial_outliers = initial_outliers,
    initial_outlier_percentage = initial_outlier_percentage,
    initial_top_residuals = initial_top_residuals,
    clean_observations = if (exists("r_df_clean")) nrow(r_df_clean) else initial_total_data
)
'''

# Execute R script
result = r(r_script)

# Extract results in Python
initial_summary = result.rx2('initial_summary')
initial_outlier_percentage = result.rx2('initial_outlier_percentage')[0]
initial_top_residuals = result.rx2('initial_top_residuals')
clean_observations = result.rx2('clean_observations')[0]

# Print Python-side summary
print(f"Model Summaries saved to glmm_summary_scalable.txt")
print(f"Initial Outlier Percentage (> |3|): {initial_outlier_percentage:.2f}%")
print(f"Number of Initial Outliers: {len(result.rx2('initial_outliers').rx2('resid'))}")
print(f"Cleaned Observations: {clean_observations}")