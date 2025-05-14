import pandas as pd
import numpy as np
from pymer4.models import Lmer
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, globalenv
from rpy2.robjects.packages import importr
import statsmodels.api as sm  # For the qqplot
pandas2ri.activate()

# --- Define file names for saving results ---
summary_file = "model_summary"
coefficients_file = "model_coefficients"
random_effects_file = "random_effects"

# Print versions for debugging
import pymer4
print(f"pymer4 version: {pymer4.__version__}")

# Set R_HOME and PATH for pymer4
r_path = r'C:\Program Files\R\R-4.5.0'
os.environ['R_HOME'] = r_path
os.environ['PATH'] = r_path + r'\bin;' + os.environ.get('PATH', '')

lme4 = importr('lme4')
print(f"lme4 version: {robjects.r('packageVersion(\"lme4\")')[0]}")

### provide your code here for plotting###
def assess_normality(data, title):
    """
    Performs visual and statistical tests for normality on a given dataset.
    """
    print(f"\n--- Assessing Normality for: {title} ---")

    # 1. Visual Inspection
    plt.figure(figsize=(12, 5))

    # QQ-Plot
    plt.subplot(1, 2, 1)
    sm.qqplot(data, stats.norm, fit=True, line='45')
    plt.title(f'QQ-Plot for {title}')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(data, bins='auto', density=True, alpha=0.7, color='skyblue')
    mu, std = np.mean(data), np.std(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Histogram for {title}')

    plt.tight_layout()
    plt.show()

    # 2. Statistical Tests
    if len(data) > 3: # Shapiro-Wilk requires more than 3 samples
        shapiro_test = stats.shapiro(data)
        print(f"Shapiro-Wilk Test for {title}:")
        print(f"  Statistic (W): {shapiro_test.statistic:.4f}")
        print(f"  P-value: {shapiro_test.pvalue:.4f}")
        if shapiro_test.pvalue > 0.05:
            print("  Conclusion: The data appears to be normally distributed (fail to reject null hypothesis).")
        else:
            print("  Conclusion: The data does not appear to be normally distributed (reject null hypothesis).")
    else:
        print(f"Shapiro-Wilk Test for {title}: Not performed (n <= 3)")

    if len(data) > 0:
        # Kolmogorov-Smirnov Test (with Lilliefors correction - implemented via statsmodels)
        lilliefors_test = sm.stats.diagnostic.lilliefors(data, dist='norm')
        print(f"\nLilliefors Test for {title}:")
        print(f"  Statistic (D): {lilliefors_test[0]:.4f}")
        print(f"  P-value: {lilliefors_test[1]:.4f}")
        if lilliefors_test[1] > 0.05:
            print("  Conclusion: The data appears to be normally distributed (fail to reject null hypothesis).")
        else:
            print("  Conclusion: The data does not appear to be normally distributed (reject null hypothesis).")

        # Anderson-Darling Test
        anderson_test = stats.anderson(data, dist='norm')
        print(f"\nAnderson-Darling Test for {title}:")
        print(f"  Statistic (A): {anderson_test.statistic:.4f}")
        print("  Critical Values:", anderson_test.critical_values)
        print("  Significance Levels:", anderson_test.significance_level)
        for i in range(len(anderson_test.critical_values)):
            if anderson_test.statistic < anderson_test.critical_values[i]:
                print(f"  At {anderson_test.significance_level[i]}% significance level, data looks normal (fail to reject null hypothesis).")
                break
            elif i == len(anderson_test.critical_values) - 1:
                print(f"  At all tested significance levels, data does not look normal (reject null hypothesis).")
    else:
        print(f"Normality tests for {title}: No data to test.")

# 1. Load and preprocess JSON data
json_path = r'd:\OneDrive - Universität Mannheim\MMM\Master Thesis\Teapot\results\task1\task1_data.json'
try:
    df = pd.read_json(json_path, lines=True)
except FileNotFoundError:
    print(f"Error: File {json_path} not found. Check path.")
    raise

# Filter for mistral model


# Replace values in prompt column and convert to category
df['model'] = df['model'].replace('llama3.2', 'llama3b').replace('llama3.3', 'llama70b')
df['prompt'] = df['prompt'].replace('no', 'benchmark').replace('prompt1', 'persona')
df['prompt'] = df['prompt'].astype('category')
df = df[df['model'] == 'deepseek-r1']

# Drop unnecessary column
df.drop(columns=['all_fit'], inplace=True)

# Rename columns
df.rename(
    columns={
        'c_all_fit_hits': 'correct_match',
        'r_all_fit_hits': 'related_match',
        'r_all_fit_recall': 'related_recall'
    },
    inplace=True
)

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

# Merge with main dataframe
df = df.merge(occupation, left_on='title', right_on='title', how='left')
df = df.merge(zone, left_on='title', right_on='title', how='left')

# Create industry column
df["industry"] = df["code"].apply(lambda x: x.split("-")[0] if pd.notnull(x) else np.nan)
df["industry"] = df["industry"].fillna('Unknown')

# Convert to categorical *before* sending to R
df['prompt'] = pd.Categorical(df['prompt'], categories=['benchmark', 'persona'], ordered=True)
df['industry'] = pd.Categorical(df['industry']) # Convert industry to category
df["zone"] = df["zone"].astype(str)  # Convert zone to string
df['zone'] = pd.Categorical(df['zone']) # Convert title to category


# Store original industry categories
original_industry_categories = df['industry'].cat.categories.tolist()
print("Original industry categories:", original_industry_categories)

# Store original industry categories
original_zone_categories = df['zone'].cat.categories.tolist()
print("Original zone categories:", original_zone_categories)

# Convert DataFrame to R and assign to global environment
globalenv['r_df'] = pandas2ri.py2rpy(df)

# Set industry as factor with '51' as first level in R
r('''
if ("51" %in% levels(r_df$industry)) {
    r_df$industry <- factor(r_df$industry, levels = c("51", setdiff(levels(r_df$industry), "51")))
} else {
    message("Warning: '51' not found in industry levels. Using default order.")
}
''')

# --- Fit the model using the r_df in the R environment ---
r_model_formula = 'correct_match ~ prompt * industry + (1 | title)'
r_code_fit = f'r_model <- lme4::glmer({r_model_formula}, data = r_df, family = binomial, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))'
r(r_code_fit)

# --- Check if r_model was created ---
r_model_exists = r('exists("r_model")')[0]
print(f"\nDid R create 'r_model'? {r_model_exists}")

fixed_effects_industry_pandas = None  # Initialize

# --- Extract fixed effects coefficients from the R model as a data frame ---
if r_model_exists:
    # --- Save the full summary to a text file ---
    r(f'sink("{summary_file}_industry.txt")')
    r('print(summary(r_model))')
    r('sink()')
    print(f"\nR Model summary saved to: {summary_file}")

    # --- Save the coefficients to a text file (tab-separated) ---
    r(f'write.table(coef(summary(r_model)), file="{coefficients_file}_industry.txt", sep="\\t", quote=FALSE, row.names=TRUE)')
    print(f"Model coefficients saved to: {coefficients_file}")

    # --- Save the random effects variance to a text file ---
    r(f'sink("{random_effects_file}_industry.txt")')
    r('print(VarCorr(r_model))')
    r('sink()')
    print(f"Random effects variances saved to: {random_effects_file}")

    # --- Extract coefficients to Pandas DataFrame for plotting ---
    coefficients_r = r('as.data.frame(summary(r_model)$coefficients)')
    fixed_effects_industry_pandas = pandas2ri.rpy2py(coefficients_r)

else:
    print("\nError: 'r_model' was not found in the R environment. Check for errors during model fitting.")

if fixed_effects_industry_pandas is not None:
    # Assess normality of industry-related fixed effects coefficients
    industry_coefficients = fixed_effects_industry_pandas.loc[fixed_effects_industry_pandas.index.str.startswith('industry'), 'Estimate'].values
    assess_normality(industry_coefficients, "Fixed Effects Coefficients (Zone Model)")

    # You can also assess the normality of the interaction terms with zone
    interaction_industry_coefficients = fixed_effects_industry_pandas.loc[fixed_effects_industry_pandas.index.str.startswith('prompt.L:industry'), 'Estimate'].values
    assess_normality(interaction_industry_coefficients, "Interaction Coefficients (Prompt x industry)")

# Set zone as factor with '2' as first level in R
r('''
if ("2" %in% levels(r_df$zone)) {
    r_df$zone <- factor(r_df$zone, levels = c("2", setdiff(levels(r_df$zone), "2")))
} else {
    message("Warning: '2' not found in zone levels. Using default order.")
}
''')

# --- Fit the model using the r_df in the R environment ---
r_model_formula = 'correct_match ~ prompt * zone + (1 | title)'
r_code_fit = f'r_model <- lme4::glmer({r_model_formula}, data = r_df, family = binomial, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))'
r(r_code_fit)

# --- Check if r_model was created ---
r_model_exists = r('exists("r_model")')[0]
print(f"\nDid R create 'r_model'? {r_model_exists}")

fixed_effects_zone_pandas = None  # Initialize

# --- Extract fixed effects coefficients from the R model as a data frame ---
if r_model_exists:
    # --- Save the full summary to a text file ---
    r(f'sink("{summary_file}_zone.txt")')
    r('print(summary(r_model))')
    r('sink()')
    print(f"\nR Model summary saved to: {summary_file}_zone.txt")

    # --- Save the coefficients to a text file (tab-separated) ---
    r(f'write.table(coef(summary(r_model)), file="{coefficients_file}_zone.txt", sep="\\t", quote=FALSE, row.names=TRUE)')
    print(f"Model coefficients saved to: {coefficients_file}_zone.txt")

    # --- Save the random effects variance to a text file ---
    r(f'sink("{random_effects_file}_zone.txt")')
    r('print(VarCorr(r_model))')
    r('sink()')
    print(f"Random effects variances saved to: {random_effects_file}_zone.txt")

    # --- Extract coefficients to Pandas DataFrame for plotting ---
    coefficients_r = r('as.data.frame(summary(r_model)$coefficients)')
    fixed_effects_zone_pandas = pandas2ri.rpy2py(coefficients_r)

else:
    print("\nError: 'r_model' was not found in the R environment. Check for errors during model fitting.")




if fixed_effects_zone_pandas is not None:
    # Assess normality of Zone-related fixed effects coefficients
    industry_coefficients = fixed_effects_zone_pandas.loc[fixed_effects_zone_pandas.index.str.startswith('zone'), 'Estimate'].values
    assess_normality(industry_coefficients, "Fixed Effects Coefficients (Zone Model)")

    # You can also assess the normality of the interaction terms with zone
    interaction_industry_coefficients = fixed_effects_zone_pandas.loc[fixed_effects_zone_pandas.index.str.startswith('prompt.L:zone'), 'Estimate'].values
    assess_normality(interaction_industry_coefficients, "Interaction Coefficients (Prompt x Zone)")

print("\nAnalysis complete. Results saved to text files and normality assessed.")