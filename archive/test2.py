import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_json("results/task1/task1_data.json", lines=True)
#replace llama3.2 in the model column with llama3b
df['model'] = df['model'].replace('llama3.2', 'llama3b').replace('llama3.3', 'llama70b')
#replace llama3.2 in the model column with llama3b

#replace llama3.2 in the model column with llama3b
df['prompt'] = df['prompt'].replace('no', 'benchmark').replace('prompt1', 'persona')

df.drop(columns=['all_fit'], inplace=True)

df.rename(
    columns={
        'c_all_fit_hits': 'correct_match',
        'r_all_fit_hits': 'related_match',
        'r_all_fit_recall': 'related_recall'
    },
    inplace=True
)

code_path = "datasets/occupation.xlsx"
zone_path = "datasets/job_zone.xlsx"
# Read the Excel file
occupation = pd.read_excel(code_path)
occupation.drop(columns=['Description'], inplace=True)
occupation.columns = ['code', 'title']

zone = pd.read_excel(zone_path)
zone = zone[['Title', 'Job Zone']]
zone.columns = ['title', 'zone']

# Merge the occupation data with the main dataframe
df = df.merge(occupation, left_on='title', right_on='title', how='left')
df = df.merge(zone, left_on='title', right_on='title', how='left')
df["area"] = df["code"].apply(lambda x: x.split("-")[0])

df = df[df['model'] == 'mistral']

print("Initial DataFrame Info:")
df.info()
print("\nFirst 5 rows of data:")
print(df.head())

# --- Data Type Conversions and Preparations ---
df['prompt'] = pd.Categorical(df['prompt'], categories=['benchmark', 'persona'], ordered=False)
df['zone'] = pd.Categorical(df['zone'])
df['area'] = pd.Categorical(df['area'])
df['code_id'] = pd.Categorical(df['code']) # Unique ID for random effects

df['correct_match'] = df['correct_match'].astype(int)
df['related_match'] = df['related_match'].astype(int)
df['num_related_matches'] = (df['related_recall'] * 10).round().astype(int) # Assuming max 10 related jobs

print("Initial DataFrame Info:")
df.info()
print("\nFirst 5 rows of data:")
print(df.head())
print(f"\nUnique codes: {df['code_id'].nunique()}")


# --- Corrected Helper Function for GEE Model Fitting ---
def run_gee_model(df, outcome_var, family_class, link_str, offset_var=None):
    """
    Runs a GEE model for the specified outcome and prints a summary.
    Returns the results object for further analysis.
    """
    # Use C() for categorical variables explicitly in the formula
    formula = f"{outcome_var} ~ C(prompt, Treatment(reference='benchmark')) * C(zone) + C(prompt, Treatment(reference='benchmark')) * C(area)"

    print(f"\n--- Running GEE Model for: {outcome_var} ---")
    print(f"Formula: {formula}")
    print(f"Family: {family_class.__name__}, Link: {link_str}")

    try:
        # Instantiate the family object correctly
        current_family = family_class(link=link_str)

        if offset_var:
             model = smf.gee(formula, data=df, groups=df['code_id'],
                              family=current_family,
                              cov_struct=sm.cov_struct.Exchangeable(),
                              offset=df[offset_var])
        else:
            model = smf.gee(formula, data=df, groups=df['code_id'],
                            family=current_family,
                            cov_struct=sm.cov_struct.Exchangeable()) # Exchangeable for within-group correlation

        results = model.fit()
        print(results.summary())

        # Extract coefficients and p-values for reporting
        print("\nSelected Coefficients (Exponentiated if link is logit/log):")
        # Check if the link is a log or logit link for exponentiation
        if current_family.link.name in ['logit', 'log']:
            print(np.exp(results.params))
        else:
            print(results.params)
        print("P-values for coefficients:")
        print(results.pvalues)

        return results # RETURN THE RESULTS OBJECT
    except Exception as e:
        print(f"Error fitting model for {outcome_var}: {e}")
        return None # Return None if fitting fails

# --- Running the Models for Each Outcome ---

# 1. Model for correct_match (Binary Outcome)
results_correct_match = run_gee_model(df, 'correct_match', sm.families.Binomial, 'logit')

# 2. Model for related_match (Binary Outcome)
results_related_match = run_gee_model(df, 'related_match', sm.families.Binomial, 'logit')

# 3. Model for num_related_matches (Count Outcome, 0-10)
# Check for overdispersion first
mean_num_related = df['num_related_matches'].mean()
var_num_related = df['num_related_matches'].var()
print(f"\nMean of num_related_matches: {mean_num_related:.2f}")
print(f"Variance of num_related_matches: {var_num_related:.2f}")
if var_num_related > mean_num_related * 1.5:
    print("Likely overdispersion detected. Using NegativeBinomial family.")
    # Use NegativeBinomial for num_related_matches due to likely overdispersion
    results_num_related_matches = run_gee_model(df, 'num_related_matches', sm.families.NegativeBinomial, 'log')
else:
    print("Overdispersion not strongly indicated. Using Poisson family.")
    # Using Poisson if no strong overdispersion
    results_num_related_matches = run_gee_model(df, 'num_related_matches', sm.families.Poisson, 'log')

# --- Post-Analysis: Predictions and Interaction Visualization (Conceptual) ---
# Now 'results_correct_match', 'results_related_match', 'results_num_related_matches'
# hold the fitted model objects, so you can perform predictions.

# Example for correct_match predictions by zone
if results_correct_match:
    print("\n--- Generating Predictions for correct_match by Zone ---")
    # Create new data to predict on, varying prompt and zone, holding area constant
    # Choose a representative area (e.g., the mode)
    representative_area = df['area'].mode()[0]
    
    zones = df['zone'].cat.categories
    prompts = df['prompt'].cat.categories

    # Create all combinations for prediction
    new_data_zone = pd.DataFrame({
        'prompt': np.tile(prompts, len(zones)),
        'zone': np.repeat(zones, len(prompts)),
        'area': representative_area,
        'code_id': df['code_id'].sample(1).iloc[0] # Dummy for model, actual value doesn't affect population-averaged predictions
    })
    
    # Ensure categorical types are correct for prediction
    new_data_zone['prompt'] = pd.Categorical(new_data_zone['prompt'], categories=prompts, ordered=False)
    new_data_zone['zone'] = pd.Categorical(new_data_zone['zone'], categories=zones)
    new_data_zone['area'] = pd.Categorical(new_data_zone['area'], categories=df['area'].cat.categories)
    new_data_zone['code_id'] = pd.Categorical(new_data_zone['code_id'], categories=df['code_id'].cat.categories)


    # Get predictions
    predicted_frame = results_correct_match.get_prediction(new_data_zone).summary_frame()
    new_data_zone = pd.concat([new_data_zone, predicted_frame[['mean', 'obs_ci_lower', 'obs_ci_upper']]], axis=1)

    print(new_data_zone[['prompt', 'zone', 'mean', 'obs_ci_lower', 'obs_ci_upper']])

    # You can now plot these 'mean' values to visualize the interaction.
    # For example, using matplotlib/seaborn:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=new_data_zone, x='zone', y='mean', hue='prompt', dodge=True, errorbar=('ci', 95))
    plt.title(f'Predicted Correct Match Probability by Job Zone (Area: {representative_area})')
    plt.xlabel('Job Zone')
    plt.ylabel('Predicted Probability of Correct Match')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# You would repeat similar prediction and plotting steps for 'area' and for the other models.