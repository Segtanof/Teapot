import pandas as pd

#import occupation description
df_occupation = pd.read_csv('occupation.txt', sep='\t')
df_occupation = df_occupation.dropna(subset=["occupation_description"])
df_occupation = df_occupation[["occupation_code", "occupation_name","occupation_description"]]
df_occupation = df_occupation.rename(columns={"occupation_code": "OCC_CODE"})
#print(df_occupation.head())

#import wage stat
df_wage = pd.read_excel("testocc.xlsx")
df_wage = df_wage[["OCC_CODE", "A_MEAN", "O_GROUP"]]
df_wage['OCC_CODE'] = df_wage['OCC_CODE'].str.replace('-', '').astype(int)
df_wage = df_wage[df_wage['O_GROUP'] == "detailed"]
#print(df_wage.head())

#join both dfs together
merged_df = pd.merge(df_occupation, df_wage, on='OCC_CODE', how='inner').drop(columns=["O_GROUP"])
print(merged_df.head())

#pass that to agent




#get the result
