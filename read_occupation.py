import pandas as pd

df = pd.read_csv('occupation.txt', sep='\t')

print(df.head(10))
