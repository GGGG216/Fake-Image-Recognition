import pandas as pd

df = pd.read_excel("A230.1_BatchDataSheets.xlsx")
print("所有列名:", df.columns.tolist())
print("前5行数据:")
print(df.head())