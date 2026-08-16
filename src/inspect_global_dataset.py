import pandas as pd

print("=" * 70)
print("MINDPULSE - GLOBAL MENTAL HEALTH DATASET INSPECTION")
print("=" * 70)

file_path = "data/raw/Global_Mental_Health_Lifestyle_Survey.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    
    values = df[column].dropna().unique()
    
    if len(values) <= 20:
        print(values)
    else:
        print(f"{len(values)} unique values")

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe(include="all").T)

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)