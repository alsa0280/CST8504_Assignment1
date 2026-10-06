import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_CSV_PATH = '../data/raw/weather_data.csv'
DASH = '-'
NUM_DASHES = 10
DASH_HEADER = "\n" + DASH * NUM_DASHES + "\n"

df = pd.read_csv(RAW_CSV_PATH)

# First 5 records
print(f"\nFIRST 5 RECORDS {DASH_HEADER} {df.head()}")

# Data types of columns
print(f"\nDATA TYPES {DASH_HEADER} {df.dtypes}")

# Summary statistics
print(f"\nSUMMARY STATISTICS {DASH_HEADER} {df.describe()}")

# Check how many NaN or NaT values are in each column
print(f"\nHOW MANY NULL VALUES PER COLUMN? {DASH_HEADER} {df.isna().sum()}")
# Replace any feature NaN values with their columnal median
for column in df.columns[:-1]:
    df[column] = df[column].fillna(df[column].median())
# Remove any unlabelled instances
df = df.dropna(subset=['day_type'])

# Check for duplicates
print(f"\nHOW MANY DUPLICATES IN DATASET? {DASH_HEADER} {df.duplicated().sum()}")
# Remove any duplicates
df = df.drop_duplicates()

# Scale numeric features
scaler = MinMaxScaler()
numeric_cols = df.columns[:-1]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
