import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib

RAW_CSV_PATH = 'data/raw/weather_data.csv'
DASH = '-'
NUM_DASHES = 10
DASH_HEADER = "\n" + DASH * NUM_DASHES + "\n"

class DataProcessor:
    def __init__(self, csv_path=RAW_CSV_PATH):
        self.df = pd.read_csv(csv_path)
        self.raw_df = self.df.copy()

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.preprocess_data()
        self.save_processed_data()

        day_type_classes = self.df['day_type'].unique()
        day_type_counts = self.df['day_type'].value_counts()
        plt.bar(day_type_classes, day_type_counts)

    def preprocess_data(self):
        self.replace_nan()
        self.remove_duplicates()
        self.create_subsets()
        self.normalize_numeric_columns()
    def replace_nan(self):
        # Replace any NaN/None feature values with the column's median
        for column in self.df.columns[:-1]:
            self.df[column] = self.df[column].fillna(self.df[column].median())
        # Remove any unlabelled instances
        self.df = self.df.dropna(subset=['day_type'])
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
    def normalize_numeric_columns(self):
        numeric_cols = self.df.columns[:-1]
        scaler = MinMaxScaler().fit(self.df_train[numeric_cols])
        for split in (self.df_train, self.df_val, self.df_test):
            split[numeric_cols] = scaler.transform(split[numeric_cols])
        self.df[numeric_cols] = scaler.transform(self.df[numeric_cols])
        joblib.dump(scaler, "models/scaler.pkl")

    def create_subsets(self):
        train, temp = train_test_split(self.df, test_size=0.30,stratify=self.df['day_type'], random_state=42)
        val, test = train_test_split(temp, test_size=0.50,stratify=temp['day_type'], random_state=42)
        self.df_train, self.df_val, self.df_test = train.copy(), val.copy(), test.copy()
    def save_processed_data(self):
        self.df_train.to_csv('data/processed/weather_data_train.csv', index=False)
        self.df_val.to_csv('data/processed/weather_data_val.csv', index=False)
        self.df_test.to_csv('data/processed/weather_data_test.csv', index=False)

if __name__ == "__main__":
    data_processor = DataProcessor()




"""
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
"""

# Create training set (70%), test set (15%), validation set (15%)
"""
df_size = len(df)
train_size = int(df_size*0.7)
test_size, val_size = [df_size-train_size, df_size-train_size]

df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:val_size]
df_test = df.iloc[val_size:]
"""


