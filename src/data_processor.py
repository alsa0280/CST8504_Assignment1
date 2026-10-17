import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

RAW_CSV_PATH = 'data/raw/weather_data.csv'

class DataProcessor:
    def __init__(self, csv_path=RAW_CSV_PATH):
        self.df = pd.read_csv(csv_path)
        self.raw_df = self.df.copy()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.preprocess_data()
        self.save_processed_data()
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
