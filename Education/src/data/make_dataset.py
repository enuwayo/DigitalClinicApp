import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["target"])  # Drop rows with NaNs in target
    df['target'] = df['target'].map({'Dropout': 0, 'Graduate': 1})
    df = df.dropna()  # Drop any remaining rows with NaNs
    return df

