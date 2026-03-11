import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

TRAIN_FILE = BASE_DIR / "train.csv"
TEST_FILE = BASE_DIR / "test.csv"

def ingest_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    return train_df, test_df

if __name__ == "__main__":
    ingest_data()