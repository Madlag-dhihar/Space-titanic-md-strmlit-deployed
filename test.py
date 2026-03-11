import pandas as pd
from feature_en import feature_engineering

df = pd.read_csv("train.csv")

df = feature_engineering(df)

print(df.shape)
