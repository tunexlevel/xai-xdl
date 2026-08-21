from data_loader import load_uspto_file
import pandas as pd


df = load_uspto_file('data/uspto50k/raw_train.csv', max_samples=5)

json_str = df.to_json(orient='records', indent=2)

print(json_str)
