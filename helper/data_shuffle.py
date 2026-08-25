import pandas as pd


SOURCE_FILE = "data/uspto50k/processed_ocr.csv"
OUTPUT_FILE = "data/uspto50k/processed_ocr_shuffled.csv"


df = pd.read_csv(SOURCE_FILE)

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset shuffled successfully.")