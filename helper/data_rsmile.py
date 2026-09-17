
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from helper.data_loader import load_uspto_file
from helper.rsmile_dataframe import prepare_rsmiles_dataframe


FILE_PATH = "data/raw/ocr/ocrtrain_multi_reactant_processed.csv"  # ROOT / "data" / "raw" / "train-data" / f"{DATASET_NAME}.csv"


raw_df = load_uspto_file(FILE_PATH)

aligned_df = prepare_rsmiles_dataframe(raw_df, augment_times=1)

# Save the aligned dataframe to a CSV file
aligned_df.to_csv("data/raw/ocr/ocrtrain_multi_reactant_aligned.csv", index=False)

print(f"Aligned dataframe saved to {FILE_PATH}")