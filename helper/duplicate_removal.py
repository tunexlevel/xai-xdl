import argparse
from pathlib import Path

import time

import pandas as pd


def remove_duplicate_reactions(df, sample_size=10):
    """Remove duplicate reactant/product reaction rows while keeping the first occurrence."""
    if df is None:
        return df

    required_cols = ["reactants", "products"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_normalized = df.copy()
    for col in required_cols:
        df_normalized[col] = df_normalized[col].map(lambda value: str(value).strip() if pd.notna(value) else "")

    df_normalized = df_normalized[(df_normalized["reactants"] != "") & (df_normalized["products"] != "")].copy()

    duplicate_mask = df_normalized.duplicated(subset=required_cols, keep="first")
    removed_duplicates = df_normalized[duplicate_mask].head(sample_size).reset_index(drop=True)
    df_cleaned = df_normalized[~duplicate_mask].reset_index(drop=True)

    initial_count = len(df)
    duplicates_removed = initial_count - len(df_cleaned)


    print(f"Initial reactions: {initial_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Remaining reactions: {len(df_cleaned)}")

    return df_cleaned, removed_duplicates


def remove_duplicate_reactions_from_file(input_path, output_path=None, sample_output_path=None, sample_size=10):
    """Read a CSV file, remove duplicate reaction rows, and save a cleaned copy."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    cleaned, removed_duplicates = remove_duplicate_reactions(df, sample_size=sample_size)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_deduplicated{input_path.suffix}")
    else:
        output_path = Path(output_path)

    cleaned.to_csv(output_path, index=False)
    print(f"Saved deduplicated file to: {output_path}")

    if sample_output_path is None:
        sample_output_path = input_path.with_name(f"{input_path.stem}_duplicate_samples{input_path.suffix}")
    else:
        sample_output_path = Path(sample_output_path)

    if not removed_duplicates.empty:
        removed_duplicates.to_csv(sample_output_path, index=False)
        print(f"Saved sample duplicates to: {sample_output_path}")

    return cleaned, removed_duplicates


if __name__ == "__main__":
    
    start_message = "Starting duplicate removal process..."
    print('' + '=' * len(start_message))
    print(start_message)
    print('' + '=' * len(start_message))
    
    parser = argparse.ArgumentParser(description="Remove duplicate reactions from a CSV file.")
    parser.add_argument("input_path", nargs="?", default="data/uspto_mit_unmapped.csv")
    parser.add_argument("-o", "--output", dest="output_path", default="data/uspto_mit_unmapped_deduplicated.csv")
    parser.add_argument("--samples", dest="sample_output_path", default="data/uspto_mit_unmapped_duplicate_samples.csv")
    args = parser.parse_args()

    start_time = time.time()
    remove_duplicate_reactions_from_file(args.input_path, args.output_path, args.sample_output_path)
    elapsed_time = time.time() - start_time
    print(f"Duplicate removal completed in {elapsed_time:.2f} seconds.")