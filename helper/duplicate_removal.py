import argparse
from pathlib import Path

import pandas as pd


def remove_duplicate_reactions(df):
    """Remove duplicate reactant/product reaction rows while keeping the first occurrence."""
    if df is None:
        return df

    required_cols = ["reactants", "products"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_cleaned = df.copy()

    for col in required_cols:
        df_cleaned[col] = df_cleaned[col].map(lambda value: str(value).strip() if pd.notna(value) else "")

    df_cleaned = df_cleaned[(df_cleaned["reactants"] != "") & (df_cleaned["products"] != "")]
    df_cleaned = df_cleaned.drop_duplicates(subset=required_cols, keep="first").reset_index(drop=True)

    initial_count = len(df)
    duplicates_removed = initial_count - len(df_cleaned)

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Remaining reactions: {len(df_cleaned)}")

    return df_cleaned


def remove_duplicate_reactions_from_file(input_path, output_path=None):
    """Read a CSV file, remove duplicate reaction rows, and save a cleaned copy."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    cleaned = remove_duplicate_reactions(df)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_deduplicated{input_path.suffix}")
    else:
        output_path = Path(output_path)

    cleaned.to_csv(output_path, index=False)
    print(f"Saved deduplicated file to: {output_path}")
    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate reactions from a CSV file.")
    input_path = "data/uspto50k/processed.csv"
    output_path = "data/uspto50k/processed_deduplicated.csv"

    remove_duplicate_reactions_from_file(input_path, output_path)