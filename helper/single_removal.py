import argparse
import time
from pathlib import Path

import pandas as pd


def remove_single_reaction(df, sample_size=10):
    """
    Remove reaction rows where the reactants column contains only
    a single reactant.

    Reactants are assumed to be separated by '.' in SMILES.

    Examples:
        CCO              -> single reactant -> REMOVE
        CCO.CN           -> two reactants   -> KEEP
        CCO.CN.O         -> three reactants -> KEEP
    """

    if df is None:
        return df, pd.DataFrame()

    required_cols = ["reactants", "products"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_normalized = df.copy()

    # Normalize reactants and products
    for col in required_cols:
        df_normalized[col] = df_normalized[col].map(
            lambda value: str(value).strip()
            if pd.notna(value)
            else ""
        )

    # Remove rows with empty reactants or products
    df_normalized = df_normalized[
        (df_normalized["reactants"] != "") &
        (df_normalized["products"] != "")
    ].copy()

    # Count the number of reactants in each reaction.
    # Multiple reactants are separated by '.'
    df_normalized["reactant_count"] = (
        df_normalized["reactants"]
        .str.split(".")
        .str.len()
    )

    # Identify rows containing exactly one reactant
    single_mask = df_normalized["reactant_count"] == 1

    # Keep a sample of removed single-reactant reactions
    removed_singles = (
        df_normalized[single_mask]
        .head(sample_size)
        .drop(columns=["reactant_count"])
        .reset_index(drop=True)
    )

    # Remove ALL single-reactant reactions
    df_cleaned = (
        df_normalized[~single_mask]
        .drop(columns=["reactant_count"])
        .reset_index(drop=True)
    )

    # Statistics
    initial_count = len(df)
    valid_count = len(df_normalized)
    singles_removed = single_mask.sum()
    remaining_count = len(df_cleaned)

    print(f"Initial reactions: {initial_count}")
    print(f"Valid reactions: {valid_count}")
    print(f"Single-reactant reactions removed: {singles_removed}")
    print(f"Remaining reactions: {remaining_count}")

    return df_cleaned, removed_singles


def remove_single_reaction_from_file(
    input_path,
    output_path=None,
    sample_output_path=None,
    sample_size=10
):
    """
    Read a CSV file, remove single-reactant reaction rows,
    and save the cleaned copy.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print(f"Reading: {input_path}")

    df = pd.read_csv(input_path)

    cleaned, removed_singles = remove_single_reaction(
        df,
        sample_size=sample_size
    )

    # Output cleaned dataset
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_multi_reactant{input_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    cleaned.to_csv(output_path, index=False)

    print(f"Saved cleaned file to: {output_path}")

    # Output sample of removed reactions
    if sample_output_path is None:
        sample_output_path = input_path.with_name(
            f"{input_path.stem}_single_reactant_samples{input_path.suffix}"
        )
    else:
        sample_output_path = Path(sample_output_path)

    if not removed_singles.empty:
        removed_singles.to_csv(
            sample_output_path,
            index=False
        )

        print(
            f"Saved sample single-reactant reactions to: "
            f"{sample_output_path}"
        )
    else:
        print("No single-reactant reactions were found.")

    return cleaned, removed_singles


if __name__ == "__main__":

    start_message = "Starting single-reactant removal process..."

    print("=" * len(start_message))
    print(start_message)
    print("=" * len(start_message))

    parser = argparse.ArgumentParser(
        description="Remove single-reactant reactions from a CSV file."
    )

    parser.add_argument(
        "input_path",
        nargs="?",
        default="data/raw/final/reactant_product_deduplicated.csv"
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="data/raw/final/reactant_product_multi_reactant.csv"
    )

    parser.add_argument(
        "--samples",
        dest="sample_output_path",
        default="data/raw/final/reactant_product_single_reactant_samples.csv"
    )

    parser.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=10
    )

    args = parser.parse_args()

    start_time = time.time()

    remove_single_reaction_from_file(
        args.input_path,
        args.output_path,
        args.sample_output_path,
        args.sample_size
    )

    elapsed_time = time.time() - start_time

    print(
        f"Single-reactant removal completed "
        f"in {elapsed_time:.2f} seconds."
    )

    