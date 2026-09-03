import time

import pandas as pd
from rdkit import Chem
from utils import map_smiles


def load_uspto_file(path, max_samples=None, type=None):
    try:
        df = pd.read_csv(path, sep=",", skiprows=2)
        
        if type == "ocr":
            # Handle CSV format as index [0] and [1]
            if len(df.columns) >= 2:
                df.columns = ['reactants', 'products']
            else:
                raise ValueError(f"Expected columns 'reactants' and 'products' not found in {path}")
        elif type == "mit":
            if len(df.columns) >= 1:
                    df.columns = ['reactions']
                    
                    split_smiles = df['reactions'].str.split(">", expand=True)
                    
                    if split_smiles.shape[1] >= 3:
                        df['reactants'] = split_smiles[0].str.strip()
                        df['products'] = split_smiles[2].str.strip()  # skip reagents
                    else:
                        raise ValueError(f"Unexpected reaction column format in {path}")
            else:
                raise ValueError(f"Unsupported file format in {path}")
        
        else:
            # Handle raw USPTO CSV format: id,class,reactants>reagents>production
            if len(df.columns) >= 3:
                smiles_col = df.columns[2]
                split_smiles = df[smiles_col].astype(str).str.split(">", expand=True)
                if split_smiles.shape[1] >= 3:
                    df['reactants'] = split_smiles[0].str.strip()
                    df['products'] = split_smiles[2].str.strip()  # skip reagents
                else:
                    raise ValueError(f"Unexpected reaction column format in {path}")
            else:
                raise ValueError(f"Unsupported file format in {path}")

        # Drop missing data
        df = df.dropna(subset=['reactants', 'products'])

        if max_samples:
            df = df.head(max_samples)
        return df[['reactants', 'products']]
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return
    

def remove_atom_mapping(smiles):
    try:
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        unmapped_smiles = Chem.MolToSmiles(mol)
        return unmapped_smiles
    except Exception as e:
        print(f"Error removing atom mapping from SMILES {smiles}: {e}")
        return None

def clean_smiles(smiles):
    # Remove quotes and all whitespace
    return smiles.replace('"', '').replace("'", '').replace(' ', '').strip()

def process_uspto_file(input_path, output_path, max_samples=None, type=None, remove_mapping=True):
    df = load_uspto_file(input_path, max_samples, type)
    if df is None:
        print("No data to process.")
        return
    
    if remove_mapping:
        # Remove atom mapping from reactants and products
        df['reactants'] = df['reactants'].apply(clean_smiles).apply(remove_atom_mapping)
        df['products'] = df['products'].apply(clean_smiles).apply(remove_atom_mapping)
    else:
        # Clean SMILES while adding atom mapping safely
        # map_smiles should return None on failure rather than crashing
        df['reactants'] = df['reactants'].apply(clean_smiles).apply(map_smiles)
        df['products'] = df['products'].apply(clean_smiles).apply(map_smiles)
            
    # Drop rows that failed parsing/mapping
    initial_count = len(df)
    df = df.dropna(subset=['reactants', 'products'])
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} invalid/unmappable reactions.")

    # Save the processed DataFrame to a new CSV file
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")

def process_ocr_file(input_path, output_path, max_samples=None, type=None, remove_mapping=True):
    df = load_uspto_file(input_path, max_samples, type)
    if df is None:
        print("No data to process.")
        return
    
    if remove_mapping:
        # Remove atom mapping from reactants and products
        df['reactants'] = df['reactants'].apply(clean_smiles).apply(remove_atom_mapping)
        df['products'] = df['products'].apply(clean_smiles).apply(remove_atom_mapping)

    else:
        # Clean SMILES without removing atom mapping
        df['reactants'] = df['reactants'].apply(clean_smiles)
        df['products'] = df['products'].apply(clean_smiles)
    
    # Drop rows with None values after mapping removal
    df = df.dropna(subset=['reactants', 'products'])

    # Save the processed DataFrame to a new CSV file
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")

def load_chemxai_file(path, split=None, max_samples=None):
    """
    Load the cleaned ChemXAI reaction dataset.

    Required columns:
        reactants
        products
        split

    For Experiment 1:
        reactants -> products
    """

    df = pd.read_csv(path)

    required_columns = ["reactants", "products", "split"]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep only what Experiment 1 needs
    df = df[required_columns].copy()

    # Select split
    if split is not None:
        df = df[df["split"] == split].copy()

    # Remove missing values
    df = df.dropna(subset=["reactants", "products"])

    # Convert to strings
    df["reactants"] = df["reactants"].astype(str).str.strip()
    df["products"] = df["products"].astype(str).str.strip()

    # Remove empty reactions
    df = df[
        (df["reactants"] != "") &
        (df["products"] != "")
    ]

    # Optional sample limit
    if max_samples is not None:
        df = df.head(max_samples)

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df):,} reactions")
    print(f"Split: {split}")

    return df

def process_chemxai_file(input_path, output_path, max_samples=None, type=None):
    df = load_chemxai_file(input_path)
    if df is None:
        print("No data to process.")
        return
    
    # Remove atom mapping from reactants and products
    # df['reactants'] = df['reactants'].apply(clean_smiles).apply(remove_atom_mapping)
    # df['products'] = df['products'].apply(clean_smiles).apply(remove_atom_mapping)

    # Drop rows with None values after mapping removal
    df = df.dropna(subset=['reactants', 'products'])

    # Save the processed DataFrame to a new CSV file
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")
     
def main(run_type="all"):
    starting_message = f"Starting data processing for run_type: {run_type}"
    
    print(starting_message)
    
    start_time = time.time()
    
    if run_type in ["all", "uspto_mapped"]:
        # USPTO50k Mapped processing
        source_file = "data/raw/uspto50k/raw_train.csv"
        target_file = "data/uspto50k_mapped.csv"
        process_uspto_file(source_file, target_file, type="uspto", remove_mapping=False)

    if run_type in ["all", "uspto_unmapped"]:
        # USPTO50k Unmapped processing
        source_file = "data/raw/uspto50k/raw_train.csv"
        target_file = "data/uspto50k_unmapped.csv"
        process_uspto_file(source_file, target_file,  type="uspto", remove_mapping=True)
    
    if run_type in ["all", "uspto_test_mapped"]:
            # USPTO50k Mapped processing
            source_file = "data/raw/uspto50k/raw_test.csv"
            target_file = "data/uspto50k_mapped_test.csv"
            process_uspto_file(source_file, target_file, type="uspto", remove_mapping=False)
    
    if run_type in ["all", "uspto_test_unmapped"]:
        # USPTO50k Unmapped processing
        source_file = "data/raw/uspto50k/raw_test.csv"
        target_file = "data/uspto50k_unmapped_test.csv"
        process_uspto_file(source_file, target_file, type="uspto", remove_mapping=True)
    
    if run_type in ["all", "chemxai"]:
        # ChemXAI processing
        source_file = "data/raw/chemxai/cleaned_chemxai.csv"
        target_file = "data/chemxai/processed_train.csv"
        process_chemxai_file(source_file, target_file, type="chemxai")

    if run_type in ["all", "ocr"]:
        # OCR processing
        source_file = "data/raw/ocr/ocr_train.csv"
        target_file = "data/ocr/processed_train.csv"
        process_ocr_file(source_file, target_file, type="ocr", remove_mapping=True)
        
        
    if run_type in ["all", "uspto_mit_unmapped"]:
        # OCR processing
        source_file = "data/raw/uspto_mit/USPTO_MIT.csv"
        target_file = "data/uspto_mit_unmapped.csv"
        process_uspto_file(source_file, target_file, type="mit", remove_mapping=True)
        
    
    if run_type in ["all", "uspto_mit_mapped"]:
        # OCR processing
        source_file = "data/uspto_mit_unmapped.csv"
        target_file = "data/uspto_mit_mapped.csv"
        process_uspto_file(source_file, target_file, type="ocr", remove_mapping=False)
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print(f"Data processing completed for run_type: {run_type} in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Change run_type as needed: 
    # "all", "uspto_unmapped", "uspto_mapped", "chemxai",
    # "ocr", "uspto_test_mapped", "uspto_test_unmapped"
    main(run_type="uspto_mit_mapped")
