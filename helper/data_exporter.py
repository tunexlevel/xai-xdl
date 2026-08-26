import pandas as pd
from rdkit import Chem


def load_uspto_file(path, max_samples=None, type=None):
    try:
        df = pd.read_csv(path, sep=",", skiprows=2)
        
        if type == "ocr":
            # Handle CSV format as index [0] and [1]
            if len(df.columns) >= 2:
                df.columns = ['reactants', 'products']
            else:
                raise ValueError(f"Expected columns 'reactants' and 'products' not found in {path}")
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

def process_uspto_file(input_path, output_path, max_samples=None, type=None):
    df = load_uspto_file(input_path, max_samples, type)
    if df is None:
        print("No data to process.")
        return
    
    # Remove atom mapping from reactants and products
    df['reactants'] = df['reactants'].apply(clean_smiles).apply(remove_atom_mapping)
    df['products'] = df['products'].apply(clean_smiles).apply(remove_atom_mapping)

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
        
        
        
source_file = "data/raw/output/train.csv"
# target_file = "data/uspto50k/processed_ocr.csv"


process_chemxai_file(source_file, "data/uspto50k/processed_train.csv")

# val_df = load_chemxai_file(
#     source_file,
#     split="validation"
# )

# test_df = load_chemxai_file(
#     source_file,
#     split="test"
# )

# process_uspto_file(source_file, target_file, max_samples=100000, type="ocr")