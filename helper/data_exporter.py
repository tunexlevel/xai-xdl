import pandas as pd
from rdkit import Chem


def load_uspto_file(path, max_samples=None):
    try:
        df = pd.read_csv(path, sep=",", skiprows=2)
        
        # Split SMILES into reactants, reagents, products
        smiles_col = df.columns[2] 
        split_smiles = df[smiles_col].str.split(">", expand=True)
        df['reactants'] = split_smiles[0].str.strip()
        df['products'] = split_smiles[2].str.strip()  # skip reagents

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
    

def process_uspto_file(input_path, output_path, max_samples=None):
    df = load_uspto_file(input_path, max_samples)
    if df is None:
        print("No data to process.")
        return

    # Remove atom mapping from reactants and products
    df['reactants'] = df['reactants'].apply(remove_atom_mapping)
    df['products'] = df['products'].apply(remove_atom_mapping)

    # Drop rows with None values after mapping removal
    df = df.dropna(subset=['reactants', 'products'])

    # Save the processed DataFrame to a new CSV file
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data to {output_path}: {e}")
        
        

file = "data/uspto50k/raw_test.csv"

process_uspto_file(file, "data/uspto50k/tested.csv", max_samples=50000)