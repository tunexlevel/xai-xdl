import pandas as pd

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
    

