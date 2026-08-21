import pandas as pd


def load_uspto_file(path, max_samples=None):
    try:
        df = pd.read_csv(path)

        # Handle already-processed CSV format: reactants,products
        if {'reactants', 'products'}.issubset(df.columns):
            data = df[['reactants', 'products']].copy()
        else:
            # Handle raw USPTO CSV format: id,class,reactants>reagents>production
            if len(df.columns) >= 3:
                smiles_col = df.columns[2]
                split_smiles = df[smiles_col].astype(str).str.split(">", expand=True)
                if split_smiles.shape[1] >= 3:
                    data = pd.DataFrame({
                        'reactants': split_smiles[0].str.strip(),
                        'products': split_smiles[2].str.strip()
                    })
                else:
                    raise ValueError(f"Unexpected reaction column format in {path}")
            else:
                raise ValueError(f"Unsupported file format in {path}")

        data = data.dropna(subset=['reactants', 'products'])
        data['reactants'] = data['reactants'].astype(str).str.strip()
        data['products'] = data['products'].astype(str).str.strip()
        data = data[(data['reactants'] != '') & (data['products'] != '')]

        if max_samples is not None:
            data = data.head(max_samples)
        return data[['reactants', 'products']]
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return None
    

