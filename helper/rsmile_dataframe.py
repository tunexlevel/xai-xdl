import pandas as pd
from tqdm import tqdm
from helper.utils import get_root_aligned_pair



def prepare_rsmiles_dataframe(df_mapped: pd.DataFrame, augment_times: int = 1) -> pd.DataFrame:
    """
    Transforms an atom-mapped dataframe (with 'products' and 'reactants' columns)
    into Root-aligned SMILES pairs.
    """
    records = []
    
    for _, row in tqdm(df_mapped.iterrows(), total=len(df_mapped), desc="Aligning R-SMILES"):
        p_raw = row['products']
        r_raw = row['reactants']
        
        # 1x deterministic base alignment
        p_aligned, r_aligned = get_root_aligned_pair(p_raw, r_raw, augment_root=False)
        if p_aligned and r_aligned:
            records.append({'products': p_aligned, 'reactants': r_aligned})
            
        # Optional root-enumeration augmentation (as in the paper)
        for _ in range(augment_times - 1):
            p_aug, r_aug = get_root_aligned_pair(p_raw, r_raw, augment_root=True)
            if p_aug and r_aug:
                records.append({'products': p_aug, 'reactants': r_aug})
                
    return pd.DataFrame(records)



