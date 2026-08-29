import time

import pandas as pd
from rdkit import Chem
from utils import map_smiles, strip_atom_mapping



def transpose(reaction, type):
    if type == 'mapped':
        return map_smiles(reaction)
    else:
        return strip_atom_mapping(reaction)


#[Br:1][CH2:2][CH2:3][O:4][C:5](=[O:6])[c:7]1[n:8][n:9](-[c:10]2[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]2[Cl:17])[c:18](-[c:19]2[cH:20][cH:21][c:22]([O:23][CH3:24])[cH:25][cH:26]2)[c:27]1[CH3:28]
# .[Br:1][CH2:28][c:27]1[c:7]([C:5]([O:4][CH2:3][CH3:2])=[O:6])[n:8][n:9](-[c:10]2[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]2[Cl:17])[c:18]1-[c:19]1[cH:20][cH:21][c:22]([O:23][CH3:24])[cH:25][cH:26]1
# .CCOC(=O)c1nn(-c2ccc(Cl)cc2Cl)c(-c2ccc(OC)cc2)c1CBr
reaction = '[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]'
output = transpose(reaction, 'unmapped')
print(output)


