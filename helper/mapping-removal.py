from rdkit import Chem

#mapped = "O[CH2:1][c:2]1[c:3]([CH3:4])[o:5][n:6][c:7]1-[c:8]1[cH:9][cH:10][c:11]([F:12])[cH:13][n:14]1.[CH3:15][O:16][C:17](=[O:18])[c:19]1[cH:20][c:21]([OH:22])[n:23][s:24]1"
mapped = "O[C@H:1]([CH3:2])[c:3]1[cH:4][c:5]([F:6])[cH:7][c:8]([F:9])[cH:10]1.[CH3:11][C@@H:12]1[CH2:13][NH:14][CH2:15][CH2:16][NH:17]1"

mol = Chem.MolFromSmiles(mapped)

for atom in mol.GetAtoms():
    atom.SetAtomMapNum(0)

unmapped = Chem.MolToSmiles(mol)

print(unmapped)