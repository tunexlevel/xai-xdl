import pandas as pd

from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors

from ord_schema import message_helpers
from ord_schema.proto import reaction_pb2


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "open-reaction-database/ord-data"
SPLIT = "train"

TARGET_COUNT = 20_000

OUTPUT_FILE = "undergrad_organic_smiles_20k.csv"

# Structural complexity limits
MAX_MOL_WEIGHT = 450
MAX_HEAVY_ATOMS = 30
MAX_RINGS = 3

# Reaction-level limits
MAX_REACTANTS = 4
MAX_PRODUCTS = 2


# ============================================================
# ORD PROTOBUF -> REACTION SMILES
# ============================================================

def extract_reaction_smiles(reaction_bytes: bytes) -> str | None:
    """
    Deserialize an ORD Reaction protobuf and obtain reaction SMILES.

    ORD records from the Hugging Face dataset contain the serialized
    Reaction protobuf in the `reaction` field.
    """

    try:
        reaction = reaction_pb2.Reaction()
        reaction.ParseFromString(reaction_bytes)

        # Use ORD's helper to obtain an existing reaction SMILES
        # or generate one from the reaction inputs/outcomes.
        reaction_smiles = message_helpers.get_reaction_smiles(
            reaction,
            generate_if_missing=True,
            allow_incomplete=False,
            validate=True,
            canonical=True,
        )

        return reaction_smiles

    except Exception:
        return None


# ============================================================
# MOLECULE VALIDATION
# ============================================================

def molecule_is_simple(mol) -> bool:
    """
    Check whether an individual molecule is sufficiently simple
    for our undergraduate organic chemistry dataset.
    """

    if mol is None:
        return False

    try:
        # Molecular weight
        if Descriptors.MolWt(mol) > MAX_MOL_WEIGHT:
            return False

        # Heavy atoms
        if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            return False

        # Rings
        if mol.GetRingInfo().NumRings() > MAX_RINGS:
            return False

        return True

    except Exception:
        return False


# ============================================================
# REACTION FILTER
# ============================================================

def is_undergrad_appropriate(rxn_smiles: str) -> bool:
    """
    Determine whether a reaction is sufficiently simple and valid
    for an undergraduate organic chemistry dataset.
    """

    if not isinstance(rxn_smiles, str):
        return False

    if not rxn_smiles:
        return False

    try:
        parts = rxn_smiles.split(">")

        # Reaction SMILES must have:
        # reactants > reagents > products
        if len(parts) != 3:
            return False

        reactants = parts[0]
        reagents = parts[1]
        products = parts[2]

        # Both sides must exist
        if not reactants or not products:
            return False

        # --------------------------------------------------------
        # Extract molecules
        # --------------------------------------------------------

        reactant_smiles = [
            smi for smi in reactants.split(".")
            if smi
        ]

        product_smiles = [
            smi for smi in products.split(".")
            if smi
        ]

        # Avoid reactions with too many components
        if len(reactant_smiles) > MAX_REACTANTS:
            return False

        if len(product_smiles) > MAX_PRODUCTS:
            return False

        if len(reactant_smiles) == 0:
            return False

        if len(product_smiles) == 0:
            return False

        # --------------------------------------------------------
        # Validate reactants
        # --------------------------------------------------------

        reactant_molecules = []

        for smi in reactant_smiles:

            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                return False

            if not molecule_is_simple(mol):
                return False

            reactant_molecules.append(mol)

        # --------------------------------------------------------
        # Validate products
        # --------------------------------------------------------

        product_molecules = []

        for smi in product_smiles:

            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                return False

            if not molecule_is_simple(mol):
                return False

            product_molecules.append(mol)

        # --------------------------------------------------------
        # Make sure reaction actually changes something
        # --------------------------------------------------------

        canonical_reactants = sorted(
            Chem.MolToSmiles(mol)
            for mol in reactant_molecules
        )

        canonical_products = sorted(
            Chem.MolToSmiles(mol)
            for mol in product_molecules
        )

        if canonical_reactants == canonical_products:
            return False

        return True

    except Exception:
        return False


# ============================================================
# MAIN DATA EXTRACTION
# ============================================================

def main():

    print("=" * 70)
    print("UNDERGRADUATE ORGANIC REACTION DATASET GENERATOR")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_NAME}")
    print(f"Target:  {TARGET_COUNT:,} reactions")
    print()

    print("Loading ORD dataset from Hugging Face...")

    dataset_stream = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
    )

    print("ORD dataset loaded successfully.")
    print()
    print("Beginning extraction...")
    print()

    extracted = []

    seen_reactions = set()

    scanned = 0
    invalid = 0
    duplicates = 0
    accepted = 0

    # ========================================================
    # STREAM DATA
    # ========================================================

    for entry in dataset_stream:

        scanned += 1

        reaction_id = entry.get("reaction_id", "")

        reaction_bytes = entry.get("reaction")

        if reaction_bytes is None:
            invalid += 1
            continue

        # ----------------------------------------------------
        # Convert protobuf -> reaction SMILES
        # ----------------------------------------------------

        reaction_smiles = extract_reaction_smiles(
            reaction_bytes
        )

        if reaction_smiles is None:
            invalid += 1
            continue

        # ----------------------------------------------------
        # Apply structural/reaction filters
        # ----------------------------------------------------

        if not is_undergrad_appropriate(reaction_smiles):
            invalid += 1
            continue

        # ----------------------------------------------------
        # Remove duplicate reactions
        # ----------------------------------------------------

        if reaction_smiles in seen_reactions:
            duplicates += 1
            continue

        seen_reactions.add(reaction_smiles)

        # ----------------------------------------------------
        # Store reaction
        # ----------------------------------------------------

        extracted.append(
            {
                "reaction_id": reaction_id,
                "reaction_smiles": reaction_smiles,
            }
        )

        accepted += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if accepted % 500 == 0:

            print(
                f"Accepted: {accepted:,} / {TARGET_COUNT:,} | "
                f"Scanned: {scanned:,} | "
                f"Rejected: {invalid:,} | "
                f"Duplicates: {duplicates:,}"
            )

        # ----------------------------------------------------
        # Stop when target reached
        # ----------------------------------------------------

        if accepted >= TARGET_COUNT:
            break

    # ========================================================
    # SAVE DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Scanned reactions : {scanned:,}")
    print(f"Accepted reactions: {accepted:,}")
    print(f"Rejected reactions: {invalid:,}")
    print(f"Duplicates removed: {duplicates:,}")
    print()

    if not extracted:

        print("ERROR: No reactions passed the filters.")
        return

    df = pd.DataFrame(extracted)

    # --------------------------------------------------------
    # Final duplicate safety check
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["reaction_smiles"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved dataset to:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("Dataset shape:")
    print(df.shape)

    print()
    print("First 5 reactions:")
    print(df.head().to_string(index=False))

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()