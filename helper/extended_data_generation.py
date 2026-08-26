#!/usr/bin/env python3

"""
ChemXAI Master Reaction Dataset Generator

Preprocessing pipeline:

1. Load USPTO-MIT
2. Load USPTO-Condition
3. Parse reaction SMILES
4. Keep agents separate from explicit reagents
5. Remove atom mapping for canonical chemistry representation
6. Classify molecules as:
       parseable
       sanitizable
       usable
7. Calculate molecular complexity descriptors
8. Detect duplicates
9. Generate source-aware train/validation/test splits
10. Save diagnostics and statistics

Inputs:
    data/raw/USPTO_MIT.csv
    data/raw/USPTO_Condition.csv

Outputs:
    data/main/master_reactions.csv
    data/main/duplicates.csv
    data/main/invalid_reactions.csv
    data/main/dataset_statistics.json
    data/main/dataset_statistics.txt
    data/main/split_statistics.json
    data/main/train.csv
    data/main/validation.csv
    data/main/test.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MIT_PATH = Path("data/raw/USPTO_MIT.csv")
DEFAULT_CONDITION_PATH = Path("data/raw/USPTO_Condition.csv")
DEFAULT_OUTPUT_DIR = Path("data/output")

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VALIDATION_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10

DEFAULT_RANDOM_SEED = 42


# ============================================================
# RDKit LOGGING
# ============================================================

# Suppress noisy RDKit warnings during bulk preprocessing.
# Validation status is recorded explicitly in the dataset.
RDLogger.DisableLog("rdApp.*")


# ============================================================
# GENERAL UTILITIES
# ============================================================

def clean_text(value) -> str:
    """
    Convert a dataframe value into a clean string.

    NaN, None and empty strings become "".
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def remove_atom_mapping(smiles: str) -> str:
    """
    Remove atom-map numbers from a SMILES string.

    Example:

        [CH3:1][CH2:2]O

    becomes:

        [CH3][CH2]O
    """

    smiles = clean_text(smiles)

    if not smiles:
        return ""

    return re.sub(r":\d+", "", smiles)


def split_molecules(value: str) -> List[str]:
    """
    Split a dot-separated molecular component.
    """

    value = clean_text(value)

    if not value:
        return []

    return [
        molecule.strip()
        for molecule in value.split(".")
        if molecule.strip()
    ]


def count_molecules(value: str) -> int:
    """
    Count molecules in a dot-separated SMILES string.
    """

    return len(split_molecules(value))


def combine_fields(values: List[str]) -> str:
    """
    Combine multiple SMILES fields.

    Empty fields are ignored.

    Example:

        ["CO", "", "[Cl-]"]

    becomes:

        "CO.[Cl-]"
    """

    components = []

    for value in values:

        value = clean_text(value)

        if not value:
            continue

        components.extend(split_molecules(value))

    return ".".join(components)


# ============================================================
# REACTION PARSING
# ============================================================

def parse_reaction_smiles(
    reaction: str,
) -> Tuple[str, str, str]:
    """
    Parse reaction SMILES.

    Supports:

        reactants>agents>products

    and:

        reactants>>products

    Returns:

        reactants
        agents
        products
    """

    reaction = clean_text(reaction)

    if not reaction:
        return "", "", ""

    parts = reaction.split(">")

    if len(parts) == 3:

        reactants = parts[0].strip()
        agents = parts[1].strip()
        products = parts[2].strip()

        return reactants, agents, products

    if len(parts) == 2:

        reactants = parts[0].strip()
        products = parts[1].strip()

        return reactants, "", products

    return "", "", ""


# ============================================================
# MOLECULE VALIDATION
# ============================================================

def inspect_molecule(smiles: str) -> Dict:
    """
    Inspect a single molecule.

    Three distinct stages are recorded:

        parseable
        sanitizable
        usable

    Definitions:

        parseable:
            RDKit can construct a molecular graph.

        sanitizable:
            RDKit can successfully sanitize the molecule.

        usable:
            Molecule can be sanitized and the required
            molecular descriptors can be calculated safely.
    """

    smiles = clean_text(smiles)

    result = {
        "parseable": False,
        "sanitizable": False,
        "usable": False,
        "error": "",
    }

    if not smiles:

        result["error"] = "empty_smiles"

        return result

    # --------------------------------------------------------
    # Stage 1: Parse
    # --------------------------------------------------------

    try:

        mol = Chem.MolFromSmiles(
            smiles,
            sanitize=False,
        )

    except Exception as exc:

        result["error"] = (
            f"parse_exception: {str(exc)}"
        )

        return result

    if mol is None:

        result["error"] = "parse_failed"

        return result

    result["parseable"] = True

    # --------------------------------------------------------
    # Stage 2: Sanitize
    # --------------------------------------------------------

    try:

        Chem.SanitizeMol(mol)

    except Exception as exc:

        result["error"] = (
            f"sanitize_failed: {str(exc)}"
        )

        return result

    result["sanitizable"] = True

    # --------------------------------------------------------
    # Stage 3: Descriptor calculation
    # --------------------------------------------------------

    try:

        _ = Descriptors.MolWt(mol)
        _ = Descriptors.HeavyAtomCount(mol)
        _ = rdMolDescriptors.CalcTPSA(mol)
        _ = Crippen.MolLogP(mol)
        _ = Lipinski.NumHDonors(mol)
        _ = Lipinski.NumHAcceptors(mol)
        _ = Lipinski.NumRotatableBonds(mol)
        _ = rdMolDescriptors.CalcFractionCSP3(mol)

    except Exception as exc:

        result["error"] = (
            f"descriptor_failed: {str(exc)}"
        )

        return result

    result["usable"] = True

    return result


def inspect_component(component: str) -> Dict:
    """
    Inspect every molecule in a component.

    Returns aggregate validation status.
    """

    component = clean_text(component)

    molecules = split_molecules(component)

    if not molecules:

        return {
            "parseable": True,
            "sanitizable": True,
            "usable": True,
            "error": "",
            "num_molecules": 0,
        }

    statuses = []

    for molecule in molecules:

        molecule_unmapped = remove_atom_mapping(
            molecule
        )

        statuses.append(
            inspect_molecule(
                molecule_unmapped
            )
        )

    parseable = all(
        item["parseable"]
        for item in statuses
    )

    sanitizable = all(
        item["sanitizable"]
        for item in statuses
    )

    usable = all(
        item["usable"]
        for item in statuses
    )

    errors = [
        item["error"]
        for item in statuses
        if item["error"]
    ]

    return {
        "parseable": parseable,
        "sanitizable": sanitizable,
        "usable": usable,
        "error": " | ".join(errors),
        "num_molecules": len(molecules),
    }


# ============================================================
# CANONICALIZATION
# ============================================================

def canonicalize_molecule(
    smiles: str,
) -> Tuple[str, Dict]:
    """
    Canonicalize one unmapped molecule.

    Returns:

        canonical_smiles
        validation_status
    """

    smiles = clean_text(smiles)

    status = inspect_molecule(smiles)

    if not status["usable"]:

        return smiles, status

    try:

        mol = Chem.MolFromSmiles(
            smiles,
            sanitize=True,
        )

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        return canonical, status

    except Exception as exc:

        status["usable"] = False
        status["error"] = (
            f"canonicalization_failed: {str(exc)}"
        )

        return smiles, status


def canonicalize_component(
    component: str,
) -> Tuple[str, Dict]:
    """
    Canonicalize a dot-separated reaction component.

    Atom mapping is removed before canonicalization.
    """

    component = clean_text(component)

    molecules = split_molecules(component)

    if not molecules:

        return "", {
            "parseable": True,
            "sanitizable": True,
            "usable": True,
            "error": "",
            "num_molecules": 0,
        }

    canonical_molecules = []
    statuses = []

    for molecule in molecules:

        unmapped = remove_atom_mapping(
            molecule
        )

        canonical, status = canonicalize_molecule(
            unmapped
        )

        canonical_molecules.append(
            canonical
        )

        statuses.append(status)

    canonical_molecules = sorted(
        canonical_molecules
    )

    aggregate = {
        "parseable": all(
            item["parseable"]
            for item in statuses
        ),
        "sanitizable": all(
            item["sanitizable"]
            for item in statuses
        ),
        "usable": all(
            item["usable"]
            for item in statuses
        ),
        "error": " | ".join(
            item["error"]
            for item in statuses
            if item["error"]
        ),
        "num_molecules": len(
            canonical_molecules
        ),
    }

    return ".".join(
        canonical_molecules
    ), aggregate


# ============================================================
# COMPLEXITY DESCRIPTORS
# ============================================================

DESCRIPTOR_COLUMNS = [
    "molwt",
    "heavy_atoms",
    "rings",
    "aromatic_rings",
    "rotatable_bonds",
    "hbd",
    "hba",
    "tpsa",
    "fraction_csp3",
    "logp",
]


def empty_descriptors() -> Dict:
    """
    Return empty descriptor values.
    """

    return {
        name: 0.0
        for name in DESCRIPTOR_COLUMNS
    }


def molecule_descriptors(smiles: str) -> Dict:
    """
    Calculate descriptors for one molecule.

    Returns zeros when the molecule cannot be used.
    """

    smiles = clean_text(smiles)

    if not smiles:

        return empty_descriptors()

    try:

        mol = Chem.MolFromSmiles(
            remove_atom_mapping(smiles),
            sanitize=True,
        )

        if mol is None:

            return empty_descriptors()

        return {
            "molwt": float(
                Descriptors.MolWt(mol)
            ),

            "heavy_atoms": float(
                Descriptors.HeavyAtomCount(mol)
            ),

            "rings": float(
                rdMolDescriptors.CalcNumRings(mol)
            ),

            "aromatic_rings": float(
                rdMolDescriptors.CalcNumAromaticRings(
                    mol
                )
            ),

            "rotatable_bonds": float(
                Lipinski.NumRotatableBonds(mol)
            ),

            "hbd": float(
                Lipinski.NumHDonors(mol)
            ),

            "hba": float(
                Lipinski.NumHAcceptors(mol)
            ),

            "tpsa": float(
                rdMolDescriptors.CalcTPSA(mol)
            ),

            "fraction_csp3": float(
                rdMolDescriptors.CalcFractionCSP3(mol)
            ),

            "logp": float(
                Crippen.MolLogP(mol)
            ),
        }

    except Exception:

        return empty_descriptors()


def aggregate_descriptors(
    component: str,
    prefix: str,
) -> Dict:
    """
    Calculate aggregate molecular descriptors.

    Aggregation:

        MW / heavy atoms / rings / etc.
            = sum across molecules

        fraction_csp3 / logP
            = mean across molecules
    """

    molecules = split_molecules(
        component
    )

    if not molecules:

        result = {}

        for column in DESCRIPTOR_COLUMNS:

            result[
                f"{prefix}_{column}"
            ] = 0.0

        return result

    descriptors = [
        molecule_descriptors(
            molecule
        )
        for molecule in molecules
    ]

    result = {}

    for column in DESCRIPTOR_COLUMNS:

        values = [
            item[column]
            for item in descriptors
        ]

        if column in {
            "fraction_csp3",
            "logp",
        }:

            value = sum(values) / len(values)

        else:

            value = sum(values)

        result[
            f"{prefix}_{column}"
        ] = round(value, 6)

    return result


def calculate_reaction_descriptors(
    reactants: str,
    products: str,
) -> Dict:
    """
    Calculate complexity descriptors for
    reactants and products.
    """

    result = {}

    result.update(
        aggregate_descriptors(
            reactants,
            "reactant",
        )
    )

    result.update(
        aggregate_descriptors(
            products,
            "product",
        )
    )

    return result


# ============================================================
# DUPLICATE KEY
# ============================================================

def create_duplicate_key(
    reactants: str,
    products: str,
    agents: str = "",
    reagents: str = "",
    catalysts: str = "",
    solvents: str = "",
) -> str:
    """
    Create a deterministic duplicate key.

    Chemistry is represented canonically.

    Conditions are retained where available.

    Agents are included separately from explicit
    reagents/catalysts/solvents.
    """

    fields = [
        clean_text(reactants),
        clean_text(products),
        clean_text(agents),
        clean_text(reagents),
        clean_text(catalysts),
        clean_text(solvents),
    ]

    canonical_string = "||".join(
        fields
    )

    return hashlib.sha256(
        canonical_string.encode("utf-8")
    ).hexdigest()


# ============================================================
# COMMON RECORD BUILDER
# ============================================================

def build_record(
    reaction_id: str,
    source: str,
    dataset: str,
    raw_reaction: str,
    mapped_reactants: str,
    mapped_agents: str,
    mapped_reagents: str,
    mapped_catalysts: str,
    mapped_solvents: str,
    mapped_products: str,
    reactants: str,
    agents: str,
    reagents: str,
    catalysts: str,
    solvents: str,
    products: str,
) -> Dict:
    """
    Build a standardized master record.
    """

    reactant_status = inspect_component(
        reactants
    )

    product_status = inspect_component(
        products
    )

    agent_status = inspect_component(
        agents
    )

    reagent_status = inspect_component(
        reagents
    )

    catalyst_status = inspect_component(
        catalysts
    )

    solvent_status = inspect_component(
        solvents
    )

    descriptors = calculate_reaction_descriptors(
        reactants,
        products,
    )

    duplicate_key = create_duplicate_key(
        reactants=reactants,
        products=products,
        agents=agents,
        reagents=reagents,
        catalysts=catalysts,
        solvents=solvents,
    )

    record = {
        "reaction_id": reaction_id,

        "source": source,

        "dataset": dataset,

        "raw_reaction": raw_reaction,

        # ----------------------------------------------------
        # Mapped representation
        # ----------------------------------------------------

        "mapped_reactants": mapped_reactants,
        "mapped_agents": mapped_agents,
        "mapped_reagents": mapped_reagents,
        "mapped_catalysts": mapped_catalysts,
        "mapped_solvents": mapped_solvents,
        "mapped_products": mapped_products,

        # ----------------------------------------------------
        # Unmapped canonical representation
        # ----------------------------------------------------

        "reactants": reactants,
        "agents": agents,
        "reagents": reagents,
        "catalysts": catalysts,
        "solvents": solvents,
        "products": products,

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        "has_agents": bool(agents),
        "has_reagents": bool(reagents),
        "has_catalysts": bool(catalysts),
        "has_solvents": bool(solvents),

        # ----------------------------------------------------
        # Molecule counts
        # ----------------------------------------------------

        "num_reactant_molecules":
            count_molecules(reactants),

        "num_product_molecules":
            count_molecules(products),

        "num_agent_molecules":
            count_molecules(agents),

        "num_reagent_molecules":
            count_molecules(reagents),

        "num_catalyst_molecules":
            count_molecules(catalysts),

        "num_solvent_molecules":
            count_molecules(solvents),

        # ----------------------------------------------------
        # Reactant validation
        # ----------------------------------------------------

        "reactants_parseable":
            reactant_status["parseable"],

        "reactants_sanitizable":
            reactant_status["sanitizable"],

        "reactants_usable":
            reactant_status["usable"],

        "reactants_validation_error":
            reactant_status["error"],

        # ----------------------------------------------------
        # Product validation
        # ----------------------------------------------------

        "products_parseable":
            product_status["parseable"],

        "products_sanitizable":
            product_status["sanitizable"],

        "products_usable":
            product_status["usable"],

        "products_validation_error":
            product_status["error"],

        # ----------------------------------------------------
        # Agent validation
        # ----------------------------------------------------

        "agents_parseable":
            agent_status["parseable"],

        "agents_sanitizable":
            agent_status["sanitizable"],

        "agents_usable":
            agent_status["usable"],

        "agents_validation_error":
            agent_status["error"],

        # ----------------------------------------------------
        # Reagent validation
        # ----------------------------------------------------

        "reagents_parseable":
            reagent_status["parseable"],

        "reagents_sanitizable":
            reagent_status["sanitizable"],

        "reagents_usable":
            reagent_status["usable"],

        "reagents_validation_error":
            reagent_status["error"],

        # ----------------------------------------------------
        # Catalyst validation
        # ----------------------------------------------------

        "catalysts_parseable":
            catalyst_status["parseable"],

        "catalysts_sanitizable":
            catalyst_status["sanitizable"],

        "catalysts_usable":
            catalyst_status["usable"],

        "catalysts_validation_error":
            catalyst_status["error"],

        # ----------------------------------------------------
        # Solvent validation
        # ----------------------------------------------------

        "solvents_parseable":
            solvent_status["parseable"],

        "solvents_sanitizable":
            solvent_status["sanitizable"],

        "solvents_usable":
            solvent_status["usable"],

        "solvents_validation_error":
            solvent_status["error"],

        # ----------------------------------------------------
        # Reaction-level validation
        # ----------------------------------------------------

        "parseable_reaction": (
            reactant_status["parseable"]
            and product_status["parseable"]
        ),

        "sanitizable_reaction": (
            reactant_status["sanitizable"]
            and product_status["sanitizable"]
        ),

        "usable_reaction": (
            reactant_status["usable"]
            and product_status["usable"]
        ),

        # ----------------------------------------------------
        # Backward-compatible aliases
        # ----------------------------------------------------

        "valid_reactants":
            reactant_status["usable"],

        "valid_products":
            product_status["usable"],

        "valid_agents":
            agent_status["usable"],

        "valid_reagents":
            reagent_status["usable"],

        "valid_catalysts":
            catalyst_status["usable"],

        "valid_solvents":
            solvent_status["usable"],

        "valid_reaction": (
            reactant_status["usable"]
            and product_status["usable"]
        ),

        # ----------------------------------------------------
        # Duplicate key
        # ----------------------------------------------------

        "duplicate_key": duplicate_key,
    }

    # Add complexity descriptors.
    record.update(descriptors)

    return record


# ============================================================
# USPTO-MIT PROCESSING
# ============================================================

def process_uspto_mit(
    path: Path,
) -> List[Dict]:
    """
    Process USPTO-MIT.

    Important:

    The middle field of:

        reactants>agents>products

    is stored as AGENTS.

    It is NOT automatically classified as a reagent.
    """

    print()
    print("=" * 70)
    print("Processing USPTO-MIT")
    print("=" * 70)

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    if "reactions" not in df.columns:

        raise ValueError(
            "USPTO-MIT must contain a "
            f"'reactions' column. Found: "
            f"{list(df.columns)}"
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    records = []

    for index, row in df.iterrows():

        raw_reaction = clean_text(
            row["reactions"]
        )

        if not raw_reaction:
            continue

        (
            reactants_raw,
            agents_raw,
            products_raw,
        ) = parse_reaction_smiles(
            raw_reaction
        )

        if not reactants_raw or not products_raw:
            continue

        mapped_reactants = clean_text(
            reactants_raw
        )

        mapped_agents = clean_text(
            agents_raw
        )

        mapped_products = clean_text(
            products_raw
        )

        reactants, _ = canonicalize_component(
            reactants_raw
        )

        agents, _ = canonicalize_component(
            agents_raw
        )

        products, _ = canonicalize_component(
            products_raw
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We do NOT infer that the MIT middle field is
        # reagent/catalyst/solvent.
        # ----------------------------------------------------

        reagents = ""
        catalysts = ""
        solvents = ""

        record = build_record(
            reaction_id=f"USPTO_MIT_{index}",

            source="USPTO_MIT",

            dataset="USPTO_MIT",

            raw_reaction=raw_reaction,

            mapped_reactants=mapped_reactants,

            mapped_agents=mapped_agents,

            mapped_reagents="",

            mapped_catalysts="",

            mapped_solvents="",

            mapped_products=mapped_products,

            reactants=reactants,

            agents=agents,

            reagents=reagents,

            catalysts=catalysts,

            solvents=solvents,

            products=products,
        )

        records.append(record)

    print(
        "Reaction records parsed: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# USPTO-CONDITION PROCESSING
# ============================================================

def process_uspto_condition(
    path: Path,
) -> List[Dict]:
    """
    Process USPTO_Condition.

    Expected columns:

        source
        canonical_rxn
        catalyst1
        solvent1
        solvent2
        reagent1
        reagent2
        dataset
    """

    print()
    print("=" * 70)
    print("Processing USPTO_Condition")
    print("=" * 70)

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = [
        "source",
        "canonical_rxn",
        "catalyst1",
        "solvent1",
        "solvent2",
        "reagent1",
        "reagent2",
        "dataset",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "USPTO_Condition is missing columns: "
            f"{missing}"
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    records = []

    for index, row in df.iterrows():

        source = clean_text(
            row["source"]
        )

        raw_reaction = clean_text(
            row["canonical_rxn"]
        )

        if not raw_reaction:
            continue

        (
            reactants_raw,
            agents_raw,
            products_raw,
        ) = parse_reaction_smiles(
            raw_reaction
        )

        if not reactants_raw or not products_raw:
            continue

        catalyst_raw = clean_text(
            row["catalyst1"]
        )

        solvent_raw = combine_fields(
            [
                row["solvent1"],
                row["solvent2"],
            ]
        )

        reagent_raw = combine_fields(
            [
                row["reagent1"],
                row["reagent2"],
            ]
        )

        # The canonical reaction itself may contain
        # an agents field. Keep it separate.
        mapped_reactants = reactants_raw
        mapped_agents = agents_raw
        mapped_products = products_raw

        mapped_reagents = reagent_raw
        mapped_catalysts = catalyst_raw
        mapped_solvents = solvent_raw

        reactants, _ = canonicalize_component(
            reactants_raw
        )

        agents, _ = canonicalize_component(
            agents_raw
        )

        products, _ = canonicalize_component(
            products_raw
        )

        reagents, _ = canonicalize_component(
            reagent_raw
        )

        catalysts, _ = canonicalize_component(
            catalyst_raw
        )

        solvents, _ = canonicalize_component(
            solvent_raw
        )

        dataset_name = clean_text(
            row["dataset"]
        )

        dataset = (
            f"USPTO_Condition_{dataset_name}"
            if dataset_name
            else "USPTO_Condition"
        )

        record = build_record(
            reaction_id=(
                f"USPTO_CONDITION_{index}"
            ),

            source=source,

            dataset=dataset,

            raw_reaction=raw_reaction,

            mapped_reactants=mapped_reactants,

            mapped_agents=mapped_agents,

            mapped_reagents=mapped_reagents,

            mapped_catalysts=mapped_catalysts,

            mapped_solvents=mapped_solvents,

            mapped_products=mapped_products,

            reactants=reactants,

            agents=agents,

            reagents=reagents,

            catalysts=catalysts,

            solvents=solvents,

            products=products,
        )

        records.append(record)

    print(
        "Reaction records parsed: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# DATASET COMBINATION
# ============================================================

def combine_datasets(
    mit_records: List[Dict],
    condition_records: List[Dict],
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("Combining datasets")
    print("=" * 70)

    records = (
        mit_records
        + condition_records
    )

    df = pd.DataFrame(records)

    print(
        f"Combined rows: {len(df):,}"
    )

    return df


# ============================================================
# DUPLICATE HANDLING
# ============================================================

def detect_duplicates(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect duplicate reactions using duplicate_key.

    The first occurrence is retained.
    """

    print()
    print("=" * 70)
    print("Duplicate detection")
    print("=" * 70)

    duplicate_mask = df.duplicated(
        subset=["duplicate_key"],
        keep="first",
    )

    duplicates = df[
        duplicate_mask
    ].copy()

    cleaned = df[
        ~duplicate_mask
    ].copy()

    print(
        f"Duplicates detected: "
        f"{len(duplicates):,}"
    )

    print(
        f"Remaining reactions: "
        f"{len(cleaned):,}"
    )

    return cleaned, duplicates


# ============================================================
# SOURCE HANDLING
# ============================================================

def normalize_source(
    source: str,
    dataset: str,
) -> str:
    """
    Ensure every reaction has a source.

    Explicit source is preferred.

    Dataset is used as a fallback.
    """

    source = clean_text(source)

    if source:
        return source

    dataset = clean_text(dataset)

    if dataset:
        return dataset

    return "UNKNOWN"


# ============================================================
# SOURCE-AWARE SPLIT
# ============================================================

def deterministic_bucket(
    value: str,
) -> float:
    """
    Convert a string into a deterministic number
    in [0, 1).
    """

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    integer = int(
        digest[:16],
        16,
    )

    return integer / float(
        16 ** 16
    )


def assign_source_aware_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> pd.DataFrame:
    """
    Assign train/validation/test splits.

    Split proportions are applied independently within
    each source.

    This prevents a large source from dominating the
    entire validation/test set while ensuring that
    different sources are represented in every split.

    The assignment is deterministic.
    """

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-8:

        raise ValueError(
            "Train/validation/test ratios must "
            f"sum to 1.0. Got {total_ratio}"
        )

    df = df.copy()

    df["split_source"] = [
        normalize_source(
            source,
            dataset,
        )
        for source, dataset
        in zip(
            df["source"],
            df["dataset"],
        )
    ]

    split_values = []

    for _, row in df.iterrows():

        reaction_id = clean_text(
            row["reaction_id"]
        )

        source = clean_text(
            row["split_source"]
        )

        key = (
            f"{source}||{reaction_id}"
        )

        bucket = deterministic_bucket(
            key
        )

        if bucket < train_ratio:

            split = "train"

        elif bucket < (
            train_ratio
            + validation_ratio
        ):

            split = "validation"

        else:

            split = "test"

        split_values.append(split)

    df["split"] = split_values

    return df


# ============================================================
# STATISTICS
# ============================================================

def percentage(
    value: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        100.0 * value / total,
        3,
    )


def calculate_statistics(
    df: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> Dict:

    total = len(df)

    stats = {
        "total_reactions": total,

        "duplicates_removed":
            len(duplicates),

        "datasets":
            df["dataset"]
            .value_counts()
            .to_dict(),

        "sources":
            df["split_source"]
            .value_counts()
            .to_dict(),

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        "parseable_reactions":
            int(
                df["parseable_reaction"].sum()
            ),

        "sanitizable_reactions":
            int(
                df["sanitizable_reaction"].sum()
            ),

        "usable_reactions":
            int(
                df["usable_reaction"].sum()
            ),

        "parseable_reaction_percentage":
            percentage(
                int(
                    df[
                        "parseable_reaction"
                    ].sum()
                ),
                total,
            ),

        "sanitizable_reaction_percentage":
            percentage(
                int(
                    df[
                        "sanitizable_reaction"
                    ].sum()
                ),
                total,
            ),

        "usable_reaction_percentage":
            percentage(
                int(
                    df[
                        "usable_reaction"
                    ].sum()
                ),
                total,
            ),

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        "agent_coverage":
            int(df["has_agents"].sum()),

        "agent_coverage_percentage":
            percentage(
                int(
                    df["has_agents"].sum()
                ),
                total,
            ),

        "reagent_coverage":
            int(df["has_reagents"].sum()),

        "reagent_coverage_percentage":
            percentage(
                int(
                    df["has_reagents"].sum()
                ),
                total,
            ),

        "catalyst_coverage":
            int(df["has_catalysts"].sum()),

        "catalyst_coverage_percentage":
            percentage(
                int(
                    df["has_catalysts"].sum()
                ),
                total,
            ),

        "solvent_coverage":
            int(df["has_solvents"].sum()),

        "solvent_coverage_percentage":
            percentage(
                int(
                    df["has_solvents"].sum()
                ),
                total,
            ),

        # ----------------------------------------------------
        # Molecule counts
        # ----------------------------------------------------

        "average_reactants":
            round(
                df[
                    "num_reactant_molecules"
                ].mean(),
                3,
            ),

        "average_products":
            round(
                df[
                    "num_product_molecules"
                ].mean(),
                3,
            ),

        "average_agents":
            round(
                df[
                    "num_agent_molecules"
                ].mean(),
                3,
            ),

        "average_reagents":
            round(
                df[
                    "num_reagent_molecules"
                ].mean(),
                3,
            ),

        "average_catalysts":
            round(
                df[
                    "num_catalyst_molecules"
                ].mean(),
                3,
            ),

        "average_solvents":
            round(
                df[
                    "num_solvent_molecules"
                ].mean(),
                3,
            ),
    }

    # --------------------------------------------------------
    # Split statistics
    # --------------------------------------------------------

    if "split" in df.columns:

        stats["splits"] = (
            df["split"]
            .value_counts()
            .to_dict()
        )

        stats["split_percentages"] = {
            split: percentage(
                int(
                    (
                        df["split"]
                        == split
                    ).sum()
                ),
                total,
            )
            for split in [
                "train",
                "validation",
                "test",
            ]
        }

    return stats


def calculate_split_statistics(
    df: pd.DataFrame,
) -> Dict:
    """
    Detailed source-by-split statistics.
    """

    table = (
        df.groupby(
            [
                "split_source",
                "split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    result = {}

    for source, row in table.iterrows():

        result[
            str(source)
        ] = {
            column: int(
                row.get(
                    column,
                    0,
                )
            )
            for column in [
                "train",
                "validation",
                "test",
            ]
        }

    return result


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    stats: Dict,
    output_dir: Path,
):

    json_path = (
        output_dir
        / "dataset_statistics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    txt_path = (
        output_dir
        / "dataset_statistics.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ChemXAI Dataset Statistics\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        for key, value in stats.items():

            f.write(
                f"{key}: {value}\n"
            )

    print(
        f"Statistics written to: "
        f"{json_path}"
    )

    print(
        f"Statistics written to: "
        f"{txt_path}"
    )


def save_split_statistics(
    stats: Dict,
    output_dir: Path,
):

    path = (
        output_dir
        / "split_statistics.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    print(
        f"Split statistics written to: "
        f"{path}"
    )


# ============================================================
# INVALID RECORD REPORT
# ============================================================

def save_invalid_records(
    df: pd.DataFrame,
    output_dir: Path,
):

    invalid = df[
        ~df["usable_reaction"]
    ].copy()

    path = (
        output_dir
        / "invalid_reactions.csv"
    )

    invalid.to_csv(
        path,
        index=False,
    )

    print(
        f"Invalid/unsupported reactions "
        f"written to: {path}"
    )

    # --------------------------------------------------------
    # Validation failure summary
    # --------------------------------------------------------

    failure_summary = {
        "total_invalid": len(invalid),

        "reactants_not_parseable":
            int(
                (~invalid[
                    "reactants_parseable"
                ]).sum()
            ),

        "reactants_not_sanitizable":
            int(
                (
                    invalid[
                        "reactants_parseable"
                    ]
                    & ~invalid[
                        "reactants_sanitizable"
                    ]
                ).sum()
            ),

        "products_not_parseable":
            int(
                (~invalid[
                    "products_parseable"
                ]).sum()
            ),

        "products_not_sanitizable":
            int(
                (
                    invalid[
                        "products_parseable"
                    ]
                    & ~invalid[
                        "products_sanitizable"
                    ]
                ).sum()
            ),
    }

    summary_path = (
        output_dir
        / "validation_failure_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            failure_summary,
            f,
            indent=2,
        )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    df: pd.DataFrame,
    output_dir: Path,
):

    train = df[
        df["split"] == "train"
    ].copy()

    validation = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    train_path = (
        output_dir
        / "train.csv"
    )

    validation_path = (
        output_dir
        / "validation.csv"
    )

    test_path = (
        output_dir
        / "test.csv"
    )

    train.to_csv(
        train_path,
        index=False,
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
    )

    print()
    print("Splits written:")
    print(
        f"  Train:      {train_path} "
        f"({len(train):,})"
    )

    print(
        f"  Validation: {validation_path} "
        f"({len(validation):,})"
    )

    print(
        f"  Test:       {test_path} "
        f"({len(test):,})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate ChemXAI master reaction "
            "dataset."
        )
    )

    parser.add_argument(
        "--mit",
        type=Path,
        default=DEFAULT_MIT_PATH,
        help="Path to USPTO-MIT CSV",
    )

    parser.add_argument(
        "--condition",
        type=Path,
        default=DEFAULT_CONDITION_PATH,
        help="Path to USPTO-Condition CSV",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Training split ratio",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Validation split ratio",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test split ratio",
    )

    args = parser.parse_args()

    output_dir = args.output

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "CHEMXAI MASTER DATASET GENERATOR"
    )
    print("=" * 70)

    print(
        f"USPTO-MIT:       {args.mit}"
    )

    print(
        f"USPTO-Condition: {args.condition}"
    )

    print(
        f"Output:          {output_dir}"
    )

    print()
    print(
        "Split ratios:"
    )

    print(
        f"  Train:      {args.train_ratio}"
    )

    print(
        f"  Validation: {args.validation_ratio}"
    )

    print(
        f"  Test:       {args.test_ratio}"
    )

    # --------------------------------------------------------
    # Validate split ratios
    # --------------------------------------------------------

    ratio_sum = (
        args.train_ratio
        + args.validation_ratio
        + args.test_ratio
    )

    if abs(ratio_sum - 1.0) > 1e-8:

        raise ValueError(
            "Train/validation/test ratios "
            f"must sum to 1.0. Got {ratio_sum}"
        )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not args.mit.exists():

        raise FileNotFoundError(
            f"USPTO-MIT not found: "
            f"{args.mit}"
        )

    if not args.condition.exists():

        raise FileNotFoundError(
            "USPTO-Condition not found: "
            f"{args.condition}"
        )

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    mit_records = process_uspto_mit(
        args.mit
    )

    condition_records = (
        process_uspto_condition(
            args.condition
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    df = combine_datasets(
        mit_records,
        condition_records,
    )

    # --------------------------------------------------------
    # Normalize source
    # --------------------------------------------------------

    df["source"] = [
        normalize_source(
            source,
            dataset,
        )
        for source, dataset
        in zip(
            df["source"],
            df["dataset"],
        )
    ]

    # --------------------------------------------------------
    # Detect duplicates
    # --------------------------------------------------------

    df, duplicates = detect_duplicates(
        df
    )

    # --------------------------------------------------------
    # Save duplicates
    # --------------------------------------------------------

    duplicates_path = (
        output_dir
        / "duplicates.csv"
    )

    duplicates.to_csv(
        duplicates_path,
        index=False,
    )

    print(
        f"Duplicates written to: "
        f"{duplicates_path}"
    )

    # --------------------------------------------------------
    # Source-aware split
    #
    # IMPORTANT:
    #
    # Splitting happens AFTER duplicate removal.
    # --------------------------------------------------------

    df = assign_source_aware_split(
        df,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = calculate_statistics(
        df,
        duplicates,
    )

    save_statistics(
        stats,
        output_dir,
    )

    split_stats = (
        calculate_split_statistics(
            df
        )
    )

    save_split_statistics(
        split_stats,
        output_dir,
    )

    # --------------------------------------------------------
    # Invalid records
    # --------------------------------------------------------

    save_invalid_records(
        df,
        output_dir,
    )

    # --------------------------------------------------------
    # Save master dataset
    # --------------------------------------------------------

    master_path = (
        output_dir
        / "master_reactions.csv"
    )

    df.to_csv(
        master_path,
        index=False,
    )

    print(
        f"Master dataset written to: "
        f"{master_path}"
    )

    # --------------------------------------------------------
    # Save train/validation/test
    # --------------------------------------------------------

    save_splits(
        df,
        output_dir,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DATASET GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Master dataset: "
        f"{master_path}"
    )

    print(
        f"Total reactions: "
        f"{len(df):,}"
    )

    print(
        f"Duplicates removed: "
        f"{len(duplicates):,}"
    )

    print(
        f"Parseable reactions: "
        f"{stats['parseable_reactions']:,} "
        f"({stats['parseable_reaction_percentage']}%)"
    )

    print(
        f"Sanitizable reactions: "
        f"{stats['sanitizable_reactions']:,} "
        f"({stats['sanitizable_reaction_percentage']}%)"
    )

    print(
        f"Usable reactions: "
        f"{stats['usable_reactions']:,} "
        f"({stats['usable_reaction_percentage']}%)"
    )

    print(
        f"Agent coverage: "
        f"{stats['agent_coverage_percentage']}%"
    )

    print(
        f"Reagent coverage: "
        f"{stats['reagent_coverage_percentage']}%"
    )

    print(
        f"Catalyst coverage: "
        f"{stats['catalyst_coverage_percentage']}%"
    )

    print(
        f"Solvent coverage: "
        f"{stats['solvent_coverage_percentage']}%"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Inspect dataset_statistics.txt, "
        "validation_failure_summary.json, "
        "invalid_reactions.csv and "
        "split_statistics.json before training."
    )


if __name__ == "__main__":
    main()