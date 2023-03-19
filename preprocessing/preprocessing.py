"""
Preprocessing module
"""
import numpy as np
from preprocessing.chemistry import *

def filter_null_molecules(
        df,
        molecule_col = 'molecule'
    ):
    """
    Remove rows where 'molecule' column is NULL.
    """
    return df[~df[molecule_col].isna()]

def create_chemistry_features(df):
    """
    Compute all chemistry features. DataFrame requires Molecule Column 'molecule'
    """

    df = compute_radical_electrons(df)

    df = compute_exact_molecular_weight(df)

    df = compute_avg_molecular_weight_no_hydrogen(df)

    df = compute_valence_electrons(df)

    df = compute_avg_molecular_weight(df)

    df = compute_max_partial_charge(df)

    df = compute_min_partial_charge(df)

    df = compute_aliphatic_carbocycles(df)

    df = compute_aliphatic_heterocycles(df)

    df = compute_aromatic_carbocycles(df)

    df = compute_aromatic_heterocycles(df)

    df = compute_amide_bonds(df)

    df = compute_num_atoms(df)

    df = compute_bridgehead_atoms(df)

    df = compute_hbond_acceptors(df)

    df = compute_hbond_donors(df)

    df = compute_lipinski_hbond_acceptors(df)

    df = compute_lipinski_hbond_donors(df)

    df = compute_heavy_atoms(df)

    df = compute_hetero_atoms(df)

    df = compute_rotatable_bonds(df)

    df = compute_spiro_atoms(df)

    return df

def remove_null_columns(
        df, 
        null_threshold_pct = 0.5
    ):
    """
    Remove columns with specified threshold frequency of null values.
    """
    row_count = df.shape[0]
    
    # Compute % of Null Entries
    df_nulls = df.isna().sum()/row_count
    
    # Filter based on Threshold
    df_filtered = df_nulls[df_nulls < null_threshold_pct]
    
    # Filter Columns from DataFrame
    columns = df_filtered.index
    df = df[columns]
    
    return df

def count_null_features(
        df, 
        output_col = "null_feature_count"
    ):
    """
    Count number of null values across all features.
    """
    df[output_col] = df.isna().sum(axis = 1)

    return df

def compute_feature_sum(
        df, 
        output_col = "num_positive_classes"
    ):
    """
    Sum feature values for each row. Since each feature is a binary category, it counts the number of positive class labels.
    ** May have misinterpreted columns as all signifying different types of toxicity.
    """
    df[output_col] = df.sum(axis = 1, numeric_only = True)
    
    return df

def compute_binary_class_label(
        df, 
        input_col = "num_positive_classes", 
        output_col = "toxicity_target"
    ):

    """
    If any feature is positive (1), final output target variable is positive (1). 
    If they're all NaN or negative (0), then target variable is negative (0).
    """
    df[output_col] = np.select(
        [
            df[input_col] > 0
        ],
        [
            1
        ],
        default = 0
    )

    return df

def filter_non_feature_columns(
    df,
    molecule_col = 'molecule',
    target_col = 'toxicity_target'
    ):
    """
    Remove non-feature columns. Target column and molecule column must have already been computed.
    """

    print(type(df))

    df = df[[molecule_col, target_col]]

    return df

def create_target_variable(df):
    """
    Compute Target Variable
    """

    df = compute_feature_sum(df)

    df = compute_binary_class_label(df)

    return df

def compute_full_preprocessing(df):
    """
    Compute end-to-end preprocessing
    """
    # Create Molecule Column
    df = create_rdkit_molecule(df)

    # Filter Null Molecules
    df = filter_null_molecules(df)

    # Create Target Column
    df = create_target_variable(df)

    # Remove unnecessary Columns
    df = filter_non_feature_columns(df)

    # Compute Chemistry Features
    df = create_chemistry_features(df)

    return df


