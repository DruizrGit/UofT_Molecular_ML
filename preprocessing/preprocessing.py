"""
Preprocessing module
"""
import numpy as np

def remove_null_columns(df, null_threshold_pct = 0.5):
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

def count_null_features(df, output_col = "null_feature_count"):
    """
    Count number of null values across all features.
    """
    df[output_col] = df.isna().sum(axis = 1)

    return df

def compute_feature_sum(df, output_col = "num_positive_classes"):
    """
    Sum feature values for each row. Since each feature is a binary category, it counts the number of positive class labels.
    ** May have misinterpreted columns as all signifying different types of toxicity.
    """
    df[output_col] = df.sum(axis = 1, numeric_only = True)

    return df

def construct_binary_class_label(df, input_col = "num_positive_classes", output_col = "toxicity_target"):
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
