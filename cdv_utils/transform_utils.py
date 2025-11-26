"""
Data Transformation Utilities for RealCause

This module provides functions to transform the processed sepsis dataset
into the format required by the RealCause framework.
"""

import pandas as pd
import numpy as np


def prepare_realcause_dataset(df_sepsis_extended, outcome_column='cycle_time', 
                             decision_column='admission_decision'):
    """
    Transform the sepsis dataset into RealCause format.
    
    Parameters:
    -----------
    df_sepsis_extended : pd.DataFrame
        Processed sepsis dataset with features and outcomes
    outcome_column : str
        Name of the outcome column (default: 'cycle_time')
    decision_column : str
        Name of the treatment/decision column (default: 'admission_decision')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame in RealCause format with features, treatment (t), outcome (y),
        and placeholders for counterfactuals (y0, y1, ite)
    """
    # Define metadata columns to exclude from features
    metadata = ['case_id', 'admission_decision', 'admission_ts',
                'pre_admission_sequence', 'pre_admission_variables',
                'pre_admission_activities', 'cycle_time']
    
    # Get feature column names (exclude metadata)
    feature_names = list(df_sepsis_extended.columns[~df_sepsis_extended.columns.isin(metadata)])
    
    # Create the RealCause dataset
    df_for_realcause = df_sepsis_extended[feature_names].copy()
    
    # Add treatment variable (t)
    df_for_realcause['t'] = df_sepsis_extended[decision_column]
    
    # Add outcome variable (y)
    df_for_realcause['y'] = df_sepsis_extended[outcome_column]
    
    # Add placeholders for counterfactuals
    df_for_realcause["y0"] = np.nan
    df_for_realcause["y1"] = np.nan
    df_for_realcause["ite"] = np.nan
    
    return df_for_realcause


def encode_treatment_variable(df_for_realcause, treatment_mapping={'IC': 1, 'NC': 0}):
    """
    Encode the treatment variable with numeric values.
    
    Parameters:
    -----------
    df_for_realcause : pd.DataFrame
        DataFrame with treatment column 't'
    treatment_mapping : dict
        Mapping from treatment labels to numeric values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded treatment variable
    """
    df_encoded = df_for_realcause.copy()
    df_encoded['t'] = df_encoded['t'].map(treatment_mapping)
    return df_encoded


def convert_boolean_columns(df_for_realcause):
    """
    Convert boolean string columns to numeric format.
    
    Parameters:
    -----------
    df_for_realcause : pd.DataFrame
        DataFrame that may contain string boolean values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with boolean strings converted to numeric (0/1)
    """
    df_converted = df_for_realcause.copy()
    
    # Get object columns (potential string columns)
    object_columns = df_converted.select_dtypes(include='object').columns.tolist()
    
    # Convert string "True"/"False" values to boolean
    for col in object_columns:
        if df_converted[col].dtype == 'object':
            # Check if all values in column are either True, False or NaN
            unique_values = df_converted[col].dropna().unique()
            if set(unique_values).issubset({'True', 'False', True, False}):
                # Convert to bool type
                df_converted[col] = df_converted[col].map({
                    'True': 1, 'False': 0, True: 1, False: 0
                })
                
                # Convert column type to int
                df_converted[col] = df_converted[col].astype('int')
    
    return df_converted


def filter_outliers(df_for_realcause, outcome_column='y', max_value=65760):
    """
    Filter outliers from the outcome variable.
    
    Parameters:
    -----------
    df_for_realcause : pd.DataFrame
        DataFrame with outcome column
    outcome_column : str
        Name of the outcome column to filter
    max_value : float
        Maximum allowed value for the outcome variable
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers filtered out
    """
    df_filtered = df_for_realcause[df_for_realcause[outcome_column] <= max_value].reset_index(drop=True)
    
    print(f"Filtered dataset: {len(df_filtered)} cases (removed {len(df_for_realcause) - len(df_filtered)} outliers)")
    
    return df_filtered


def clean_dataset(df_for_realcause, columns_to_drop=None):
    """
    Clean the dataset by removing specified columns.
    
    Parameters:
    -----------
    df_for_realcause : pd.DataFrame
        DataFrame to clean
    columns_to_drop : list or None
        List of column names to drop (default: ['Diagnose'])
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    if columns_to_drop is None:
        columns_to_drop = ['Diagnose']
    
    df_clean = df_for_realcause.copy()
    
    # Drop columns that exist in the dataset
    existing_columns = [col for col in columns_to_drop if col in df_clean.columns]
    if existing_columns:
        df_clean = df_clean.drop(columns=existing_columns)
        print(f"Dropped columns: {existing_columns}")
    
    return df_clean


def save_realcause_dataset(df_for_realcause, filename="datasets/sepsis_cases.csv"):
    """
    Save the processed dataset in RealCause format.
    
    Parameters:
    -----------
    df_for_realcause : pd.DataFrame
        Processed dataset in RealCause format
    filename : str
        Output filename
        
    Returns:
    --------
    None
    """
    df_for_realcause.to_csv(filename, index=None)
    print(f"Dataset saved to {filename}")
    print(f"Shape: {df_for_realcause.shape}")
    print(f"Columns: {list(df_for_realcause.columns)}")


def complete_transformation_pipeline(df_sepsis_extended, outcome_column='cycle_time',
                                   decision_column='admission_decision',
                                   treatment_mapping={'IC': 1, 'NC': 0},
                                   max_outcome_value=65760,
                                   columns_to_drop=None,
                                   output_file="datasets/sepsis_cases.csv"):
    """
    Complete transformation pipeline from processed sepsis data to RealCause format.
    
    Parameters:
    -----------
    df_sepsis_extended : pd.DataFrame
        Processed sepsis dataset with features and outcomes
    outcome_column : str
        Name of the outcome column
    decision_column : str
        Name of the treatment/decision column
    treatment_mapping : dict
        Mapping from treatment labels to numeric values
    max_outcome_value : float
        Maximum allowed value for outcome variable (for outlier filtering)
    columns_to_drop : list or None
        List of columns to drop from final dataset
    output_file : str
        Output filename for the final dataset
        
    Returns:
    --------
    pd.DataFrame
        Final dataset in RealCause format
    """
    print("Starting transformation pipeline...")
    
    # Step 1: Prepare basic RealCause format
    df_realcause = prepare_realcause_dataset(df_sepsis_extended, outcome_column, decision_column)
    print(f"Step 1 - Initial RealCause format: {df_realcause.shape}")
    
    # Step 2: Encode treatment variable
    df_realcause = encode_treatment_variable(df_realcause, treatment_mapping)
    print(f"Step 2 - Treatment encoded")
    
    # Step 3: Convert boolean columns
    df_realcause = convert_boolean_columns(df_realcause)
    print(f"Step 3 - Boolean columns converted")
    
    # Step 4: Filter outliers
    df_realcause = filter_outliers(df_realcause, 'y', max_outcome_value)
    print(f"Step 4 - Outliers filtered: {df_realcause.shape}")
    
    # Step 5: Clean dataset
    df_realcause = clean_dataset(df_realcause, columns_to_drop)
    print(f"Step 5 - Dataset cleaned: {df_realcause.shape}")
    
    # Step 6: Save dataset
    save_realcause_dataset(df_realcause, output_file)
    
    print("Transformation pipeline completed!")
    
    return df_realcause