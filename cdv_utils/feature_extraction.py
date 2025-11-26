"""
Feature Extraction Module for Sepsis Dataset

This module provides functions to extract features from sepsis event log data,
focusing on pre-admission variables and measurements.
"""

import pandas as pd
import numpy as np


def extract_pre_admission_variables_with_values(df):
    """
    Extract all variables before admission with their values, 
    numbering repeated measurements sequentially.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame with case_id, act, ts, and other attribute columns
        
    Returns:
    --------
    dict
        Dictionary mapping case_id to list of pre-admission variables
    """
    admission_acts = ['Admission IC', 'Admission NC']
    pre_admission_data = {}
    
    for case_id, case_df in df.groupby('case_id'):
        # Sort by timestamp
        case_df = case_df.sort_values('ts')
        
        # Find first admission decision (if any)
        first_admission = case_df[case_df['act'].isin(admission_acts)]
        
        if not first_admission.empty:
            # Get the timestamp of first admission
            first_admission_ts = first_admission['ts'].iloc[0]
            
            # Get all events before this timestamp
            pre_admission_events = case_df[case_df['ts'] < first_admission_ts]
        else:
            # If no admission, use all events
            pre_admission_events = case_df
        
        # Dictionary to track how many times we've seen each variable
        var_counts = {}
        
        # Dictionary to store the variables and their values
        case_variables = {}
        
        # Go through events in chronological order
        for _, row in pre_admission_events.iterrows():
            # Get all columns that have non-null values
            for col in df.columns:
                if col not in ['case_id', 'act', 'ts', 'lifecycle:transition','org:group'] and not pd.isna(row[col]):
                    # Count occurrences of this variable
                    var_counts[col] = var_counts.get(col, 0) + 1
                    
                    # Create variable name with sequence number
                    var_name = f"{col}_{var_counts[col]}" if var_counts[col] > 1 else col
                    
                    # Store the variable and its value
                    case_variables[var_name] = row[col]
        
        # Store variables for this case
        pre_admission_data[case_id] = sorted(case_variables)
    
    return pre_admission_data


def extract_pre_admission_features(df, df_sepsis):
    """
    Extract all features from events before admission timestamp for each case_id.
    If multiple values exist for a column, create numbered columns (_2, _3, etc.)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full event log DataFrame
    df_sepsis : pd.DataFrame
        DataFrame with admission decisions and timestamps
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted pre-admission features for each case
    """
    # Initialize an empty dataframe to store results
    result_df = pd.DataFrame(index=df_sepsis['case_id'].unique())
    
    # Get all possible feature columns (excluding metadata columns)
    feature_cols = [col for col in df.columns if col not in ['case_id', 'act', 'ts', 
                                                             'lifecycle:transition', 'org:group']]
    
    # Process each case
    for case_id in df_sepsis['case_id'].unique():
        # Get admission timestamp for this case
        admission_ts = df_sepsis.loc[df_sepsis['case_id'] == case_id, 'admission_ts'].iloc[0] \
                        if not df_sepsis.loc[df_sepsis['case_id'] == case_id, 'admission_ts'].isna().iloc[0] else None
        
        # Get all events for this case
        case_events = df[df['case_id'] == case_id]
        
        # If admission timestamp exists, filter events before admission
        if admission_ts is not None:
            case_events = case_events[case_events['ts'] < admission_ts]
        
        # Sort by timestamp to process events chronologically
        case_events = case_events.sort_values('ts')
        
        # Store activity sequence before admission
        result_df.at[case_id, 'pre_admission_activities'] = ','.join(sorted(case_events['act'].tolist()))
        
        # Process each feature column
        for col in feature_cols:
            # Track how many times we've seen this column
            col_count = 0
            
            # Go through events in chronological order
            for _, row in case_events.iterrows():
                # If column has a non-null value
                if not pd.isna(row[col]):
                    col_count += 1
                    # Create column name with suffix if needed
                    col_name = f"{col}_{col_count}" if col_count > 1 else col
                    # Store the value
                    result_df.at[case_id, col_name] = row[col]
        
        # Calculate cycle time before admission
        if not case_events.empty:
            cycle_time = (case_events['ts'].max() - case_events['ts'].min()).total_seconds()
            result_df.at[case_id, 'cycle_time'] = cycle_time
    
    # Reset index to get case_id as a column
    result_df = result_df.reset_index().rename(columns={'index': 'case_id'})
    
    return result_df


def assign_variants(df_sepsis_extended, k=3):
    """
    Assign variant numbers based on pre-admission variables frequency.
    
    Parameters:
    -----------
    df_sepsis_extended : pd.DataFrame
        DataFrame with pre_admission_variables column
    k : int
        Number of variants (top k-1 most frequent + 1 for others)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added variant column
    """
    # Count the occurrences of each "pre_admission_variables" group
    variant_counts = df_sepsis_extended['pre_admission_variables'].value_counts()
    
    # Get the top k-1 most frequent variants
    top_variants = variant_counts.index[:k-1]
    
    # Create a new "variant" column
    def assign_variant(row):
        for i, variant in enumerate(top_variants, start=1):
            if row['pre_admission_variables'] == variant:
                return i  # Assign variant number based on rank
        return k  # Assign the k-th variant for all others
    
    df_result = df_sepsis_extended.copy()
    df_result['variant'] = df_result.apply(assign_variant, axis=1)
    
    return df_result