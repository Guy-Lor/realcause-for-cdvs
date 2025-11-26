"""
Sepsis Data Loading and Initial Processing Module

This module provides functions to load and preprocess the sepsis dataset from XES format,
extracting case IDs, activities, timestamps, and admission decisions.
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
try:
    import pm4py
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.log.importer.xes import importer as xes_importer
except ImportError:
    raise ImportError("pm4py is required for this module. Please install it via 'pip install pm4py'.")

def _hash(cols):
    """Generate a short hash from a list of column names."""
    return hashlib.md5(",".join(sorted(cols)).encode()).hexdigest()[:8]


def load_sepsis_data(xes_file_path="datasets/Sepsis Cases - Event Log.xes.gz"):
    """
    Load sepsis event log data from XES file and convert to DataFrame format.
    
    Parameters:
    -----------
    xes_file_path : str
        Path to the XES file containing the sepsis event log
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: case_id, act (activity), ts (timestamp), and other event attributes
    """
    xes = Path(xes_file_path).expanduser()
    log = xes_importer.apply(str(xes))
    df = log_converter.apply(log, variant=log_converter.TO_DATA_FRAME)
    
    # Standardize column names
    df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "act",
        "time:timestamp": "ts"
    }, inplace=True)
    
    return df


def extract_admission_decisions(df):
    """
    Extract admission decisions (IC/NC) for each case from the event log.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log DataFrame with case_id, act, and ts columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with case_id, admission_decision, and admission_ts columns
    """
    cases = df['case_id'].unique()
    df_sepsis = pd.DataFrame({'case_id': cases})
    admission_data = []

    for case_id in cases:
        case_df = df[df['case_id'] == case_id]
        
        # Get all admission events (both IC and NC)
        admission_events = case_df[case_df['act'].isin(['Admission IC', 'Admission NC'])]
        
        if not admission_events.empty:
            # Sort by timestamp to get the earliest admission
            first_admission = admission_events.sort_values('ts').iloc[0]
            
            # Extract the admission type from the activity name
            if first_admission['act'] == 'Admission IC':
                admission_type = 'IC'
            elif first_admission['act'] == 'Admission NC':
                admission_type = 'NC'
            else:
                raise ValueError(f"Unexpected admission activity: {first_admission['act']}")
            admission_ts = first_admission['ts']
        else:
            admission_type = None
            admission_ts = None
        
        admission_data.append({
            'case_id': case_id,
            'admission_decision': admission_type,
            'admission_ts': admission_ts
        })

    # Convert to dataframe and merge with df_sepsis
    admission_df = pd.DataFrame(admission_data)
    df_sepsis = pd.merge(df_sepsis, admission_df, on='case_id', how='left')
    
    # Filter out cases without admission decisions
    df_sepsis = df_sepsis[df_sepsis['admission_decision'].notna()]
    
    return df_sepsis


def extract_activity_sequences(df, df_sepsis):
    """
    Extract pre-admission activity sequences for each case.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full event log DataFrame
    df_sepsis : pd.DataFrame
        DataFrame with admission decisions
        
    Returns:
    --------
    pd.DataFrame
        df_sepsis with added pre_admission_sequence column
    """
    activity_sequences = {}
    
    for case_id, case_df in df.groupby('case_id'):
        case_df = case_df.sort_values('ts')
        first_admission = case_df[case_df['act'].isin(['Admission IC', 'Admission NC'])]
        
        if not first_admission.empty:
            first_admission_ts = first_admission['ts'].iloc[0]
            pre_admission_acts = sorted(case_df[case_df['ts'] < first_admission_ts]['act'].tolist())
        else:
            pre_admission_acts = sorted(case_df['act'].tolist())
        
        activity_sequences[case_id] = ','.join(pre_admission_acts)

    # Add the sequence to the dataframe
    df_sepsis = df_sepsis.copy()
    df_sepsis['pre_admission_sequence'] = df_sepsis['case_id'].map(activity_sequences)
    
    return df_sepsis