"""
File for loading Sepsis dataset. This dataset contains patient data related to sepsis cases.
The dataset includes various medical measurements and has treatment (t) and outcome (y) columns.
"""

import os
import pandas as pd
import numpy as np
from utils import to_data_format, NUMPY, PANDAS_SINGLE, DATA_FOLDER

def load_sepsis(data_format=NUMPY, binary_treatment=True, dataroot=None, return_ate=False, return_ites=False, variant=None):
    """
    Load the sepsis dataset.
    
    Parameters:
    data_format (str): Format to return the data in (numpy, pandas, or torch)
    binary_treatment (bool): Whether to ensure treatment is binary
    dataroot (str): Root directory for data
    return_ate (bool): Whether to return the average treatment effect
    return_ites (bool): Whether to return individual treatment effects
    
    Returns:
    Depending on the parameters, returns:
    - w, t, y: covariates, treatment, and outcome
    - If return_ate=True, also returns the average treatment effect
    - If return_ites=True, also returns individual treatment effects
    """
    if dataroot is None:
        dataroot = DATA_FOLDER
      # Load the dataset
    csv_path = os.path.join(dataroot, 'sepsis_cases.csv')
    print(f"Loading sepsis dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter by variant if provided
    if variant is not None and 'variant' in df.columns:
        df = df[df['variant'] == variant]
    
    # df = df.drop(columns=['variant'])
    # Extract treatment (t) and outcome (y)
    t = df['t']
    y = df['y']
    # All other columns except y, t, y0, y1, and ite are considered covariates
    w_cols = [col for col in df.columns if col not in ['t', 'y', 'y0', 'y1', 'ite', 'variant']]
    w = df[w_cols]

    w = w.fillna(-1)
    
    # Process categorical variables - convert to numeric
    for col in w.columns:
        if w[col].dtype == 'object' or w[col].dtype == 'bool':
            # If column contains boolean or string values, convert to numeric
            if w[col].dtype == 'bool':
                w[col] = w[col].astype(int)
            else:
                print(col)
                # For string columns, use one-hot encoding
                w = pd.get_dummies(w, columns=[col], prefix=col)
    
    # Fill any NaN values with -1
    
    
    # Calculate ATE and ITEs if available in the dataset and requested
    ate = None
    ites = None
    if 'y0' in df.columns and 'y1' in df.columns and not df['y0'].isnull().all() and not df['y1'].isnull().all():
        if return_ate:
            y0 = df['y0']
            y1 = df['y1']
            ate = float(np.mean(y1 - y0))
        
        if return_ites:
            ites = df['y1'] - df['y0']
    
    # Create return dictionary
    result = to_data_format(data_format, w, t, y)
    
    if return_ate:
        if isinstance(result, tuple):
            result = {'w': result[0], 't': result[1], 'y': result[2], 'ate': ate}
        else:
            result['ate'] = ate
    
    if return_ites:
        if isinstance(result, tuple) and 'ate' not in locals():
            result = {'w': result[0], 't': result[1], 'y': result[2], 'ites': ites}
        else:
            result['ites'] = ites
    
    return result
