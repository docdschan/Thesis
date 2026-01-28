import pandas as pd
import random
from typing import List, Tuple

def inject_synthetic_drift(
    X: pd.DataFrame, 
    drift_start_idx: int, 
    magnitude_sigma: float = 3.0, 
    affected_ratio: float = 0.5,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Injects a sudden mean shift drift into a subset of features.
    The drift magnitude is scaled relative to each feature's standard deviation
    to ensure comparable severity across different units (MW, %, Euro, etc.).

    Parameters
    ----------
    X : pd.DataFrame
        The original feature matrix.
    drift_start_idx : int
        The row index where the drift begins.
    magnitude_sigma : float
        The strength of the drift in standard deviations (e.g., 3.0 = strong drift).
    affected_ratio : float
        The proportion of features to be drifted (0.0 to 1.0).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    X_drifted : pd.DataFrame
        The modified dataframe containing the drift.
    drifted_features : List[str]
        List of column names that were modified.
    """
    X_drifted = X.copy()
    n_features = X.shape[1]
    
    # 1. Determine which features to drift
    n_affected = max(1, int(n_features * affected_ratio))
    
    random.seed(random_seed)
    drift_candidates = list(X.columns)
    
    # Optional: Filter out pure noise columns if needed, but for now we drift anything
    affected_features = random.sample(drift_candidates, n_affected)
    
    print(f"[INFO] Injecting drift into {n_affected}/{n_features} features starting at index {drift_start_idx}.")
    
    # 2. Apply Drift
    for feature in affected_features:
        # Calculate std deviation of the pre-drift phase to scale magnitude
        # Use global std for simplicity in this batch experiment
        std_val = X[feature].std()
        
        # If feature is constant (std=0), fallback to adding raw magnitude
        if std_val == 0:
            shift_val = magnitude_sigma
        else:
            shift_val = magnitude_sigma * std_val
        
        # Add shift from start_idx to the end
        # We use .iloc for integer-based indexing on rows, and .columns.get_loc for column index
        col_idx = X_drifted.columns.get_loc(feature)
        X_drifted.iloc[drift_start_idx:, col_idx] += shift_val

    return X_drifted, affected_features