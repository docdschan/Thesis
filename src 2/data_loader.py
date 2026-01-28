from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

# Configuration of Feature Groups
target_col_name = "verbrauch_realisiert__netzlast_mwh"

# Definition of feature subsets
feature_definitions = {
    # 1. LOAD (Low)
    "load_metrics": [
        "verbrauch_prog__netzlast_mwh",
        "verbrauch_prog__residuallast_mwh",
        "verbrauch_realisiert__netzlast_inkl_pumpspeicher_mwh",
        "verbrauch_realisiert__pumpspeicher_mwh",
        "verbrauch_realisiert__residuallast_mwh",
        "lastverlauf_netzeinspeisung__last_mw",
        "regelzonenlast__prognose_mw",
        "regelzonenlast__ist_wert_mw",
        "vertikale_netzlast__prognose_mw",
        "vertikale_netzlast__ist_wert_mw",
    ],

    # 2. GENERATION (Mid part 1)
    "generation": [
        "erzeugung_realisiert__biomasse_mwh",
        "erzeugung_realisiert__wasserkraft_mwh",
        "erzeugung_realisiert__wind_onshore_mwh",
        "erzeugung_realisiert__photovoltaik_mwh",
        "erzeugung_realisiert__sonstige_erneuerbare_mwh",
        "erzeugung_realisiert__steinkohle_mwh",
        "erzeugung_realisiert__erdgas_mwh",
        "erzeugung_realisiert__pumpspeicher_mwh",
        "erzeugung_realisiert__sonstige_konventionelle_mwh",
        "erzeugung_prog_day_ahead__gesamt_mwh",
        "erzeugung_prog_day_ahead__photovoltaik_und_wind_mwh",
        "pv_einspeisung__prognose_mw",
        "wind_einspeisung__prognose_mw",
    ],
    
    # 3. GRID PHYSICALS & CAPACITY (Mid part 2)
    "grid_physical_and_capacity": [
        # Installed Capacities 
        "erzeugung_installierte_leistung__biomasse_mw",
        "erzeugung_installierte_leistung__wasserkraft_mw",
        "erzeugung_installierte_leistung__wind_onshore_mw",
        "erzeugung_installierte_leistung__photovoltaik_mw",
        "erzeugung_installierte_leistung__steinkohle_mw",
        "erzeugung_installierte_leistung__erdgas_mw",
        "erzeugung_installierte_leistung__pumpspeicher_mw",
        
        # Physical Flows 
        "markt_physikalischer_stromfluss__daenemark_1_export_mwh",
        "markt_physikalischer_stromfluss__daenemark_1_import_mwh",
        "markt_physikalischer_stromfluss__niederlande_export_mwh", 
        "markt_physikalischer_stromfluss__niederlande_import_mwh",
        "markt_physikalischer_stromfluss__schweden_4_export_mwh",
        "markt_physikalischer_stromfluss__schweden_4_import_mwh",
        "markt_physikalischer_stromfluss__tschechien_export_mwh",
        "markt_physikalischer_stromfluss__tschechien_import_mwh",
        "markt_physikalischer_stromfluss__norwegen_export_mwh",
        "markt_physikalischer_stromfluss__norwegen_import_mwh",
    ]
}

def get_experimental_dataset(df: pd.DataFrame, scenario: str = "high") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix X and target vector y based on the selected scenario.
    
    Updated Logic:
    - 'low': Restricted to load metrics (~10 features).
    - 'mid': Load + Generation + Physical Flows + Capacity.
    - 'high': ALL available features (~104 features).
    - 'noise': 'high' + 50 noise features (~154 features).
    """
    # 1. Extract Target
    if target_col_name not in df.columns:
        raise ValueError(f"Target column '{target_col_name}' not found in dataframe!")
    
    y = df[target_col_name]

    # 2. Select Features based on Scenario
    if scenario == "low":
        # Strict subset: Only load metrics
        selected_cols = [c for c in feature_definitions["load_metrics"] if c in df.columns]
        x_data = df[selected_cols].copy()
        
    elif scenario == "mid":
        # Expanded subset: Load + Generation + Physical/Capacity
        wanted_cols = (feature_definitions["load_metrics"] + 
                       feature_definitions["generation"] + 
                       feature_definitions["grid_physical_and_capacity"])
        
        selected_cols = [c for c in wanted_cols if c in df.columns]
        x_data = df[selected_cols].copy()
        
    elif scenario in ["high", "noise"]:
        # MAX DIMENSIONALITY: Use ALL columns except the target
        selected_cols = [c for c in df.columns if c != target_col_name]
        x_data = df[selected_cols].copy()
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # 3. Noise Injection
    if scenario == "noise":
        n_rows = len(x_data)
        n_noise_features = 50 
        noise_matrix = np.random.normal(loc=0, scale=1, size=(n_rows, n_noise_features))
        noise_df = pd.DataFrame(
            noise_matrix, 
            index=x_data.index, 
            columns=[f"noise_{i}" for i in range(n_noise_features)]
        )
        x_data = pd.concat([x_data, noise_df], axis=1)
        print(f"[INFO] Added {n_noise_features} synthetic noise features for scenario '{scenario}'.")

    print(f"[INFO] Dataset prepared for scenario '{scenario}': X shape={x_data.shape}, y shape={y.shape}")
    
    return x_data, y

def load_clean_data(base_dir: Path) -> pd.DataFrame:
    """
    Loads the dataset (Parquet preferred, else CSV), performs trimming of artifacts
    (first/last rows), and imputes missing values to ensure a continuous stream.

    Processing Steps:
    1. Load data (Parquet priority).
    2. Ensure DatetimeIndex and sort chronologically.
    3. Trimming: Remove the first row and the last 4 rows to eliminate aggregation artifacts.
    4. Imputation: Apply Forward-Fill (ffill) followed by Backward-Fill (bfill).
    5. Final Cleanup: Drop columns/rows that remain fully empty.

    Parameters
    ----------
    base_dir : Path
        Directory containing 'data.parquet' or 'data.csv'.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe indexed by timestamp, ready for drift injection.
    """
    parquet_path = base_dir / "data_5y.parquet"
    csv_path = base_dir / "data_5y.csv"

    # 1. Load Data
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"[INFO] Loaded data from: {parquet_path}")
    elif csv_path.exists():
        # Adjust separators if necessary based on your raw data format
        df = pd.read_csv(csv_path) 
        print(f"[INFO] Loaded data from: {csv_path}")
    else:
        raise FileNotFoundError(
            f"Neither data.parquet nor data.csv found in {base_dir}"
        )

    # 2. Handle Datetime Index
    # If a specific column like 'ts' or 'timestamp' exists, convert it.
    # Otherwise, assume index is already datetime or convertible.
    time_col_candidates = ['ts', 'timestamp', 'date', 'datum']
    
    # Check if index is already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        found_col = None
        for col in df.columns:
            if col.lower() in time_col_candidates:
                found_col = col
                break
        
        if found_col:
            df[found_col] = pd.to_datetime(df[found_col])
            df = df.set_index(found_col)
            print(f"[INFO] Set index using column: {found_col}")
        else:
            # Fallback: Try converting the existing index
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"[WARNING] Could not automatically parse index as datetime: {e}")

    df = df.sort_index()
    print(f"[INFO] Shape before cleaning: {df.shape}")
    print(f"[INFO] NaNs before cleaning: {df.isnull().sum().sum()}")

    # 3. Trimming (Remove Edge Artifacts)
    # Remove 1st row (start artifact) and last 4 rows (end aggregation lag)
    if len(df) > 10: 
        df = df.iloc[1:-4].copy()
    
    # 4. Imputation (Fill Gaps)
    # Forward fill first (assume state persistence), then backfill for initial gaps
    df = df.ffill().bfill()

    # 5. Final Sanity Check
    # If columns are fully empty (all NaNs), ffill/bfill won't help -> drop them.
    if df.isnull().sum().sum() > 0:
        print("[WARNING] Remaining NaNs detected. Dropping fully empty columns/rows...")
        df = df.dropna(axis=1, how='all') # Drop empty columns
        df = df.dropna(axis=0, how='any') # Drop rows with remaining NaNs
    
    print(f"[INFO] Shape after cleaning: {df.shape}")
    print(f"[INFO] Final NaN count: {df.isnull().sum().sum()}")
    
    return df