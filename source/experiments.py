import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from typing import Any, Dict, List
from river import drift, preprocessing
import time
from collections import deque
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Any

from src.detectors import RiverCompatiblePCA
from src.detectors import AdaptiveModel, GroupedWassersteinDetector

# Optional: Import River for ADWIN (System 1)
try:
    from river import drift
except ImportError:
    drift = None

def run_comparison_experiment(
    X_drifted: pd.DataFrame, 
    drift_start_idx: int,
    affected_features: List[str],
    scenario_name: str
) -> Dict[str, Any]:
    """
    Runs the online drift detection loop comparing Baseline (Univariate) vs. Mitigation (PCA-based).
    """
    print(f"\n[START] Running experiment for scenario: '{scenario_name}' (Features: {X_drifted.shape[1]})")
    
    # 1. Initialize Models
    
    # Preprocessing: Incremental Standard Scaler
    scaler = preprocessing.StandardScaler()
    
    # Approach A: Baseline (Univariate KSWIN on *every* feature)
    baseline_detectors = {
        col: drift.KSWIN(window_size=200, stat_size=30) 
        for col in X_drifted.columns
    }
    
    # Approach B: Mitigation Strategy (PCA Wrapper + KSWIN)
    n_components = 3
    # If there are fewer features than components (e.g., low-dim), limit
    n_components = min(n_components, X_drifted.shape[1])
    
    pca = RiverCompatiblePCA(n_components=n_components)
    
    # Detector on the first Principal Component (PC1)
    pca_detector = drift.KSWIN(window_size=200, stat_size=30)
    
    # 2. Metric Containers 
    results = {
        "scenario": scenario_name,
        "n_features": X_drifted.shape[1],
        "baseline_fp_count": 0,
        "baseline_detection_step": None,
        "mitigation_fp_count": 0,
        "mitigation_detection_step": None
    }
    
    start_time = time.time()
    
    # 3. The Streaming Loop
    for i, (idx, row) in enumerate(X_drifted.iterrows()):
        
        # Convert row to dictionary for river
        x_raw = row.to_dict()
        
        # Incremental Scaling
        scaler.learn_one(x_raw)
        x_scaled = scaler.transform_one(x_raw)
        
        # A: Baseline Detection (Checking all features)
        baseline_alarm_this_step = False
        
        for feature_name, val in x_scaled.items():
            detector = baseline_detectors[feature_name]
            detector.update(val)
            
            if detector.drift_detected:
                baseline_alarm_this_step = True
        
        if baseline_alarm_this_step:
            if i < drift_start_idx:
                results["baseline_fp_count"] += 1
            elif i >= drift_start_idx and results["baseline_detection_step"] is None:
                results["baseline_detection_step"] = i
                
        # B: Mitigation Detection (PCA-based) 
        # 1. Update PCA with scaled vector (via Wrapper)
        pca.learn_one(x_scaled)
        
        # 2. Transform to latent space (via Wrapper)
        x_latent = pca.transform_one(x_scaled)
        
        # 3. Monitor PC1 (Key 0 in the dictionary)
        pc1_val = x_latent.get(0, 0.0) 
        pca_detector.update(pc1_val)
        
        if pca_detector.drift_detected:
            if i < drift_start_idx:
                results["mitigation_fp_count"] += 1
            elif i >= drift_start_idx and results["mitigation_detection_step"] is None:
                results["mitigation_detection_step"] = i

        if i % 5000 == 0:
            print(f"   Step {i}/{len(X_drifted)} processed...")

    # 4. Final Calculations 
    if results["baseline_detection_step"]:
        results["baseline_latency"] = results["baseline_detection_step"] - drift_start_idx
    else:
        results["baseline_latency"] = None 

    if results["mitigation_detection_step"]:
        results["mitigation_latency"] = results["mitigation_detection_step"] - drift_start_idx
    else:
        results["mitigation_latency"] = None

    # FPR Calculation
    n_stable_steps = drift_start_idx
    results["baseline_fpr"] = results["baseline_fp_count"] / max(1, n_stable_steps)
    results["mitigation_fpr"] = results["mitigation_fp_count"] / max(1, n_stable_steps)
    
    duration = time.time() - start_time
    print(f"[DONE] Finished in {duration:.2f}s. Baseline FPR: {results['baseline_fpr']:.4f} | PCA FPR: {results['mitigation_fpr']:.4f}")
    
    return results


def run_adaptive_forecasting_experiment(
    X: pd.DataFrame, 
    y: pd.Series, 
    detector_type: str,
    feature_groups: Dict = None
) -> Dict[str, Any]:
    """
    Executes the Adaptive Forecasting Experiment (RO3).
    
    Workflow:
    1. Train XGBoost on initial 20% of data.
    2. Stream the remaining data.
    3. Monitor for drift using the selected detector strategy.
    4. Retrain model if drift is detected.
    5. Track RMSE and retraining count.
    
    Parameters
    ----------
    detector_type : str
        'baseline' (no retraining), 
        'adwin' (System 1: Output monitoring), 
        'grouped' (System 3: Local Input monitoring / Wasserstein)
    """
    print(f"\n[EXP] Starting Adaptive Forecasting with Strategy: {detector_type.upper()}")
    
    # 1. Split Data (Warm-Start vs. Stream)
    initial_train_size = 35000 
    X_train_init = X.iloc[:initial_train_size]
    y_train_init = y.iloc[:initial_train_size]
    
    X_stream = X.iloc[initial_train_size:]
    y_stream = y.iloc[initial_train_size:]
    
    # 2. Initialize and Train Model
    model = AdaptiveModel()
    model.train(X_train_init, y_train_init)
    drift_reasons = []
    
    # 3. Initialize Detector
    detector = None
    if detector_type == 'adwin':
        if drift is None:
            raise ImportError("River library not found. Cannot use ADWIN.")
        # delta = sensitivity (lower means more sensitive)
        detector = drift.ADWIN(delta=0.002) 
        
    elif detector_type == 'grouped':
        if feature_groups is None:
            raise ValueError("Feature groups must be provided for 'grouped' strategy.")

        detector = GroupedWassersteinDetector(
            feature_groups,
            window_size=500,
            threshold=25.0,
            check_every=20,
            ref_sample_size=2000,
            random_state=42,
        )
        detector.fit_reference(X_train_init)
        
    # 4. Storage for Metrics
    predictions = []
    actuals = []
    retrain_points = []
    
    # Sliding History for Retraining (Last N samples)
    retrain_window_size = 10000
    history_X = deque(maxlen=retrain_window_size)
    history_y = deque(maxlen=retrain_window_size)
    
    # Pre-fill history with initial training data
    for idx in range(len(X_train_init)):
        history_X.append(X_train_init.iloc[idx])
        history_y.append(y_train_init.iloc[idx])

    # 5. Streaming Simulation Loop
    for i in range(len(X_stream)):

        if i % 5000 == 0 and i > 0:
            print(f"   Stream step {i}/{len(X_stream)} | Retrainings so far: {len(retrain_points)}")

        # Extract current single row
        x_curr = X_stream.iloc[[i]] # DataFrame
        y_curr = y_stream.iloc[i]   # Scalar
        
        # A. Predict (using current model state)
        y_pred = model.predict(x_curr)[0]
        predictions.append(y_pred)
        actuals.append(y_curr)
        
        # B. Monitor Drift
        drift_detected = False
        
        # Update History
        history_X.append(x_curr.iloc[0])
        history_y.append(y_curr)
        
        if detector_type == 'baseline':
            # Baseline never retrains
            pass 
            
        elif detector_type == 'adwin':
            # Monitor Absolute Error
            error = abs(y_curr - y_pred)
            # Normalize error slightly to help ADWIN (optional but recommended)
            detector.update(error)
            if detector.drift_detected:
                drift_detected = True

        elif detector_type == 'grouped':
            # KORREKTUR: Nur EIN Aufruf der Update-Funktion!
            is_alarm, details = detector.update(x_curr.iloc[0])
            
            if is_alarm:
                drift_detected = True
                
                # Jetzt nutzen wir die 'details' aus dem einen Aufruf
                if details:
                    culprit = max(details, key=details.get)
                    drift_score = details[culprit]
                    
                    # Save: Time, group, score
                    drift_reasons.append({
                        "step": i,
                        "date": x_curr.index[0] if isinstance(x_curr.index, pd.DatetimeIndex) else i,
                        "culprit_group": culprit,
                        "score": drift_score
                    })
                
        # C. Retrain Trigger
        if drift_detected:
            if not retrain_points or (i - retrain_points[-1] > 500):

                # Retrain only on sliding history (bounded cost)
                X_retrain = pd.DataFrame(list(history_X))
                y_retrain = pd.Series(list(history_y))

                model.update(X_retrain, y_retrain)
                retrain_points.append(i)

    # 6. Final Evaluation
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"[RESULT] Strategy: {detector_type} | RMSE: {rmse:.4f} | Retrainings: {len(retrain_points)}")
    
    return {
        "strategy": detector_type,
        "rmse": rmse,
        "retrain_steps": retrain_points,
        "drift_reasons": drift_reasons,
        "predictions": predictions,
        "actuals": actuals
    }