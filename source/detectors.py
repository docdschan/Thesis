import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict
from sklearn.decomposition import IncrementalPCA
from collections import deque
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Any, Optional


class RiverCompatiblePCA:
    """
    A wrapper that makes Scikit-Learn's IncrementalPCA stream-capable.
    
    SOLUTION FOR VALUE ERROR:
    It uses an internal buffer to ensure that ‘partial_fit’
    always receives enough data points (at least n_components).
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False
        self.feature_names = None
        
        # Buffer for Mini-Batches
        self.buffer = []
        
    def learn_one(self, x: Dict[str, float]):
        """
        Collects data points and trains the PCA as soon as there is enough data in the buffer.
        """
        # 1. Keep feature names consistent
        if self.feature_names is None:
            self.feature_names = sorted(list(x.keys()))
        
        # 2. Extract values and push to buffer
        vals = [x[k] for k in self.feature_names]
        self.buffer.append(vals)
        
        # 3. Check: Do we have enough data for an update?
        # Scikit-Learn requires: n_samples >= n_components
        # We take n_components or at least 5 samples to be safe
        min_batch_size = max(self.n_components, 5)
        
        if len(self.buffer) >= min_batch_size:
            X_batch = np.array(self.buffer)
            
            # Now call partial_fit safely
            self.pca.partial_fit(X_batch)
            self.is_fitted = True
            
            # Clear buffer for the next round
            self.buffer = []

    def transform_one(self, x: Dict[str, float]) -> Dict[int, float]:
        """
        Transforms a data point. If PCA is not yet trained (buffer is still filling), zeros are returned.
        """
        if not self.is_fitted:
            # As long as we are collecting the first few points, we cannot project anything yet.
            # We return 0.0 ("No signal").
            return {i: 0.0 for i in range(self.n_components)}
        
        # Ensure the same order
        vals = [x[k] for k in self.feature_names]
        X_batch = np.array([vals])
        
        # Transform
        X_transformed = self.pca.transform(X_batch)
        
        # Format back into a Dict
        return {i: val for i, val in enumerate(X_transformed[0])}


class AdaptiveModel:
    """
    Wrapper around XGBoost that allows for full retraining on new data chunks.
    Acts as the 'Patient' in our experiment.
    """
    def __init__(self, params: Dict[str, Any] = None):
        # Default parameters for a robust regression model
        if params is None:
            self.params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "verbosity": 0
            }
        else:
            self.params = params
            
        self.model = None
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains or retrains the model from scratch using the provided data.
        """
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions. Returns zeros if model is not yet fitted (cold start).
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def update(self, X_new: pd.DataFrame, y_new: pd.Series):
        """
        Triggers a retraining process on the provided window of data.
        """
        self.train(X_new, y_new)

class GroupedWassersteinDetector:
    """
    Monitors feature groups using Wasserstein distance between a reference window and a sliding window.

    Performance fixes for long streams:
    - check_every: evaluate drift only every N steps
    - ref_sample_size: downsample reference signals to reduce Wasserstein cost
    - numpy ring buffer: avoid rebuilding DataFrames at each step
    """

    def __init__(
        self,
        feature_groups: Dict[str, List[str]],
        window_size: int = 500,
        threshold: float = 0.1,
        check_every: int = 20,
        ref_sample_size: int = 2000,
        random_state: int = 42,
    ):
        self.groups = feature_groups
        self.window_size = int(window_size)
        self.threshold = float(threshold)

        self.check_every = int(check_every)
        self.ref_sample_size = int(ref_sample_size)
        self.random_state = int(random_state)

        self.is_initialized = False

        # Populated in fit_reference()
        self.feature_names: List[str] = []
        self.col_to_idx: Dict[str, int] = {}
        self.group_to_idxs: Dict[str, np.ndarray] = {}
        self.reference_stats: Dict[str, np.ndarray] = {}

        # Ring buffer state
        self._buf: Optional[np.ndarray] = None
        self._buf_pos: int = 0
        self._buf_filled: int = 0
        self._step: int = 0

    def fit_reference(self, X_ref: pd.DataFrame) -> None:
        """
        Learns the baseline distribution from the reference set (e.g., initial training window).
        """
        rng = np.random.default_rng(self.random_state)

        self.feature_names = list(X_ref.columns)
        self.col_to_idx = {c: i for i, c in enumerate(self.feature_names)}

        # Precompute group column indices (fast online aggregation)
        self.group_to_idxs = {}
        for g, cols in self.groups.items():
            idxs = [self.col_to_idx[c] for c in cols if c in self.col_to_idx]
            if idxs:
                self.group_to_idxs[g] = np.asarray(idxs, dtype=int)

        # Reference distributions: group-wise mean signal over time (then downsample)
        self.reference_stats = {}
        for g, idxs in self.group_to_idxs.items():
            group_signal = X_ref.iloc[:, idxs].mean(axis=1).to_numpy(dtype=float)
            if self.ref_sample_size > 0 and len(group_signal) > self.ref_sample_size:
                sample_idx = rng.choice(len(group_signal), size=self.ref_sample_size, replace=False)
                group_signal = group_signal[sample_idx]
            self.reference_stats[g] = group_signal

        # Init ring buffer for the current window
        self._buf = np.zeros((self.window_size, len(self.feature_names)), dtype=float)
        self._buf_pos = 0
        self._buf_filled = 0
        self._step = 0

        self.is_initialized = True

    def update(self, x_new: pd.Series) -> Tuple[bool, Dict[str, float]]:
        """
        Ingest one new observation and (optionally) compute group distances.

        Returns:
            alarm: True if any group distance exceeds threshold.
            details: Wasserstein distance per group (only when evaluated).
        """
        if not self.is_initialized or self._buf is None:
            return False, {}

        # Align incoming row to reference feature order (handles missing cols safely)
        x_vec = x_new.reindex(self.feature_names).to_numpy(dtype=float)

        # Ring buffer insert
        self._buf[self._buf_pos, :] = x_vec
        self._buf_pos = (self._buf_pos + 1) % self.window_size
        self._buf_filled = min(self.window_size, self._buf_filled + 1)
        self._step += 1

        # Only start once window is full
        if self._buf_filled < self.window_size:
            return False, {}

        # Evaluate only every N steps
        if self.check_every > 1 and (self._step % self.check_every != 0):
            return False, {}

        # Reconstruct window in correct chronological order (oldest -> newest)
        if self._buf_pos == 0:
            window = self._buf
        else:
            window = np.vstack((self._buf[self._buf_pos :, :], self._buf[: self._buf_pos, :]))

        group_distances: Dict[str, float] = {}
        alarm = False

        for g, ref_signal in self.reference_stats.items():
            idxs = self.group_to_idxs.get(g)
            if idxs is None:
                continue

            curr_signal = window[:, idxs].mean(axis=1)

            wd = float(wasserstein_distance(ref_signal, curr_signal))
            group_distances[g] = wd

            if wd > self.threshold:
                alarm = True

        return alarm, group_distances