"""Feature engineering shared by training and the app (prevents train/serve skew)."""
import pandas as pd

RAW_FEATURES = ["N", "P", "K", "ph"]
EPS = 1e-5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the engineered features used during training."""
    out = df.copy()
    out["N_P_ratio"] = out["N"] / (out["P"] + EPS)
    out["K_P_ratio"] = out["K"] / (out["P"] + EPS)
    out["N_K_ratio"] = out["N"] / (out["K"] + EPS)
    out["high_potassium_flag"] = (out["K"] > 150).astype(int)
    out["high_phosphorus_flag"] = (out["P"] > 100).astype(int)
    return out
