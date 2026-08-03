import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def compute_ml_utility(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_col: str, feature_cols: list) -> dict:
    """Trains a classifier on synthetic data and evaluates on real data."""
    # Simple preprocessing
    def preprocess(df):
        df_proc = df[feature_cols + [target_col]].copy()
        for c in df_proc.columns:
            if df_proc[c].dtype == 'object':
                le = LabelEncoder()
                df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        return df_proc
        
    try:
        real_proc = preprocess(real_data)
        syn_proc = preprocess(synthetic_data)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return {"auc_roc_diff": None, "real_auc": None, "synthetic_auc": None}
        
    # Real baseline
    X_real = real_proc.drop(columns=[target_col])
    y_real = real_proc[target_col]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    
    model_r = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_r.fit(X_train_r, y_train_r)
    preds_r = model_r.predict_proba(X_test_r)[:, 1]
    auc_r = roc_auc_score(y_test_r, preds_r)
    
    # Synthetic model on real test data
    X_syn = syn_proc.drop(columns=[target_col])
    y_syn = syn_proc[target_col]
    
    model_s = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_s.fit(X_syn, y_syn)
    preds_s = model_s.predict_proba(X_test_r)[:, 1]
    auc_s = roc_auc_score(y_test_r, preds_s)
    
    return {
        "real_auc": auc_r,
        "synthetic_auc": auc_s,
        "auc_roc_diff": abs(auc_r - auc_s)
    }
