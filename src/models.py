from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from src.config import CV_FOLDS

def get_model_pipeline(model_name):
    """Factory function to get model and param grid."""
    
    if model_name == "Logistic Regression":
        model = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))
        params = {"estimator__C": [0.1, 1, 10]}
        
    elif model_name == "SVM":
        model = OneVsRestClassifier(LinearSVC(dual="auto"))
        params = {"estimator__C": [0.1, 1, 10]}
        
    elif model_name == "Random Forest":
        model = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, random_state=42))
        params = {"estimator__n_estimators": [100, 200], "estimator__max_depth": [None, 20]}
        
    elif model_name == "KNN":
        model = OneVsRestClassifier(KNeighborsClassifier())
        params = {"estimator__n_neighbors": [3, 5, 7], "estimator__weights": ["uniform", "distance"]}
        
    elif model_name == "XGBoost":
        model = OneVsRestClassifier(XGBClassifier(eval_metric="logloss", use_label_encoder=False, n_jobs=-1))
        params = {"estimator__n_estimators": [100], "estimator__max_depth": [3, 5]}
        
    elif model_name == "LightGBM":
        model = OneVsRestClassifier(LGBMClassifier(verbose=-1, force_col_wise=True))
        params = {"estimator__num_leaves": [31], "estimator__max_depth": [-1, 20]}
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, params

def train_and_tune(X_train, Y_train, model_name):
    print(f"Training {model_name}...")
    model, params = get_model_pipeline(model_name)
    grid = GridSearchCV(model, params, cv=CV_FOLDS, scoring="f1_micro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid.best_estimator_, grid.best_params_