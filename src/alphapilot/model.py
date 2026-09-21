"""Walk-forward (expanding window) LightGBM classifier.

No random train/test splits — each fold trains on all data up to split point
and predicts the next day only.  The model is retrained every 21 trading days
(monthly) to keep runtime practical; predictions are still fully out-of-sample
because each prediction uses only data available before that day.
Scaler is fit inside each fold on train data only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {"target"}


@dataclass
class WalkForwardResult:
    predictions: pd.Series          # predicted probabilities, indexed by date
    feature_importance: pd.Series   # mean importance across folds
    n_folds: int = 0
    train_accuracy: list[float] = field(default_factory=list)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def walk_forward_predict(
    feature_df: pd.DataFrame,
    mcfg: dict,
) -> WalkForwardResult:
    """Run expanding-window walk-forward and return out-of-sample probabilities.

    For each day t >= min_train_size, the model is trained on [0, t) and
    predicts day t.  This is the strictest possible no-leakage setup.

    To keep runtime reasonable we retrain every 21 trading days (monthly).
    """
    feat_cols = get_feature_cols(feature_df)
    X = feature_df[feat_cols].values
    y = feature_df["target"].values
    dates = feature_df.index

    min_train = mcfg["min_train_size"]
    retrain_freq = 21  # retrain monthly

    probs = np.full(len(feature_df), np.nan)
    importances: list[np.ndarray] = []
    train_accuracy: list[float] = []
    model: lgb.LGBMClassifier | None = None
    scaler: StandardScaler | None = None

    lgb_params = {
        "n_estimators": mcfg["n_estimators"],
        "learning_rate": mcfg["learning_rate"],
        "num_leaves": mcfg["num_leaves"],
        "random_state": mcfg["random_state"],
        "n_jobs": -1,
        "verbose": -1,
    }

    for t in range(min_train, len(feature_df)):
        if (t - min_train) % retrain_freq == 0:
            X_train, y_train = X[:t], y[:t]
            scaler = StandardScaler().fit(X_train)
            X_tr_scaled = scaler.transform(X_train)
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr_scaled, y_train)
            importances.append(model.feature_importances_)
            # in-sample accuracy recorded per retrain fold
            train_accuracy.append(float(model.score(X_tr_scaled, y_train)))
            logger.debug("Retrained at t=%d (train size=%d)", t, t)

        assert scaler is not None and model is not None
        x_t = scaler.transform(X[t : t + 1])
        probs[t] = model.predict_proba(x_t)[0, 1]

    result_probs = pd.Series(probs, index=dates, name="prob_up")
    mean_importance = pd.Series(
        np.mean(importances, axis=0) if importances else np.zeros(len(feat_cols)),
        index=feat_cols,
        name="importance",
    ).sort_values(ascending=False)

    logger.info(
        "Walk-forward complete: %d predictions, %d retrains",
        int(np.isfinite(probs).sum()),
        len(importances),
    )
    return WalkForwardResult(
        predictions=result_probs,
        feature_importance=mean_importance,
        n_folds=len(importances),
        train_accuracy=train_accuracy,
    )
