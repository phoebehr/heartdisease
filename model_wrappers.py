from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps a fitted Pipeline together with a custom decision threshold, so that
    .predict() reflects the exact threshold that was chosen (e.g. via Youden's J)
    rather than the classifier's default 0.5 cutoff.

    This must live in its own importable module (not inside a notebook cell) so
    that joblib/pickle can correctly reload it later in a different process —
    e.g. app.py loading a model saved from a training notebook.
    """

    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict(self, X):
        probs = self.pipeline.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    @property
    def feature_names_in_(self):
        return self.pipeline.feature_names_in_

    @property
    def n_features_in_(self):
        return self.pipeline.n_features_in_


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Learns IQR-based lower/upper bounds for the given columns from whatever
    data it's fit on, then clips (winsorizes) those columns to those bounds
    on transform.

    Placed as the FIRST step of a Pipeline, this gets refit independently on
    every CV fold (consistent with how feature selection and scaling are
    already handled elsewhere in this project) — and once the final pipeline
    is fit on the full training set, the exact same learned bounds are applied
    automatically to any new data passed through .predict(), including a
    single live patient row from app.py. This avoids the earlier bug where
    capping was done as a one-off step outside the saved pipeline, so new
    inputs at inference time were never actually capped.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        cols = self.columns if self.columns is not None else list(X.columns)
        self.bounds_ = {}
        for col in cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)

        # Set these explicitly so Pipeline.feature_names_in_ (used by app.py
        # and by ThresholdedClassifier's feature_names_in_ property) works
        # correctly — sklearn only sets these automatically for built-in
        # estimators that go through its internal validation machinery.
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features
