from sklearn.base import BaseEstimator, ClassifierMixin


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