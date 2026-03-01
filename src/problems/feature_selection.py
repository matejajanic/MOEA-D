from __future__ import annotations
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureSelection:

    def __init__(self, seed: int = 42):

        data = load_breast_cancer()
        X = data.data
        y = data.target

        self.n_features = X.shape[1]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        # Scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test

        self.model = LogisticRegression(max_iter=2000)

    def evaluate(self, X_bin: np.ndarray) -> np.ndarray:
        """
        X_bin: binary matrix (pop_size, n_features)
        returns F matrix (pop_size, 2)
        """

        F = []

        for x in X_bin:

            # If no features selected, worst possible solution
            if np.sum(x) == 0:
                F.append([1.0, 1.0])
                continue

            selected = np.where(x == 1)[0]

            X_train_sel = self.X_train[:, selected]
            X_test_sel = self.X_test[:, selected]

            self.model.fit(X_train_sel, self.y_train)
            acc = self.model.score(X_test_sel, self.y_test)

            f1 = 1.0 - acc
            f2 = np.sum(x) / self.n_features

            F.append([f1, f2])

        return np.array(F)

    def baseline_accuracy(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model.score(self.X_test, self.y_test)