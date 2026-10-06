import numpy as np
import pandas as pd

class WOEEncoder:
    # Train-fitted WOE encoder for numeric and categorical variables.
    def __init__(self, n_bins=10, smoothing=0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.maps_ = {}
        self.iv_ = {}
        self.bin_edges_ = {}

    def fit(self, X, y, categorical_cols):
        X = X.copy()
        y = pd.Series(y, index=X.index).astype(int)
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for col in X.columns:
            if col not in categorical_cols:
                numeric = pd.to_numeric(X[col], errors="coerce")
                quantiles = numeric.dropna().quantile(
                    np.linspace(0, 1, self.n_bins + 1)
                ).values
                edges = np.unique(quantiles)
                if len(edges) >= 3:
                    self.bin_edges_[col] = edges
                    binned = pd.cut(
                        numeric, bins=edges, include_lowest=True,
                        duplicates="drop"
                    ).astype(str)
                else:
                    binned = X[col].fillna("__MISSING__").astype(str)
            else:
                binned = X[col].fillna("__MISSING__").astype(str)

            tmp = pd.DataFrame({"bin": binned, "y": y})
            grouped = tmp.groupby("bin", dropna=False)["y"].agg(
                good=lambda z: (z == 0).sum(),
                bad=lambda z: (z == 1).sum()
            )

            good_dist = (grouped["good"] + self.smoothing) / (
                total_good + self.smoothing * len(grouped)
            )
            bad_dist = (grouped["bad"] + self.smoothing) / (
                total_bad + self.smoothing * len(grouped)
            )
            woe = np.log(good_dist / bad_dist)
            iv = ((good_dist - bad_dist) * woe).sum()

            self.maps_[col] = woe.to_dict()
            self.iv_[col] = float(iv)

        return self

    def transform(self, X, categorical_cols):
        result = {}
        for col in X.columns:
            if col not in categorical_cols and col in self.bin_edges_:
                numeric = pd.to_numeric(X[col], errors="coerce")
                binned = pd.cut(
                    numeric, bins=self.bin_edges_[col],
                    include_lowest=True, duplicates="drop"
                ).astype(str)
            else:
                binned = X[col].fillna("__MISSING__").astype(str)

            mapping = self.maps_[col]
            fallback = float(np.mean(list(mapping.values()))) if mapping else 0.0
            result[col] = binned.map(mapping).fillna(fallback).astype(float)

        return pd.DataFrame(result, index=X.index)

    def fit_transform(self, X, y, categorical_cols):
        return self.fit(X, y, categorical_cols).transform(X, categorical_cols)

    def iv_table(self):
        return (
            pd.DataFrame(
                [{"feature": k, "IV": v} for k, v in self.iv_.items()]
            )
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )
