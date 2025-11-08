from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Log transformations
        for col in ['Solids', 'Conductivity', 'Trihalomethanes']:
            X[f'{col}_log'] = np.log1p(X[col])

        # Binary features
        X['ph_out_of_range'] = ((X['ph'] < 6.5) | (X['ph'] > 8.5)).astype(int)
        X['high_solids'] = (X['Solids'] > 500).astype(int)
        X['chloramine_safe'] = ((X['Chloramines'] >= 1) & (X['Chloramines'] <= 4)).astype(int)
        X['sulfate_out_of_range'] = (X['Sulfate'] > 250).astype(int)
        X['high_organic'] = (X['Organic_carbon'] > 20).astype(int)
        X['trihalo_high'] = (X['Trihalomethanes'] > 80).astype(int)
        X['turbid'] = (X['Turbidity'] > 5).astype(int)

        # Interaction / ratio features
        X['acidity_hardness_ratio'] = X['Hardness'] / X['ph']
        X['organic_contamination_index'] = X['Organic_carbon'] * X['Trihalomethanes']
        X['mineral_index'] = X['Solids'] / X['Conductivity']
        X['chlorine_impact'] = X['Chloramines'] * X['ph']

        # Handle inf / NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        # Polynomial features
        X['pH_squared'] = X['ph'] ** 2
        X['Turbidity_squared'] = X['Turbidity'] ** 2

        return X
