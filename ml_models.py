"""
ml_models.py
Ensemble ML model: AdaBoost + PyTorch Neural Network trained on
historical price movement correlated with news sentiment.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from dataclasses import dataclass
from typing import Optional

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Ensemble will use AdaBoost only.")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Feature Engineering ────────────────────────────────────────────────────────

SECTOR_ENCODING = {
    "Information Technology": 0,
    "Health Care": 1,
    "Financials": 2,
    "Consumer Discretionary": 3,
    "Communication Services": 4,
    "Industrials": 5,
    "Consumer Staples": 6,
    "Energy": 7,
    "Utilities": 8,
    "Real Estate": 9,
    "Materials": 10,
    "General Market": 11,
}

SENTIMENT_ENCODING = {"bullish": 1, "neutral": 0, "bearish": -1}
MAGNITUDE_ENCODING = {"high": 3, "medium": 2, "low": 1}
HORIZON_ENCODING = {"intraday": 0, "short_term": 1, "medium_term": 2, "long_term": 3}


def build_features(row: pd.Series) -> np.ndarray:
    """Build a feature vector from a news + sentiment row."""
    features = [
        float(row.get("sentiment_score", 0.0)),
        float(row.get("confidence", 0.5)),
        SENTIMENT_ENCODING.get(str(row.get("sentiment", "neutral")), 0),
        MAGNITUDE_ENCODING.get(str(row.get("impact_magnitude", "low")), 1),
        HORIZON_ENCODING.get(str(row.get("time_horizon", "short_term")), 1),
        float(row.get("sector_relevance", 0.1)),
        SECTOR_ENCODING.get(str(row.get("primary_sector", "General Market")), 11),
        float(row.get("sentiment_score", 0.0)) ** 2,           # non-linear term
        float(row.get("confidence", 0.5)) * abs(float(row.get("sentiment_score", 0.0))),  # interaction
        float(row.get("sector_relevance", 0.1)) * float(row.get("confidence", 0.5)),
    ]
    return np.array(features, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([build_features(row) for _, row in df.iterrows()])


# ── Synthetic Training Data Generator ─────────────────────────────────────────

def generate_synthetic_training_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic historical sentiment-price correlation data
    for training. In production, replace with real historical data
    (e.g., from yfinance + your past scraped articles).
    """
    rng = np.random.default_rng(seed)
    rows = []

    sectors = list(SECTOR_ENCODING.keys())
    sentiments = ["bullish", "neutral", "bearish"]
    magnitudes = ["high", "medium", "low"]
    horizons = ["intraday", "short_term", "medium_term", "long_term"]

    for _ in range(n_samples):
        sentiment = rng.choice(sentiments, p=[0.4, 0.3, 0.3])
        score = rng.uniform(0.2, 1.0) * (1 if sentiment == "bullish" else -1 if sentiment == "bearish" else rng.uniform(-0.1, 0.1))
        confidence = rng.uniform(0.4, 0.95)
        magnitude = rng.choice(magnitudes, p=[0.25, 0.45, 0.30])
        horizon = rng.choice(horizons, p=[0.2, 0.4, 0.3, 0.1])
        sector = rng.choice(sectors)
        relevance = rng.uniform(0.3, 1.0)

        # Ground truth: price impact (noisy version of sentiment × confidence × magnitude)
        mag_val = MAGNITUDE_ENCODING[magnitude]
        base_impact = score * confidence * mag_val * 1.5
        noise = rng.normal(0, 0.5 * mag_val)
        # Sector-specific volatility adjustments
        sector_vol = {
            "Energy": 1.4, "Information Technology": 1.3, "Utilities": 0.6,
            "Consumer Staples": 0.7, "Financials": 1.1,
        }.get(sector, 1.0)
        actual_price_change = base_impact * sector_vol + noise

        rows.append({
            "sentiment": sentiment,
            "sentiment_score": round(score, 4),
            "confidence": round(confidence, 3),
            "impact_magnitude": magnitude,
            "time_horizon": horizon,
            "primary_sector": sector,
            "sector_relevance": round(relevance, 3),
            "actual_price_change_pct": round(actual_price_change, 3),
        })

    return pd.DataFrame(rows)


# ── PyTorch Neural Network ─────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class SentimentNet(nn.Module):
        """Feed-forward neural net for price impact regression."""

        def __init__(self, input_dim: int = 10, hidden_dims: list = None):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [64, 32, 16]

            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


# ── Ensemble Model ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    sector: str
    predicted_price_impact_pct: float
    confidence_interval: tuple[float, float]
    adaboost_prediction: float
    neural_net_prediction: Optional[float]
    ensemble_weight_adaboost: float
    ensemble_weight_nn: float
    feature_importances: dict[str, float]


class EnsemblePredictor:
    """
    Ensemble of AdaBoost + Neural Network for price impact prediction.
    """

    FEATURE_NAMES = [
        "sentiment_score", "confidence", "sentiment_encoded",
        "magnitude_encoded", "horizon_encoded", "sector_relevance",
        "sector_encoded", "sentiment_score_sq", "conf_x_sentiment",
        "relevance_x_confidence",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.adaboost = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        )
        self.gbm = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05,
            max_depth=4, random_state=42,
        )
        self.nn_model = None
        self.is_trained = False
        self._nn_weight = 0.4
        self._ada_weight = 0.35
        self._gbm_weight = 0.25

    def train(self, df: pd.DataFrame = None, save: bool = True) -> dict:
        """Train the ensemble on historical data."""
        if df is None:
            logger.info("Generating synthetic training data...")
            df = generate_synthetic_training_data(n_samples=8000)

        X = build_feature_matrix(df)
        y = df["actual_price_change_pct"].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── Train AdaBoost ──
        logger.info("Training AdaBoost...")
        self.adaboost.fit(X_train_scaled, y_train)
        ada_preds = self.adaboost.predict(X_test_scaled)
        ada_mae = mean_absolute_error(y_test, ada_preds)
        ada_r2 = r2_score(y_test, ada_preds)
        logger.info(f"  AdaBoost: MAE={ada_mae:.4f}, R²={ada_r2:.4f}")

        # ── Train GBM ──
        logger.info("Training GradientBoosting...")
        self.gbm.fit(X_train_scaled, y_train)
        gbm_preds = self.gbm.predict(X_test_scaled)
        gbm_mae = mean_absolute_error(y_test, gbm_preds)
        gbm_r2 = r2_score(y_test, gbm_preds)
        logger.info(f"  GBM:      MAE={gbm_mae:.4f}, R²={gbm_r2:.4f}")

        nn_mae, nn_r2 = None, None

        # ── Train Neural Network ──
        if TORCH_AVAILABLE:
            logger.info("Training PyTorch Neural Network...")
            self.nn_model = self._train_nn(X_train_scaled, y_train)

            # Eval NN
            self.nn_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_test_scaled)
                nn_preds = self.nn_model(X_t).numpy()
            nn_mae = mean_absolute_error(y_test, nn_preds)
            nn_r2 = r2_score(y_test, nn_preds)
            logger.info(f"  Neural Net: MAE={nn_mae:.4f}, R²={nn_r2:.4f}")

            # Adjust weights based on performance
            total_inv = (1/ada_mae + 1/gbm_mae + 1/nn_mae)
            self._ada_weight = (1/ada_mae) / total_inv
            self._gbm_weight = (1/gbm_mae) / total_inv
            self._nn_weight = (1/nn_mae) / total_inv

        self.is_trained = True

        if save:
            self.save()

        metrics = {
            "adaboost_mae": ada_mae, "adaboost_r2": ada_r2,
            "gbm_mae": gbm_mae, "gbm_r2": gbm_r2,
            "nn_mae": nn_mae, "nn_r2": nn_r2,
        }
        logger.info(f"Training complete. Weights: ADA={self._ada_weight:.2f}, GBM={self._gbm_weight:.2f}, NN={self._nn_weight:.2f}")
        return metrics

    def _train_nn(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50):
        model = SentimentNet(input_dim=X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            if epoch % 10 == 0:
                logger.info(f"    NN Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

        return model

    def predict(self, row: pd.Series) -> PredictionResult:
        """Predict price impact for a single article row."""
        if not self.is_trained:
            self.train()

        X = build_features(row).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        ada_pred = float(self.adaboost.predict(X_scaled)[0])
        gbm_pred = float(self.gbm.predict(X_scaled)[0])
        nn_pred = None

        if self.nn_model is not None and TORCH_AVAILABLE:
            self.nn_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled)
                nn_pred = float(self.nn_model(X_t).item())

        # Ensemble
        if nn_pred is not None:
            ensemble = (
                self._ada_weight * ada_pred +
                self._gbm_weight * gbm_pred +
                self._nn_weight * nn_pred
            )
        else:
            ensemble = 0.6 * ada_pred + 0.4 * gbm_pred

        # Confidence interval (rough ±1 std approximation)
        preds = [ada_pred, gbm_pred] + ([nn_pred] if nn_pred else [])
        std = float(np.std(preds)) if len(preds) > 1 else 0.3
        ci = (ensemble - 1.65 * std, ensemble + 1.65 * std)

        # Feature importances from AdaBoost
        importances = dict(zip(self.FEATURE_NAMES, self.adaboost.feature_importances_))

        return PredictionResult(
            sector=str(row.get("primary_sector", "Unknown")),
            predicted_price_impact_pct=round(ensemble, 3),
            confidence_interval=(round(ci[0], 3), round(ci[1], 3)),
            adaboost_prediction=round(ada_pred, 3),
            neural_net_prediction=round(nn_pred, 3) if nn_pred else None,
            ensemble_weight_adaboost=round(self._ada_weight, 3),
            ensemble_weight_nn=round(self._nn_weight, 3) if nn_pred else 0.0,
            feature_importances=importances,
        )

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch predict for all rows, adding prediction columns."""
        if not self.is_trained:
            self.train()

        predictions = []
        for _, row in df.iterrows():
            result = self.predict(row)
            predictions.append({
                "predicted_price_impact_pct": result.predicted_price_impact_pct,
                "confidence_interval_low": result.confidence_interval[0],
                "confidence_interval_high": result.confidence_interval[1],
                "adaboost_prediction": result.adaboost_prediction,
                "nn_prediction": result.neural_net_prediction,
            })

        pred_df = pd.DataFrame(predictions, index=df.index)
        return pd.concat([df, pred_df], axis=1)

    def save(self, path: str = MODEL_DIR):
        with open(os.path.join(path, "adaboost.pkl"), "wb") as f:
            pickle.dump(self.adaboost, f)
        with open(os.path.join(path, "gbm.pkl"), "wb") as f:
            pickle.dump(self.gbm, f)
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        if self.nn_model and TORCH_AVAILABLE:
            torch.save(self.nn_model.state_dict(), os.path.join(path, "nn_model.pt"))
        logger.info(f"Models saved to {path}/")

    def load(self, path: str = MODEL_DIR) -> bool:
        try:
            with open(os.path.join(path, "adaboost.pkl"), "rb") as f:
                self.adaboost = pickle.load(f)
            with open(os.path.join(path, "gbm.pkl"), "rb") as f:
                self.gbm = pickle.load(f)
            with open(os.path.join(path, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

            nn_path = os.path.join(path, "nn_model.pt")
            if os.path.exists(nn_path) and TORCH_AVAILABLE:
                self.nn_model = SentimentNet()
                self.nn_model.load_state_dict(torch.load(nn_path, map_location="cpu"))
                self.nn_model.eval()

            self.is_trained = True
            logger.info("Models loaded successfully.")
            return True
        except FileNotFoundError:
            logger.info("No saved models found, will train from scratch.")
            return False


if __name__ == "__main__":
    predictor = EnsemblePredictor()
    metrics = predictor.train()
    print("\nTraining Metrics:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Test prediction
    test_row = pd.Series({
        "sentiment": "bullish",
        "sentiment_score": 0.75,
        "confidence": 0.82,
        "impact_magnitude": "high",
        "time_horizon": "short_term",
        "primary_sector": "Information Technology",
        "sector_relevance": 0.9,
    })
    result = predictor.predict(test_row)
    print(f"\nTest prediction (bullish IT, high confidence):")
    print(f"  Predicted impact: {result.predicted_price_impact_pct:+.2f}%")
    print(f"  95% CI: [{result.confidence_interval[0]:+.2f}%, {result.confidence_interval[1]:+.2f}%]")
    print(f"  AdaBoost: {result.adaboost_prediction:+.2f}%")
    print(f"  Neural Net: {result.neural_net_prediction}")
