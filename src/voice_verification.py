"""
voice_verification.py - Voice-Only Biometric Verifier (Mode B - Member 2)
Part of: Hierarchical Multimodal Biometric Verification System

Responsibilities:
1. Per-user Genuine vs. Impostor binary verification.
2. Models: SVM (RBF kernel) and Random Forest (Experiment E3).
3. Hyperparameter tuning using Grouped Cross-Validation (by session).
4. Interface Contract implementation:
     voice_score(audio, user_id) -> float  # Match probability 0.0 to 1.0
5. Evaluation & comparison (SVM vs. RF): FAR, FRR, EER, AUC, ROC, Confusion Matrix.
6. Strictly traditional ML (No Deep Learning).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.evaluation import evaluate, plot_confusion_matrix, plot_roc_curve, plot_score_distributions
    from src.voice_features import extract_voice_features
except ImportError:
    from evaluation import evaluate, plot_confusion_matrix, plot_roc_curve, plot_score_distributions
    from voice_features import extract_voice_features

DEFAULT_USERS = ["member1", "member2", "member3"]
MODELS_DIR = Path("models/voice")


class VoiceVerifier:
    """
    Per-user Voice Verification System supporting SVM (RBF) and Random Forest.
    """

    def __init__(self, model_type: str = "svm"):
        assert model_type in ("svm", "rf"), f"Invalid model_type: {model_type}. Choose 'svm' or 'rf'."
        self.model_type = model_type
        # user_id -> fitted Pipeline (StandardScaler + Classifier)
        self.user_models: Dict[str, Pipeline] = {}
        # user_id -> best hyperparameters and CV metrics
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        # user_id -> operational decision threshold (default 0.5)
        self.thresholds: Dict[str, float] = {u: 0.5 for u in DEFAULT_USERS}

    def _create_base_estimator(self, params: Dict[str, Any]) -> Pipeline:
        """Create a pipeline with StandardScaler and the classifier."""
        if self.model_type == "svm":
            # C=2.0 with gamma='scale' and no class_weight inflation
            clf = SVC(
                kernel="rbf",
                probability=True,
                class_weight=None,
                random_state=42,
                C=params.get("C", 2.0),
                gamma=params.get("gamma", "scale"),
            )
        else:
            clf = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                n_jobs=-1,
            )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def tune_and_train_user(
        self,
        user_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning using GroupKFold (grouped by session)
        to prevent session-based data leakage.
        """
        if self.model_type == "svm":
            param_grid = [
                {"C": 1.0, "gamma": "scale"},
                {"C": 2.0, "gamma": "scale"},
                {"C": 5.0, "gamma": "scale"},
                {"C": 10.0, "gamma": "scale"},
            ]
        else:
            param_grid = [
                {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
                {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
                {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
                {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
            ]

        # Determine number of CV splits based on distinct sessions
        unique_groups = np.unique(groups_train)
        n_splits = max(2, min(len(unique_groups), 3))
        gkf = GroupKFold(n_splits=n_splits)

        best_score = -1.0
        best_params = param_grid[0]

        if verbose:
            print(f"[{self.model_type.upper()}] Tuning {user_id} with {n_splits}-fold Session GroupKFold...")

        for params in param_grid:
            fold_aucs = []
            for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
                X_tr, y_tr = X_train[train_idx], y_train[train_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]

                # Check if validation fold has both classes
                if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
                    continue

                pipe = self._create_base_estimator(params)
                pipe.fit(X_tr, y_tr)
                scores = pipe.predict_proba(X_val)[:, 1]

                metrics = evaluate(y_val, scores)
                fold_aucs.append(metrics["AUC"])

            mean_auc = np.mean(fold_aucs) if fold_aucs else 0.5
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

        if verbose:
            print(f"  [+] Best params for {user_id}: {best_params} (CV AUC: {best_score:.4f})")

        # Fit final model on all training data with best params
        final_pipe = self._create_base_estimator(best_params)
        final_pipe.fit(X_train, y_train)

        self.user_models[user_id] = final_pipe
        self.model_metadata[user_id] = {
            "best_params": best_params,
            "cv_auc": float(best_score),
            "n_train_samples": len(X_train),
        }

        return self.model_metadata[user_id]

    def predict_score(
        self,
        audio_or_features: Union[str, Path, np.ndarray],
        user_id: str,
    ) -> float:
        """
        Interface contract:
        voice_score(audio, user_id) -> float
        Returns match probability in range [0.0, 1.0].
        """
        if user_id not in self.user_models:
            raise KeyError(
                f"User '{user_id}' not enrolled in verifier! Enrolled: {list(self.user_models.keys())}"
            )

        model = self.user_models[user_id]

        if isinstance(audio_or_features, (str, Path)):
            audio_data, sr = None, None
            try:
                from src.voice_features import load_audio, normalize_amplitude
                audio_data, sr = load_audio(audio_or_features)
            except Exception:
                pass
            if audio_data is not None:
                peak = float(np.max(np.abs(audio_data)))
                if peak < 0.005:  # Dead silence / no microphone signal
                    return 0.0
                # Normalize gain so low mic volumes match training level perfectly
                audio_data = normalize_amplitude(audio_data, target_peak=0.95)
                features = extract_voice_features(audio_data)
            else:
                features = extract_voice_features(audio_or_features)
        elif isinstance(audio_or_features, np.ndarray):
            if audio_or_features.ndim == 1 and len(audio_or_features) == 56:
                features = audio_or_features
            else:
                peak = float(np.max(np.abs(audio_or_features)))
                if peak < 0.005:  # Dead silence / no microphone signal
                    return 0.0
                from src.voice_features import normalize_amplitude
                norm_audio = normalize_amplitude(audio_or_features, target_peak=0.95)
                features = extract_voice_features(norm_audio)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_or_features)}")

        # Reshape to 2D
        X = features.reshape(1, -1)
        # Class 1 probability (genuine)
        prob = float(model.predict_proba(X)[0, 1])
        return prob

    def evaluate_on_test_set(
        self,
        user_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate model on unseen test set (session 3 + unseen impostors).
        """
        model = self.user_models[user_id]
        scores = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, scores, threshold=threshold)
        return metrics

    def save(self, directory: Union[str, Path] = MODELS_DIR) -> None:
        """Serialize models to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model_type": self.model_type,
            "user_models": self.user_models,
            "model_metadata": self.model_metadata,
            "thresholds": self.thresholds,
        }
        out_path = target_dir / f"voice_verifier_{self.model_type}.joblib"
        joblib.dump(save_data, str(out_path))
        print(f"Voice verifier saved to {out_path.resolve()}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "VoiceVerifier":
        """Deserialize models from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(str(path))
        verifier = cls(model_type=data["model_type"])
        verifier.user_models = data["user_models"]
        verifier.model_metadata = data["model_metadata"]
        verifier.thresholds = data.get("thresholds", {u: 0.5 for u in DEFAULT_USERS})
        return verifier


# Global instance for quick interface access
_ACTIVE_VERIFIER: Optional[VoiceVerifier] = None


def get_active_verifier() -> VoiceVerifier:
    global _ACTIVE_VERIFIER
    if _ACTIVE_VERIFIER is None:
        # Attempt to load saved SVM model
        default_path = MODELS_DIR / "voice_verifier_svm.joblib"
        if default_path.exists():
            _ACTIVE_VERIFIER = VoiceVerifier.load(default_path)
        else:
            _ACTIVE_VERIFIER = VoiceVerifier(model_type="svm")
    return _ACTIVE_VERIFIER


def set_active_verifier(verifier: VoiceVerifier) -> None:
    global _ACTIVE_VERIFIER
    _ACTIVE_VERIFIER = verifier


def voice_score(
    audio: Union[str, Path, np.ndarray],
    user_id: str,
) -> float:
    """
    Standard Interface Contract for Member 2 (Section 4):
    voice_score(audio, user_id) -> float
    Match probability between 0.0 and 1.0.
    """
    verifier = get_active_verifier()
    return verifier.predict_score(audio, user_id)
