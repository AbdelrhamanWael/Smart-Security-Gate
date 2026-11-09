"""
command_system.py - Speech Command Recognition & Role-Based Access Control (Bonus 1 - Member 2)
Part of: Hierarchical Multimodal Biometric Verification System

Responsibilities:
1. MFCC-based isolated-word recognizer supporting SVM and KNN with an 'unknown' class.
2. Experiment E8 runner: command classification accuracy and confusion matrix.
3. Strict Permission Table:
     - Member 1 (Super Admin): Open, Lock, Logs, Status
     - Member 2 (Operator):    Open, Lock, Status
     - Member 3 (Auditor):     Logs, Status
     - Unverified / failed:    None ("Access Denied")
4. Contract Interfaces:
     recognize_command(audio) -> str
     allowed_commands(user_id, level_reached) -> list[str]
5. Security enforcement:
     - Unverified users blocked from all actions.
     - "Permission Denied" if command not allowed for user role.
     - Session timeout management and persistent audit log (time, user, result, command).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.evaluation import plot_confusion_matrix
    from src.voice_features import extract_voice_features
except ImportError:
    from evaluation import plot_confusion_matrix
    from voice_features import extract_voice_features

COMMAND_CLASSES = ["open", "lock", "logs", "status", "unknown"]

# Role permissions definitions
ROLE_PERMISSIONS: Dict[str, Dict[str, Any]] = {
    "member1": {
        "role_name": "Super Admin",
        "commands": ["open", "lock", "logs", "status"],
        "min_level": 1,
    },
    "member2": {
        "role_name": "Operator",
        "commands": ["open", "lock", "status"],
        "min_level": 1,
    },
    "member3": {
        "role_name": "Auditor",
        "commands": ["logs", "status"],
        "min_level": 1,
    },
}

LOGS_DIR = Path("logs")
MODELS_DIR = Path("models/commands")


class CommandClassifier:
    """
    Isolated-word speech command recognizer supporting SVM and KNN classifiers.
    Recognizes: ['open', 'lock', 'logs', 'status', 'unknown']
    """

    def __init__(self, classifier_type: str = "svm"):
        assert classifier_type in ("svm", "knn"), f"Invalid type: {classifier_type}. Use 'svm' or 'knn'."
        self.classifier_type = classifier_type
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = COMMAND_CLASSES

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train classifier with feature standardization."""
        params = params or {}
        if self.classifier_type == "svm":
            clf = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
                C=params.get("C", 10.0),
                gamma=params.get("gamma", "scale"),
            )
        else:
            clf = KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "distance"),
                metric="euclidean",
            )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self.pipeline.fit(X_train, y_train)
        self.classes_ = list(self.pipeline.classes_)

    def predict(
        self,
        audio_or_features: Union[str, Path, np.ndarray],
        confidence_threshold: float = 0.45,
    ) -> Tuple[str, float]:
        """
        Recognize command from audio or feature vector.
        Returns: (predicted_command, confidence)
        """
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been trained yet!")

        if isinstance(audio_or_features, (str, Path)):
            feats = extract_voice_features(audio_or_features)
        elif isinstance(audio_or_features, np.ndarray):
            if audio_or_features.ndim == 1 and len(audio_or_features) == 56:
                feats = audio_or_features
            else:
                feats = extract_voice_features(audio_or_features)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_or_features)}")

        X = feats.reshape(1, -1)

        # Get probabilities or nearest neighbor distances
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)[0]
            max_idx = int(np.argmax(probs))
            conf = float(probs[max_idx])
            pred_class = self.classes_[max_idx]

            # If confidence is too low, fall back to "unknown"
            if conf < confidence_threshold:
                return "unknown", conf
            return pred_class, conf
        else:
            pred_class = str(self.pipeline.predict(X)[0])
            return pred_class, 1.0

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"type": self.classifier_type, "pipeline": self.pipeline, "classes": self.classes_}, str(path))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CommandClassifier":
        data = joblib.load(str(filepath))
        inst = cls(classifier_type=data["type"])
        inst.pipeline = data["pipeline"]
        inst.classes_ = data["classes"]
        return inst


# Global classifier instance
_ACTIVE_CMD_CLASSIFIER: Optional[CommandClassifier] = None


def get_active_command_classifier() -> CommandClassifier:
    global _ACTIVE_CMD_CLASSIFIER
    if _ACTIVE_CMD_CLASSIFIER is None:
        default_file = MODELS_DIR / "command_clf_svm.joblib"
        if default_file.exists():
            _ACTIVE_CMD_CLASSIFIER = CommandClassifier.load(default_file)
        else:
            _ACTIVE_CMD_CLASSIFIER = CommandClassifier(classifier_type="svm")
    return _ACTIVE_CMD_CLASSIFIER


def set_active_command_classifier(classifier: CommandClassifier) -> None:
    global _ACTIVE_CMD_CLASSIFIER
    _ACTIVE_CMD_CLASSIFIER = classifier


def recognize_command(audio: Union[str, Path, np.ndarray]) -> str:
    """
    Standard Interface Contract for Member 2 (Section 4):
    recognize_command(audio) -> str
    Returns one of: 'open', 'lock', 'logs', 'status', 'unknown'
    """
    classifier = get_active_command_classifier()
    cmd, _ = classifier.predict(audio)
    return cmd


def allowed_commands(user_id: Optional[str], level_reached: int) -> List[str]:
    """
    Standard Interface Contract for Member 2 (Section 4 & 6):
    allowed_commands(user_id, level_reached) -> list[str]

    Permission Matrix:
    - Member 1 (Super Admin): Open, Lock, Logs, Status
    - Member 2 (Operator):    Open, Lock, Status
    - Member 3 (Auditor):     Logs, Status
    - Unverified / failed:    None, "Access Denied"
    """
    if not user_id or level_reached < 1:
        return []

    user_info = ROLE_PERMISSIONS.get(user_id.lower())
    if not user_info:
        return []

    # Check minimum required biometric level
    if level_reached < user_info["min_level"]:
        return []

    return list(user_info["commands"])


class AuditLogger:
    """
    Persistent CSV & In-memory Audit Logger for security events.
    Records: timestamp, user_id, role, level_reached, command, outcome, details.
    """

    def __init__(self, log_path: Union[str, Path] = LOGS_DIR / "audit_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "user_id",
                    "role",
                    "level_reached",
                    "command",
                    "outcome",
                    "details",
                ])

    def log_event(
        self,
        user_id: str,
        role: str,
        level_reached: int,
        command: str,
        outcome: str,
        details: str = "",
    ) -> Dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp": ts,
            "user_id": user_id,
            "role": role,
            "level_reached": level_reached,
            "command": command,
            "outcome": outcome,
            "details": details,
        }
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                user_id,
                role,
                level_reached,
                command,
                outcome,
                details,
            ])
        return row

    def get_recent_logs(self, limit: int = 15) -> List[Dict[str, str]]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-limit:]


class SessionManager:
    """
    Manages authenticated biometric sessions with timeout expiration.
    Default session timeout: 60 seconds.
    """

    def __init__(self, timeout_sec: float = 60.0):
        self.timeout_sec = timeout_sec
        self.active_user: Optional[str] = None
        self.level_reached: int = 0
        self.verified_at: Optional[float] = None

    def start_session(self, user_id: str, level_reached: int) -> None:
        self.active_user = user_id
        self.level_reached = level_reached
        self.verified_at = time.time()

    def end_session(self) -> None:
        self.active_user = None
        self.level_reached = 0
        self.verified_at = None

    def is_session_valid(self) -> bool:
        if self.active_user is None or self.verified_at is None:
            return False
        elapsed = time.time() - self.verified_at
        return elapsed < self.timeout_sec

    def get_session_status(self) -> Dict[str, Any]:
        valid = self.is_session_valid()
        if not valid:
            return {"active": False, "user_id": None, "level_reached": 0, "remaining_sec": 0.0}

        elapsed = time.time() - (self.verified_at or 0.0)
        remaining = max(0.0, self.timeout_sec - elapsed)
        return {
            "active": True,
            "user_id": self.active_user,
            "level_reached": self.level_reached,
            "remaining_sec": round(remaining, 1),
        }


# Global instances
_AUDIT_LOGGER = AuditLogger()
_SESSION_MANAGER = SessionManager()


def execute_voice_command(
    audio_or_cmd: Union[str, Path, np.ndarray],
    user_id: Optional[str] = None,
    is_verified: bool = False,
    level_reached: int = 0,
) -> Dict[str, Any]:
    """
    Complete Command Execution Gate with Role-Based Security:
      1. Check authentication status: Unverified users blocked ("Access Denied").
      2. Check session validity (if applicable).
      3. Recognize speech command from audio.
      4. Check permission table: If unauthorized -> "Permission Denied".
      5. Execute action and record into audit trail.
    """
    # 1. Resolve Command
    if isinstance(audio_or_cmd, str) and audio_or_cmd.lower() in COMMAND_CLASSES:
        cmd = audio_or_cmd.lower()
    else:
        cmd = recognize_command(audio_or_cmd)

    role_info = ROLE_PERMISSIONS.get(user_id.lower()) if user_id else None
    role_name = role_info["role_name"] if role_info else "Unauthenticated"

    # 2. Gate 1: Authentication Check (Unverified / Failed users blocked)
    if not is_verified or level_reached < 1 or user_id is None:
        _AUDIT_LOGGER.log_event(
            user_id=user_id or "Anonymous",
            role=role_name,
            level_reached=level_reached,
            command=cmd,
            outcome="ACCESS_DENIED",
            details="User is unverified or authentication failed. Protected actions blocked.",
        )
        return {
            "status": "ACCESS_DENIED",
            "code": 401,
            "command": cmd,
            "message": "Access Denied: Unverified users cannot run any protected action.",
        }

    # 3. Gate 2: Unknown or unclassifiable speech
    if cmd == "unknown":
        _AUDIT_LOGGER.log_event(
            user_id=user_id,
            role=role_name,
            level_reached=level_reached,
            command="unknown",
            outcome="REJECTED_UNKNOWN",
            details="Audio could not be recognized as a valid command.",
        )
        return {
            "status": "UNKNOWN_COMMAND",
            "code": 400,
            "command": "unknown",
            "message": "Command not recognized: Out-of-vocabulary word or background noise.",
        }

    # 4. Gate 3: Authorization Check (Role permissions)
    allowed = allowed_commands(user_id, level_reached)
    if cmd not in allowed:
        _AUDIT_LOGGER.log_event(
            user_id=user_id,
            role=role_name,
            level_reached=level_reached,
            command=cmd,
            outcome="PERMISSION_DENIED",
            details=f"User '{user_id}' ({role_name}) does not have permission for '{cmd}'. Allowed: {allowed}",
        )
        return {
            "status": "PERMISSION_DENIED",
            "code": 403,
            "command": cmd,
            "allowed_commands": allowed,
            "message": f"Permission Denied: User '{user_id}' ({role_name}) is not allowed to run '{cmd.upper()}'. Allowed: {allowed}",
        }

    # 5. Execute Authorized Command Action
    action_messages = {
        "open": "SUCCESS: Gate / Vault UNLOCKED. Entry granted.",
        "lock": "SUCCESS: Gate / Vault LOCKED securely.",
        "logs": "SUCCESS: Audit logs retrieved and displayed for security inspection.",
        "status": "SUCCESS: System operational: Biometrics ACTIVE, All sensors NORMAL.",
    }
    action_msg = action_messages.get(cmd, f"Command '{cmd}' executed successfully.")

    _AUDIT_LOGGER.log_event(
        user_id=user_id,
        role=role_name,
        level_reached=level_reached,
        command=cmd,
        outcome="SUCCESS",
        details=action_msg,
    )

    return {
        "status": "SUCCESS",
        "code": 200,
        "command": cmd,
        "user_id": user_id,
        "role": role_name,
        "level_reached": level_reached,
        "message": action_msg,
    }
