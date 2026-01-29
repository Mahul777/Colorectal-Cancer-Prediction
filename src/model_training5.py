import os
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.custom_exception_2 import CustomException
from src.logger_1 import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, processed_path=r"artifacts\processed"):
        self.processed_path = os.path.normpath(processed_path)

        self.scaler = None
        self.features = None
        self.model = None
        self.metrics = {}

        os.makedirs(self.processed_path, exist_ok=True)

    # ================= LOAD =================
    def load_artifacts(self):
        self.scaler = joblib.load(os.path.join(self.processed_path, "scaler.pkl"))
        self.features = joblib.load(os.path.join(self.processed_path, "features.pkl"))
        logger.info("Artifacts loaded successfully")

    # ================= TRAIN =================
    def train(self, X_train, Y_train):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_train, Y_train)
        logger.info("Model training completed")

    # ================= EVALUATE =================
    def evaluate(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred),
            "recall": recall_score(Y_test, y_pred),
            "f1_score": f1_score(Y_test, y_pred),
            "roc_auc": roc_auc_score(Y_test, y_proba),
            "classification_report": classification_report(Y_test, y_pred)
        }

        for k, v in self.metrics.items():
            logger.info(f"{k}: {v}")

    # ================= SAVE =================
    def save_artifacts(self):
        joblib.dump(self.model, os.path.join(self.processed_path, "model.pkl"))
        joblib.dump(self.metrics, os.path.join(self.processed_path, "metrics.pkl"))
        logger.info("Model & metrics saved")

    # ================= PIPELINE =================
    def run(self, X_train, X_test, Y_train, Y_test):
        self.load_artifacts()
        self.train(X_train, Y_train)
        self.evaluate(X_test, Y_test)
        self.save_artifacts()


if __name__ == "__main__":
    try:
        from src.data_processing4 import DataProcessor

        processor = DataProcessor(
            input_path=r"artifacts\raw\data.csv",
            output_path=r"artifacts\processed"
        )

        X_train, X_test, Y_train, Y_test = processor.run()

        trainer = ModelTrainer()
        trainer.run(X_train, X_test, Y_train, Y_test)

    except Exception as e:
        raise CustomException(e, os)




