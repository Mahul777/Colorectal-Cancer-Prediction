import os
import pandas as pd
import joblib

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.custom_exception_2 import CustomException
from src.logger_1 import get_logger

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.data = None
        self.X = None
        self.Y = None

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoders = {}
        self.selected_features = []

    # ================= LOAD =================
    def load_and_clean_data(self):
        try:
            self.data = pd.read_csv(os.path.normpath(self.input_path))

            # clean column names
            self.data.columns = (
                self.data.columns
                .str.strip()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )

            logger.info("Data loaded successfully")
            return self.data

        except Exception as e:
            raise CustomException(e, os)

    # ================= PREPROCESS =================
    def preprocess_data(self):
        try:
            num_cols = self.data.select_dtypes(include=["int64", "float64"]).columns
            self.data[num_cols] = self.data[num_cols].fillna(
                self.data[num_cols].median()
            )

            logger.info("Missing values handled")
            return self.data

        except Exception as e:
            raise CustomException(e, os)

    # ================= ENCODING =================
    def encode_categorical_features(self, target_column="Survival_Prediction"):
        try:
            categorical_cols = self.data.select_dtypes(include=["object"]).columns
            categorical_cols = [c for c in categorical_cols if c != target_column]

            for col in categorical_cols:
                if self.data[col].nunique() <= 10:
                    ohe = OneHotEncoder(drop="first")
                    transformed = ohe.fit_transform(self.data[[col]])

                    # works for OLD sklearn
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()

                    ohe_df = pd.DataFrame(
                        transformed,
                        columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]],
                        index=self.data.index
                    )

                    self.data = pd.concat(
                        [self.data.drop(columns=[col]), ohe_df],
                        axis=1
                    )

                    self.one_hot_encoders[col] = ohe

                else:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.label_encoders[col] = le

            logger.info("Categorical encoding completed")
            return self.data

        except Exception as e:
            logger.error(f"Encoding error: {e}")
            raise CustomException(e, os)

    # ================= FEATURE SELECTION =================
    def select_features(self, target_column, k=8):
        try:
            if target_column not in self.data.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(self.data.columns)}"
                )

            self.Y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])

            chi2_scores, _ = chi2(self.X, self.Y)
            feature_scores = pd.Series(chi2_scores, index=self.X.columns)

            self.selected_features = feature_scores.nlargest(k).index.tolist()
            self.X = self.X[self.selected_features]

            logger.info(f"Top {k} features selected: {self.selected_features}")
            return self.X, self.Y

        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            raise CustomException(e, os)

    # ================= SPLIT =================
    def split_data(self, test_size=0.2):
        try:
            return train_test_split(
                self.X,
                self.Y,
                test_size=test_size,
                random_state=42,
                stratify=self.Y
            )
        except Exception as e:
            raise CustomException(e, os)

    # ================= SCALE =================
    def scale_features(self, X_train, X_test):
        try:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            return X_train, X_test
        except Exception as e:
            raise CustomException(e, os)

    # ================= SAVE ARTIFACTS =================
    def save_artifacts(self):
        try:
            os.makedirs(self.output_path, exist_ok=True)

            joblib.dump(self.scaler, os.path.join(self.output_path, "scaler.pkl"))
            joblib.dump(self.selected_features, os.path.join(self.output_path, "features.pkl"))
            joblib.dump(self.label_encoders, os.path.join(self.output_path, "label_encoders.pkl"))
            joblib.dump(self.one_hot_encoders, os.path.join(self.output_path, "ohe_encoders.pkl"))

            logger.info("Artifacts saved")

        except Exception as e:
            raise CustomException(e, os)

    # ================= PIPELINE =================
    def run(self, target_column="Survival_Prediction", test_size=0.2):
        try:
            self.load_and_clean_data()
            self.preprocess_data()
            self.encode_categorical_features(target_column)

            self.select_features(target_column, k=8)

            X_train, X_test, Y_train, Y_test = self.split_data(test_size)
            X_train, X_test = self.scale_features(X_train, X_test)

            self.save_artifacts()

            logger.info("Data processing pipeline completed successfully")
            return X_train, X_test, Y_train, Y_test

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise CustomException(e, os)


# ================= MAIN =================
if __name__ == "__main__":
    processor = DataProcessor(
        input_path=r"artifacts\raw\data.csv",
        output_path=r"artifacts\processed"
    )
    processor.run()
