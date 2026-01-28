import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from data_processing4 import DataProcessor
from src.custom_exception_2 import CustomException
from src.logger_1 import logging
import os

class ModelTrainer:
    def __init__(self, data_file_path, model_save_path='models/'):
        """
        Initializes the ModelTrainer with the path to the dataset.
        Args:
            data_file_path (str): Path to the dataset.
            model_save_path (str): Directory path where model and scaler will be saved.
        """
        self.data_file_path = data_file_path
        self.model_save_path = model_save_path
        self.model = None
        self.scaler = None
        self.data_processor = DataProcessor(data_file_path)

    def train_model(self):
        """
        Trains the Gradient Boosting model on the dataset.
        Returns:
            GradientBoostingClassifier: Trained model.
        """
        # Load and clean data
        data = self.data_processor.load_and_clean_data()

        # Encode categorical features
        data = self.data_processor.encode_categorical_features()

        # Define target and features
        X = data.drop(columns=['Survival_Prediction'])
        Y = data['Survival_Prediction']

        # Split the dataset into Training and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        # Feature selection using Chi-Square test
        top_features = self.data_processor.feature_selection_using_chi2(X_train, Y_train)
        X_train = X_train[top_features]
        X_test = X_test[top_features]

        # Scaling the features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train Gradient Boosting model
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(Y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(Y_test, y_pred))

        # ROC AUC Score
        y_proba = self.model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, y_proba)
        print("ROC AUC Score:", roc_auc)

        # Save the trained model and scaler
        self.save_model_and_scaler()

        return self.model, self.scaler

    def save_model_and_scaler(self):
        """
        Saves the trained model and scaler to disk.
        """
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Save model
        model_path = os.path.join(self.model_save_path, 'gradient_boosting_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # Save scaler
        scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    def load_model_and_scaler(self):
        """
        Loads the trained model and scaler from disk.
        Returns:
            model, scaler: Loaded model and scaler.
        """
        model_path = os.path.join(self.model_save_path, 'gradient_boosting_model.pkl')
        scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model and Scaler loaded from {model_path} and {scaler_path}")
        else:
            raise CustomException(f"Model or Scaler not found at {self.model_save_path}")
        
        return self.model, self.scaler

    def predict_survival(self, input_data):
        """
        Predicts survival for a new input data.
        Args:
            input_data (np.array): The input data for which to predict survival.
        Returns:
            int: Predicted survival (1 for survival, 0 for non-survival).
        """
        # Ensure the model is trained first
        if self.model is None or self.scaler is None:
            raise Exception("Model is not trained yet. Please call train_model() first.")

        # Scale the input data using the trained scaler
        input_data_scaled = self.scaler.transform(input_data)

        # Make prediction
        survival_prediction = self.model.predict(input_data_scaled)

        return survival_prediction[0]

if __name__ == "__main__":
    try:
        model_trainer = ModelTrainer(data_file_path='data/colorectal_cancer_data.csv')
        model, scaler = model_trainer.train_model()
        print("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise CustomException(e)




