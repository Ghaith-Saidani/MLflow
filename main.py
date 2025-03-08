import os
import sys
import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient  # Add this import

# ✅ Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Define a local folder for storing artifacts
ARTIFACTS_DIR = "mlartifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # Ensure directory exists

def register_model(model_name, model_uri):
    """Function to register the model in MLflow Registry"""
    client = MlflowClient()
    try:
        # Register the model version if it doesn't already exist
        client.create_registered_model(model_name)
        print(f"✅ Registered model {model_name} in MLflow Model Registry.")
    except mlflow.exceptions.MlflowException:
        print(f"⚠️ Model {model_name} is already registered.")

    # Register the version of the model
    model_version = client.create_model_version(model_name, model_uri, "")
    print(f"✅ Model version {model_version.version} created for {model_name}")


def main():
    """
    Main function to execute the ML pipeline.
    This handles data preparation, training, and evaluation.
    """

    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()

    if args.prepare:
        print("📊 Exécution de prepare_data()...")
        X_processed, y_processed, scaler, pca = prepare_data()

        # ✅ Save processed data
        joblib.dump(X_processed, os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        joblib.dump(y_processed, os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        joblib.dump(pca, os.path.join(ARTIFACTS_DIR, "pca.pkl"))

        # ✅ Log to MLflow
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("standardization", True)
            mlflow.log_param("pca_components", pca.n_components_)

            # Log metrics
            mlflow.log_metric("X_processed_shape_rows", X_processed.shape[0])
            mlflow.log_metric("X_processed_shape_cols", X_processed.shape[1])

            # Log artifacts
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
            mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "pca.pkl"))

        print("✅ Données préparées et enregistrées avec succès !")

    if args.train:
        print("🚀 Entraînement du modèle...")

        # ✅ Load preprocessed data
        X_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        y_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(ARTIFACTS_DIR, "pca.pkl"))

        # ✅ Train-Test Split
        X_train, _, y_train, _ = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        with mlflow.start_run():
            # ✅ Train Model
            trained_model = train_model(X_train, y_train)

            # ✅ Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")

            # ✅ Save trained model
            print("💾 Saving model locally as model.pkl...")
            save_model(trained_model, scaler, pca)
            print("✅ model.pkl saved successfully!")

            # ✅ Ensure input example for MLflow
            input_example = np.array([X_train[0]])  # First sample as example

            # ✅ Save model to MLflow
            mlflow.sklearn.log_model(
                trained_model,
                artifact_path="mlartifacts/random_forest_model",
                input_example=input_example,
            )

            # ✅ Register model in MLflow registry
            model_name = "random_forest_model"
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/mlartifacts/random_forest_model"
            register_model(model_name, model_uri)

            print("✅ Modèle entraîné, enregistré et enregistré dans le registre MLflow")

    if args.evaluate:
        print("📊 Évaluation du modèle...")

        # ✅ Load preprocessed data
        X_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "X_processed.pkl"))
        y_processed = joblib.load(os.path.join(ARTIFACTS_DIR, "y_processed.pkl"))

        # ✅ Train-Test Split
        _, X_test, _, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        # ✅ Load trained model
        try:
            model, _, _ = load_model()
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Error: Trained model file not found. Run make train first.")
            return

        with mlflow.start_run():
            # ✅ Evaluate Model
            accuracy = evaluate_model(model, X_test, y_test)
            print(f"🔍 Accuracy: {accuracy:.4f}")

            if accuracy is not None:
                mlflow.log_metric("accuracy", float(accuracy))
                print(f"✅ Précision {accuracy:.4f} enregistrée dans MLflow")
            else:
                print("⚠️ Warning: Accuracy is None, skipping MLflow logging.")

            print(f"✅ Évaluation terminée avec une précision de {accuracy:.4f}")


if __name__ == "__main__":
    main()
