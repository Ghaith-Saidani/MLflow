from db_utils import save_prediction
from sqlalchemy.orm import Session
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load model and preprocessors
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"
DATA_PATH = "churn_modelling.csv"  # Replace with actual dataset path

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
except FileNotFoundError:
    print("❌ Model files not found. Ensure the training pipeline has been executed.")

# Database setup - Using SQLite instead of PostgreSQL
DATABASE_URL = (
    "sqlite:///./db.sqlite"  # SQLite URL, replace with correct file path if needed
)
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)  # Added `check_same_thread` for SQLite
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define PredictionResult model
class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)
    features = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create the tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn based on input features.",
    version="1.0.0",
)


# Input schema
class InputData(BaseModel):
    Account_Length: int = Field(..., example=1)
    Area_Code: int = Field(..., example=2)
    Customer_Service_Calls: int = Field(..., example=3)
    International_Plan: int = Field(..., example=4)
    Number_of_Voicemail_Messages: int = Field(..., example=5)
    Total_Day_Calls: int = Field(..., example=6)
    Total_Day_Charge: float = Field(..., example=7.0)
    Total_Day_Minutes: float = Field(..., example=8.0)
    Total_Night_Calls: int = Field(..., example=9)
    Total_Night_Charge: float = Field(..., example=10.0)
    Total_Night_Minutes: float = Field(..., example=11.0)
    Total_Evening_Calls: int = Field(..., example=12)
    Total_Evening_Charge: float = Field(..., example=13.0)
    Total_Evening_Minutes: float = Field(..., example=14.0)
    International_Calls: int = Field(..., example=15)
    Voicemail_Plan: int = Field(..., example=16)
    Extra_Feature_1: float = Field(..., example=17.0)
    Extra_Feature_2: float = Field(..., example=18.0)
    Extra_Feature_3: float = Field(..., example=19.0)


# Response schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., example=1)
    features: Dict[str, float]  # Include features in the response


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData, db: Session = Depends(get_db)):
    try:
        # Convert input data to dictionary
        input_data = data.dict()

        # Convert input data to NumPy array
        input_features = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_features)
        input_pca = pca.transform(input_scaled)

        # Get the prediction
        prediction = model.predict(input_pca)[0]

        # Save prediction result to database
        prediction_result = PredictionResult(
            prediction=int(prediction),
            features=str(input_data),
        )
        db.add(prediction_result)
        db.commit()
        db.refresh(prediction_result)

        # Return the prediction and the input features
        return {"prediction": int(prediction), "features": input_data}

    except Exception as e:
        return {"error": str(e)}


# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
        },
    )


# Retrain endpoint schema
class Hyperparameters(BaseModel):
    n_estimators: int = Field(100, example=150)
    max_depth: int = Field(10, example=12)
    random_state: int = Field(42, example=42)


@app.post("/retrain")
def retrain(hyperparams: Hyperparameters):
    """
    Retrains the model with new hyperparameters and updates model.pkl.
    """
    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)

        # Convert categorical columns to numeric
        categorical_columns = ["State", "International plan", "Voice mail plan"]

        for col in categorical_columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col].astype("category").cat.codes
                )  # Convert categorical to numerical

        # Ensure 'Churn' is numeric
        df["Churn"] = df["Churn"].astype(int)

        # Define features and target
        X = df.drop(columns=["Churn"])  # Assuming "Churn" is the target column
        y = df["Churn"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=hyperparams.random_state
        )

        # Apply preprocessing
        X_train_scaled = scaler.transform(X_train)
        X_train_pca = pca.transform(X_train_scaled)

        # Retrain the model
        new_model = RandomForestClassifier(
            n_estimators=hyperparams.n_estimators,
            max_depth=hyperparams.max_depth,
            random_state=hyperparams.random_state,
        )
        new_model.fit(X_train_pca, y_train)

        # Save new model
        joblib.dump(new_model, MODEL_PATH)

        return {
            "message": "Model retrained successfully",
            "new_hyperparameters": hyperparams.dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
