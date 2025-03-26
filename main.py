from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import Optional
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffic Prediction API")

# Load pre-trained model and utilities (assuming they exist initially)
try:
    model = tf.keras.models.load_model('traffic_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    logger.warning("Pre-trained model or utilities not found. Please train the model first.")

# Define input data model for prediction endpoint
class TrafficInput(BaseModel):
    CarCount: int
    BikeCount: int
    BusCount: int
    TruckCount: int
    Hour: int

# Store uploaded data
UPLOAD_DIR = "uploaded_data"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
uploaded_data_path = os.path.join(UPLOAD_DIR, "new_traffic_data.csv")

# Placeholder preprocessing function
def preprocess_data(df, fit_scaler=True, scaler=None, label_encoder=None, is_prediction=False):
    """Preprocess data for training or prediction"""
    feature_columns = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour', 
                       'Vehicle_Density', 'Heavy_Vehicle_Ratio']
    
    if not is_prediction:
        # Assume 'Traffic_Situation' is the target column for training
        X = df[feature_columns]
        y = df['Traffic_Situation']
        
        # Fit or transform scaler
        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        # Fit or transform label encoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = label_encoder.transform(y)
        
        return X_scaled, y_encoded, scaler, label_encoder
    else:
        # For prediction, no target variable
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        return X_scaled, None, scaler, label_encoder

# Placeholder training function
def train_model(X_train, y_train, X_val, y_val):
    """Train a simple neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    return model, history

# Placeholder evaluation function
def evaluate_model(model, X_test, y_test, label_encoder, title="Confusion Matrix"):
    """Evaluate the model and print metrics"""
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = np.mean(y_pred == y_test)
    logger.info(f"{title} - Test Accuracy: {accuracy:.4f}")
    # Add confusion matrix plotting if needed (requires matplotlib)

@app.post("/predict")
async def predict_traffic_endpoint(data: TrafficInput):
    """Predict traffic situation based on input data"""
    try:
        input_data = data.dict()
        total = input_data['CarCount'] + input_data['BikeCount'] + input_data['BusCount'] + input_data['TruckCount']
        vehicle_density = total / 4
        heavy_vehicle_ratio = (input_data['BusCount'] + input_data['TruckCount']) / total if total > 0 else 0
        
        input_df = pd.DataFrame([{
            'CarCount': input_data['CarCount'],
            'BikeCount': input_data['BikeCount'],
            'BusCount': input_data['BusCount'],
            'TruckCount': input_data['TruckCount'],
            'Total': total,
            'Hour': input_data['Hour'],
            'Vehicle_Density': vehicle_density,
            'Heavy_Vehicle_Ratio': heavy_vehicle_ratio
        }])
        
        X_scaled, _, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler, 
                                          label_encoder=label_encoder, is_prediction=True)
        pred = model.predict(X_scaled).argmax(axis=1)
        prediction = label_encoder.inverse_transform(pred)[0]
        
        logger.info(f"Prediction made: {prediction}")
        return JSONResponse({
            "prediction": prediction,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/upload")
async def upload_data(file: UploadFile):
    """Upload new training data as CSV"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        with open(uploaded_data_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File uploaded: {file.filename}")
        return JSONResponse({
            "message": "File uploaded successfully",
            "filename": file.filename,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

@app.post("/retrain")
async def retrain_model():
    """Retrain model with uploaded data"""
    global model, scaler, label_encoder
    
    try:
        if not os.path.exists(uploaded_data_path):
            raise HTTPException(status_code=400, detail="No uploaded data found. Please upload data first.")
        
        new_df = pd.read_csv(uploaded_data_path)
        X, y, new_scaler, new_label_encoder = preprocess_data(new_df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        
        model, history = train_model(X_train, y_train, X_val, y_val)
        evaluate_model(model, X_test, y_test, new_label_encoder, title="Retrained Model Confusion Matrix")
        
        scaler = new_scaler
        label_encoder = new_label_encoder
        
        model.save('traffic_model.keras')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        
        logger.info("Model retrained and saved successfully")
        return JSONResponse({
            "message": "Model retrained successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "date": "2025-03-26"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")