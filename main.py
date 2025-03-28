from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffic Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and utilities
model = None
scaler = None
label_encoder = None

# Load pre-trained model and utilities
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

# Preprocessing function
def preprocess_data(df, fit_scaler=True, scaler=None, label_encoder=None, is_prediction=False):
    """Preprocess data for training or prediction"""
    feature_columns = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour',
                       'Vehicle_Density', 'Heavy_Vehicle_Ratio']
    
    # Check for required columns
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features and is_prediction:
        raise ValueError(f"Missing feature columns for prediction: {missing_features}")
    
    if not is_prediction:
        if 'Traffic Situation' not in df.columns:  # Updated to match space
            raise ValueError("Missing 'Traffic Situation' column in training data")
        X = df[feature_columns]
        y = df['Traffic Situation']  # Updated to match space
        
        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = label_encoder.transform(y)
        
        return X_scaled, y_encoded, scaler, label_encoder
    else:
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        return X_scaled, None, scaler, label_encoder

# Train model function
def train_model(X_train, y_train, X_val, y_val):
    """Train the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val), 
                       epochs=100, 
                       batch_size=32, 
                       callbacks=[early_stopping], 
                       verbose=1)
    return model, history

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

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Retrain the model with an uploaded dataset and return evaluation metrics"""
    global model, scaler, label_encoder
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read uploaded CSV file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Convert bytes to DataFrame
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")

        # Feature engineering
        required_columns = ['Time', 'Total', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Traffic Situation']  # Updated to match space
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
        if df['Hour'].isna().any():
            raise HTTPException(status_code=400, detail="Invalid time format in 'Time' column")
        
        df['Vehicle_Density'] = df['Total'] / 4
        df['Heavy_Vehicle_Ratio'] = (df['BusCount'] + df['TruckCount']) / df['Total'].replace(0, 1)
        df = df.drop(columns=['Time', 'Date', 'Day of the week'], errors='ignore')  # Ignore missing columns

        # Preprocess data
        X, y, new_scaler, new_label_encoder = preprocess_data(df, fit_scaler=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        
        # Train model
        new_model, history = train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        y_pred = new_model.predict(X_test).argmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # Update global model and utilities
        model = new_model
        scaler = new_scaler
        label_encoder = new_label_encoder
        
        # Save updated model and utilities
        model.save('traffic_model.keras')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        
        logger.info("Model retrained successfully")
        return JSONResponse({
            "status": "success",
            "metrics": metrics,
            "message": "Model retrained and saved successfully"
        })
        
    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file uploaded")
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty or invalid")
    except pd.errors.ParserError:
        logger.error("CSV parsing error")
        raise HTTPException(status_code=400, detail="Error parsing CSV file")
    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
