from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

app = FastAPI()

# Load initial model and preprocessing objects
try:
    model = tf.keras.models.load_model('traffic_model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except:
    model, scaler, label_encoder = None, None, None

# Data Preprocessing
def preprocess_data(df, fit_scaler=True, scaler=None, label_encoder=None, is_prediction=False):
    """Preprocess the dataset: feature engineering, encoding, scaling."""
    df_processed = df.copy()

    if not is_prediction:
        df_processed['Hour'] = pd.to_datetime(df_processed['Time'], format='%I:%M:%S %p').dt.hour
        df_processed = df_processed.drop(columns=['Time', 'Date'])

    # Feature engineering
    if 'Vehicle_Density' not in df_processed.columns:
        df_processed['Vehicle_Density'] = df_processed['Total'] / 4
    if 'Heavy_Vehicle_Ratio' not in df_processed.columns:
        df_processed['Heavy_Vehicle_Ratio'] = (df_processed['BusCount'] + df_processed['TruckCount']) / df_processed['Total'].replace(0, 1)

    # Convert 'Day of the week' to numerical values
    if not is_prediction and df_processed['Day of the week'].dtype == 'object':
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df_processed['Day of the week'] = df_processed['Day of the week'].map(day_mapping)

    if not is_prediction:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            df_processed['Traffic Situation'] = label_encoder.fit_transform(df_processed['Traffic Situation'])
        else:
            df_processed['Traffic Situation'] = label_encoder.transform(df_processed['Traffic Situation'])
        X = df_processed.drop(columns=['Traffic Situation'])
        y = df_processed['Traffic Situation']
    else:
        X = df_processed
        y = None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, label_encoder

# Model Training
def train_model(X_train, y_train, X_val, y_val, pretrained_model_path=None):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model = tf.keras.models.load_model(pretrained_model_path)
    else:
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

# Model Evaluation
def evaluate_model(model, X_test, y_test, label_encoder, title="Confusion Matrix"):
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEvaluation Metrics for {title}:")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# Updated Prediction Function
def predict_traffic(model, scaler, label_encoder, input_data):
    # Calculate total
    total = input_data['CarCount'] + input_data['BikeCount'] + input_data['BusCount'] + input_data['TruckCount']
    vehicle_density = total / 4
    heavy_vehicle_ratio = (input_data['BusCount'] + input_data['TruckCount']) / total if total > 0 else 0

    # Convert Time to Hour
    time_str = input_data['Time']
    hour = pd.to_datetime(time_str, format='%H:%M').hour

    # Convert Day to numerical value
    day_mapping = {
        'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3,
        'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
    }
    day_of_week = day_mapping[input_data['Day'].upper()]

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'CarCount': input_data['CarCount'],
        'BikeCount': input_data['BikeCount'],
        'BusCount': input_data['BusCount'],
        'TruckCount': input_data['TruckCount'],
        'Total': total,
        'Hour': hour,
        'Day of the week': day_of_week,
        'Vehicle_Density': vehicle_density,
        'Heavy_Vehicle_Ratio': heavy_vehicle_ratio
    }])

    # Ensure feature order matches the scaler's expectations
    feature_order = scaler.feature_names_in_
    input_df = input_df[feature_order]

    # Preprocess the input data
    X_scaled, _, _, _ = preprocess_data(input_df, fit_scaler=False, scaler=scaler, label_encoder=label_encoder, is_prediction=True)

    # Make prediction
    pred = model.predict(X_scaled).argmax(axis=1)
    return label_encoder.inverse_transform(pred)[0]

# Updated Input Model to Match UI
class TrafficInput(BaseModel):
    CarCount: int
    BusCount: int
    BikeCount: int
    TruckCount: int
    Day: str  # e.g., "MONDAY"
    Time: str  # e.g., "00:00"

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Traffic Prediction API is running"}

# Updated Predict Endpoint
@app.post("/predict")
async def predict_traffic_endpoint(input_data: TrafficInput):
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train or upload a model first.")
    try:
        input_dict = input_data.dict()
        prediction = predict_traffic(model, scaler, label_encoder, input_dict)
        return JSONResponse(
            status_code=200,
            content={
                "prediction": prediction,
                "input_data": input_dict
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Upload new training data endpoint
@app.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")
        
        df = pd.read_csv(file.file)
        required_columns = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 
                          'Time', 'Date', 'Day of the week', 'Traffic Situation']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSV missing required columns")
        
        upload_path = "uploaded_data.csv"
        df.to_csv(upload_path, index=False)
        
        return JSONResponse(
            status_code=200,
            content={"message": "Data uploaded successfully", "filename": file.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

# Retrain endpoint
@app.post("/retrain")
async def retrain_model():
    global model, scaler, label_encoder
    try:
        upload_path = "uploaded_data.csv"
        if not os.path.exists(upload_path):
            raise HTTPException(status_code=404, detail="No uploaded data found for retraining")
        
        df = pd.read_csv(upload_path)
        X, y, new_scaler, new_label_encoder = preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        
        model, history = train_model(
            X_train, y_train, X_val, y_val,
            pretrained_model_path='traffic_model.keras' if os.path.exists('traffic_model.keras') else None
        )
        
        accuracy = evaluate_model(model, X_test, y_test, new_label_encoder)
        model.save('traffic_model.keras')
        
        scaler = new_scaler
        label_encoder = new_label_encoder
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model retrained successfully",
                "test_accuracy": accuracy
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
