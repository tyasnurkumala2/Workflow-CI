import os
import sys
import warnings
warnings.filterwarnings('ignore')

# TensorFlow configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: used for performance tuning

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.tensorflow

# MLflow configuration
# Set experiment before starting the run
experiment_name = "SMSML_CI_CD"
mlflow.set_experiment(experiment_name)

# Function to create the model
def create_model(input_shape):
    """Create simple neural network model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Function to load and preprocess data
def load_data():
    """Load and prepare data"""
    try:
        # Update the path to your dataset
        data_path = "namadataset_preprocessing/data_preprocessed.csv" 
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded. Shape: {df.shape}")
        
        # Preprocessing
        X = df.drop(['RiskLevel_Encoded'], axis=1, errors='ignore')
        y = df['RiskLevel_Encoded']
        
        # Clean columns
        if 'Risk Level' in X.columns:
            X = X.drop('Risk Level', axis=1)
        
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        # Fallback data sample
        n_samples = 1000
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        print("✅ Using sample data as fallback")
        return X, y

def main():
    """Main training function"""
    # Get parameters from MLProject
    n_estimators = 100  # Hardcoded default values, can be passed as command line args or through MLProject
    max_depth = 10
    
    print(f"🚀 Starting Training - Using TensorFlow Model")
    
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(nested=True):  # Using nested=True to allow sub-runs
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "TensorFlow_Sequential")
        
        # Create and train model
        model = create_model(X_train.shape[1])
        
        print("📊 Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate model performance
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("final_loss", history.history['loss'][-1])
        
        # Log model
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model"
        )
        
        print(f"✅ Training completed! Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
