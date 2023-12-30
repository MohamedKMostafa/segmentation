import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'model.pkl')
scaler_file_path = os.path.join(current_dir, 'scaler.pkl')
encoder_file_path = os.path.join(current_dir, 'encoder.pkl')

# Load model, scaler, and encoder
with open(model_file_path, 'rb') as model_file, \
     open(scaler_file_path, 'rb') as scaler_file, \
     open(encoder_file_path, 'rb') as encoder_file:
    model = pickle.load(model_file)
    sc = pickle.load(scaler_file)
    le = pickle.load(encoder_file)

app = Flask(__name__)  # Initialize the Flask app

@app.route('/')  # Homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Retrieving values from form
    R = float(request.form['R'])
    F = float(request.form['F'])
    M = float(request.form['M'])
    
    # Preprocess form data similar to how training data was processed
    # (Apply necessary scaling, encoding, etc., using 'sc' and 'le')
    features = np.array([R, F, M]).reshape(1, -1)
    scaled_features = sc.transform(features)
    encoded_features = le.transform(scaled_features[:, :2])  # Assuming 2 categorical features

    # Concatenate scaled and encoded features for prediction
    processed_features = np.concatenate((scaled_features[:, 2:], encoded_features), axis=1)

    # Make prediction using the processed features
    prediction = model.predict(processed_features)
    
    return render_template('index.html', prediction_text='Predicted Segmentation: {}'.format(prediction))  # Rendering the predicted result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
