from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np
import os

# Load the model
model_path = r'C:\Users\chethu\Documents\ml\model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Serve CSS from templates folder
@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'templates'), filename)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return render_template('index.html', prediction_text=f'Predicted Species: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
