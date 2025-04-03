from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        required_fields = ['age', 'income', 'rent', 'dependents', 'employment_status', 'savings']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'status': 'error'}), 400

        # Convert input to DataFrame
        input_data = pd.DataFrame([data], columns=required_fields)

        # Encode employment_status
        input_data['employment_status'] = label_encoder.transform(input_data['employment_status'])

        # Make prediction (probability of being at risk)
        risk_prob = model.predict_proba(input_data)[:, 1][0]

        # Return result as JSON
        return jsonify({
            'risk_probability': float(risk_prob),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)