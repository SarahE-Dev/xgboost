# Housing Risk Prediction Microservice

This project is a RESTful microservice that uses an XGBoost model to predict the probability of an individual becoming unhoused based on demographic and economic data. It’s designed for structured data inputs (no free-text or LLMs) and provides a lightweight API for real-time risk assessment.

## Overview
- **Purpose**: Predict the risk of homelessness using features like age, income, rent, dependents, employment status, and savings.
- **Model**: XGBoost (gradient-boosted decision trees).
- **Output**: Probability (0 to 1) of being at risk.
- **Tech Stack**: Python, Flask, XGBoost, Pandas, Joblib.

## Features
- **API Endpoint**: `/predict` (POST) accepts JSON input and returns a risk probability.
- **Structured Data**: Handles numeric and categorical inputs (e.g., employment status).
- **Scalable**: Easy to deploy locally or in a container (e.g., Docker).

## Prerequisites
- Python 3.9+
- `pip` for installing dependencies

## Setup Instructions

### 1. Clone the Repository (or Copy Files)
If this is in a repo:
```bash
git clone <repository-url>
cd housing-risk-microservice
```
Otherwise, ensure you have the following files:
- `generate_data.py` (synthetic data script)
- `train_model.py` (model training script)
- `app.py` (microservice script)

### 2. Install Dependencies
```bash
pip install flask xgboost joblib pandas numpy sklearn
```

### 3. Generate Synthetic Data
Run the data generation script to create a sample dataset:
```bash
python generate_data.py
```
This creates `synthetic_housing_risk.csv`.

### 4. Train the Model
Train the XGBoost model and save it:
```bash
python train_model.py
```
This generates `xgboost_model.pkl` and `label_encoder.pkl`.

### 5. Run the Microservice
Start the Flask server:
```bash
python app.py
```
The service will run on `http://localhost:5000`.

## Usage

### API Endpoint: `/predict`
- **Method**: POST
- **Content-Type**: `application/json`
- **Input**: JSON object with the following fields:
  - `age` (int): Age of the individual (18-80).
  - `income` (float): Annual income in dollars.
  - `rent` (float): Monthly rent in dollars.
  - `dependents` (int): Number of dependents (0-5).
  - `employment_status` (str): One of `employed`, `unemployed`, or `part-time`.
  - `savings` (float): Total savings in dollars.
- **Output**: JSON with `risk_probability` (0 to 1) and `status`.

#### Example Request
Using `curl`:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"age": 25, "income": 25000, "rent": 1200, "dependents": 2, "employment_status": "unemployed", "savings": 500}' \
     http://localhost:5000/predict
```

Using Python:
```python
import requests
payload = {
    "age": 25,
    "income": 25000,
    "rent": 1200,
    "dependents": 2,
    "employment_status": "unemployed",
    "savings": 500
}
response = requests.post("http://localhost:5000/predict", json=payload)
print(response.json())
```

#### Example Response
```json
{
    "risk_probability": 0.85,
    "status": "success"
}
```

#### Error Response
If input is invalid:
```json
{
    "error": "Missing required fields",
    "status": "error"
}
```

## File Structure
- `generate_data.py`: Creates synthetic dataset (`synthetic_housing_risk.csv`).
- `train_model.py`: Trains the XGBoost model and saves it (`xgboost_model.pkl`, `label_encoder.pkl`).
- `app.py`: Flask microservice script.
- `synthetic_housing_risk.csv`: Sample dataset (generated).
- `xgboost_model.pkl`: Trained model file.
- `label_encoder.pkl`: Encoder for `employment_status`.

## Notes
- **Dataset**: The synthetic data uses simple rules (e.g., low income-to-rent ratio, unemployment) to label risk. Adjust `generate_data.py` for more complex logic.
- **Preprocessing**: The microservice handles categorical encoding for `employment_status`. Add scaling (e.g., `StandardScaler`) if needed.
- **Deployment**: For production, consider Docker or a cloud service (e.g., AWS, Heroku).

## License
Apache 2.0 (same as XGBoost and TensorFlow dependencies).

---
