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