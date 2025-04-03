import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

# Load data
data = pd.read_csv('synthetic_housing_risk.csv')

# Encode categorical feature
le = LabelEncoder()
data['employment_status'] = le.fit_transform(data['employment_status'])

# Features and target
X = data.drop('risk_label', axis=1)
y = data['risk_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Test probability output
probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (at risk)
print("Sample probabilities:", probs[:5])