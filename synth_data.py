import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
    'rent': np.random.uniform(500, 2000, n_samples),
    'dependents': np.random.randint(0, 5, n_samples),
    'employment_status': np.random.choice(['employed', 'unemployed', 'part-time'], n_samples, p=[0.7, 0.2, 0.1]),
    'savings': np.random.lognormal(mean=8, sigma=1, size=n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Define risk label (binary for training)
df['income_to_rent_ratio'] = df['income'] / df['rent']
df['risk_label'] = np.where(
    (df['income_to_rent_ratio'] < 2) | (df['savings'] < 1000) | (df['employment_status'] == 'unemployed'),
    1,  # At risk
    0   # Not at risk
)

# Drop temporary column
df = df.drop('income_to_rent_ratio', axis=1)

# Save to CSV
df.to_csv('synthetic_housing_risk.csv', index=False)
print(df.head())