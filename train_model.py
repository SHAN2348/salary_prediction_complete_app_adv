import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Generate synthetic data
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "experience": np.random.randint(0, 30, size=n),
    "education": np.random.choice([0, 1, 2, 3], size=n),  # 0: High School, 3: PhD
    "job_title": np.random.randint(5, 20, size=n),        # proxy: title length
    "company": np.random.randint(5, 20, size=n),
    "location": np.random.randint(5, 20, size=n),
})

# Simulate salary
df["salary"] = (
    df["experience"] * 1.5 +
    df["education"] * 4 +
    df["job_title"] * 0.5 +
    df["company"] * 0.3 +
    df["location"] * 0.2 +
    np.random.normal(0, 2, size=n)
)

# Train model
X = df.drop("salary", axis=1)
y = df["salary"]
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "salary_model.pkl")
print("✅ Model saved as salary_model.pkl")
