import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Generate fake student data
np.random.seed(42)
data_size = 200

study_hours = np.random.randint(5, 40, data_size)
attendance = np.random.randint(60, 100, data_size)
sleep_hours = np.random.randint(4, 9, data_size)

final_gpa = (
    0.03 * study_hours +
    0.02 * attendance +
    0.1 * sleep_hours +
    np.random.normal(0, 0.3, data_size)
)

final_gpa = np.clip(final_gpa, 0, 4)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "final_gpa": final_gpa
})

print("First 5 rows of data:")
print(df.head())

# Features & target
X = df[["study_hours", "attendance", "sleep_hours"]]
y = df["final_gpa"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))

# Predict example student
example_student = pd.DataFrame({
    "study_hours": [25],
    "attendance": [90],
    "sleep_hours": [7]
})

predicted_gpa = model.predict(example_student)

print("\nPredicted GPA for example student:", round(predicted_gpa[0], 2))
