import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

np.random.seed(42)
data_size = 400

study_hours = np.random.randint(5, 40, data_size)
attendance = np.random.randint(60, 100, data_size)
sleep_hours = np.random.randint(4, 9, data_size)

final_gpa = (
    0.03 * study_hours +
    0.02 * attendance +
    0.1 * sleep_hours +
    np.random.normal(0, 0.35, data_size)
)
final_gpa = np.clip(final_gpa, 0, 4)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "final_gpa": final_gpa
})

X = df[["study_hours", "attendance", "sleep_hours"]]
y = df["final_gpa"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Saved model as gpa_model.joblib")
print("MAE:", round(mae, 3))
print("R2:", round(r2, 3))

joblib.dump(model, "gpa_model.joblib")
