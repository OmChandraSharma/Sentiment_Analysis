import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("clean_data.csv")
df = df.dropna(subset=["clean_text", "sentiment"])

# Load pre-fitted TF-IDF vectorizer and label encoder
tfidf_vectorizer = joblib.load("vectorizers/tfidf_vectorizer.pkl")
label_encoder = joblib.load("vectorizers/label_encoder.pkl")

# Transform inputs and labels
X = tfidf_vectorizer.transform(df["clean_text"])
y = label_encoder.transform(df["sentiment"])

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "linearregressiontf-idf.pkl")

# Optional: Evaluation on training data
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("✅ Model trained and saved as linearregressiontf-idf.pkl")
print(f"📉 MSE: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")
