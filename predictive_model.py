import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert text columns to numbers
data["Brick"] = data["Brick"].map({"Yes": 1, "No": 0})
data["Neighborhood"] = data["Neighborhood"].astype("category").cat.codes

# Features and Target
X = data[["SqFt", "Bedrooms", "Bathrooms", "Offers", "Brick", "Neighborhood"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", round(score * 100, 2), "%")

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.savefig("graph.png")
plt.show()