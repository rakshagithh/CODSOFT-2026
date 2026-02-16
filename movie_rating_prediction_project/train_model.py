import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load Dataset (Place IMDb Movies India.csv in same folder)
df = pd.read_csv("IMDb Movies India.csv", encoding="latin-1")


print("Dataset Loaded Successfully")

# Keep important columns (modify if your column names differ)
df = df[["Genre","Director","Actor 1","Actor 2","Actor 3","Rating"]]

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ["Genre","Director","Actor 1","Actor 2","Actor 3"]:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Rating", axis=1)
y = df["Rating"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

print("Model saved as model.pkl")
