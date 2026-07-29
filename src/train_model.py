# Requirements - pickle, pandas, scikit-learn, joblib.

# Import necessary libraries.
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load the training and testing data from the pickle file.
with open("data/train_test_split.pkl", "rb") as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

# Creating the dictionary of models.
models = {
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Creating empty list to store metrics for each model.
metrics = []

# Looping each model in the dictionary to train, evaluate, and save them.
for name, model in models.items():
    
    # Training the model.
    print(f"Training {name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Taking metrics for the model.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    metrics.append([name, mse, r2, mae])

    # Saving the trained model.
    model_path = f"results/models/{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")

# Saving the obtained metrics to a CSV file.
df = pd.DataFrame(metrics, columns=["Model", "MSE", "R2", "MAE"])
df.to_csv("results/metrics.csv", index=False)
print("Metrics saved to results/metrics.csv")