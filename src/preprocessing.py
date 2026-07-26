# Requirements - pandas, scikit-learn, joblib, pickle.

# Importing required libraries.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import pickle

# Reading the dataset from the csv file.
df = pd.read_csv('data/house_price.csv')

# Converting the 'date' column to datetime format.
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Removing country as it has same value for all.
df = df.drop(columns='country')

# Dropping unnecessary columns that are not useful for the model.
columns_to_drop = ['date', 'yr_renovated', 'street', 'city', 'statezip']
df = df.drop(columns=columns_to_drop)

# Splitting the data into Test and Train datasets.
x = df.drop('price', axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Scaling the data using StandardScaler.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Saving the preprocessed data to csv files for future use.
df.to_csv('data/cleaned_house_price.csv', index=False)  # Saving the cleaned data to a csv file.
with open("data/train_test_split.pkl", "wb") as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
dump(scaler, "data/scaler.joblib")  # Saving the scaler object for future use in model deployment.