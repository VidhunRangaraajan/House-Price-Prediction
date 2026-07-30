# Requirements - pandas, scikit-learn, joblib, pickle.

# Importing required libraries.
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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

# Creating a list of catogorical columns, numerical columns.
categorical_columns = ['waterfront', 'view', 'condition']
numeric_columns = ["bedrooms","bathrooms","sqft_living","sqft_lot",
                "floors","sqft_above","sqft_basement","yr_built"]

#Encoding the categorical columns using OneHotEncoder and saving the encoders.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Splitting the data into Test and Train datasets.
x = df.drop('price', axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Scaling the data using StandardScaler.
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# Saving the preprocessed data.
df.to_csv('data/cleaned_house_price.csv', index=False)  # Saving the cleaned data to a csv file.
with open("data/train_test_split.pkl", "wb") as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
dump(preprocessor, "data/preprocessor.joblib")  # Saving the preprocessor object for future use in model deployment.