import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline and label encoder
model_path = 'model_pipeline.pkl'
loaded_objects = joblib.load(model_path)
clf = loaded_objects['pipeline']
le = loaded_objects['label_encoder']

# Load the dataset to get feature names and types (useful for input form)
file_path = 'Copy of CMH_OGS(1).csv'

df = pd.read_csv(file_path)

# Identify features and target
X = df.iloc[:, [0, 1, 2, 6, 13, 20, 29, 30]]
y = df.iloc[:, [28]]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Streamlit app
st.title('CMH Country prediction Model')
st.write('Enter the values for the following features to make a prediction:')

# Create input fields dynamically based on columns
input_data = {}
for col in X.columns:
    if col in numerical_cols:
        input_data[col] = st.number_input(f'Enter value for {col}', format="%.6f")
    else:
        input_data[col] = st.text_input(f'Enter value for {col}')


# Function to make predictions on new data
def make_predictions(new_data):
    new_data_df = pd.DataFrame(new_data, index=[0])

    # Convert categorical columns to strings
    for col in categorical_cols:
        new_data_df[col] = new_data_df[col].astype(str)

    # Make predictions
    predictions = clf.predict(new_data_df)

    # If the target was originally categorical, convert predictions back to original labels
    if le is not None:
        predictions = le.inverse_transform(predictions)

    return predictions


# Predict button
if st.button('Predict'):
    prediction = make_predictions(input_data)
    st.write(f'Prediction: {prediction[0]}')

# Run the Streamlit app
