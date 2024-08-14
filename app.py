import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print(joblib.__version__)
# Load the model using joblib

try:
    model = joblib.load('C:/Users/klbai/OneDrive/Desktop/Capstone/WalmartFinal2/gradient_boosting_model_80_20.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Verify model type
if not hasattr(model, 'predict'):
    st.error("Model does not have 'predict' method. Check the model file.")
    st.stop()

# Define holidays and their dates
holidays = {
    'Super Bowl': ['12-Feb-10', '11-Feb-11', '10-Feb-12'],
    'Labour Day': ['10-Sep-10', '9-Sep-11', '7-Sep-12'],
    'Thanksgiving': ['26-Nov-10', '25-Nov-11', '23-Nov-12'],
    'Christmas': ['31-Dec-10', '30-Dec-11', '28-Dec-12']
}

# Set up Streamlit app
st.title('Walmart Sales Prediction Dashboard')

# Custom CSS for slider color - fallback approach
st.markdown("""
    <style>
    .stSlider .stSlider-handle {
        background-color: green;
        border: 2px solid darkgreen;
    }
    .stSlider .stSlider-track {
        background-color: lightgreen;
    }
    .stSlider .stSlider-rail {
        background-color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# User input for holiday
holiday = st.selectbox('Select a Holiday', list(holidays.keys()))
date_str = st.selectbox('Select a Date', holidays[holiday])

# Convert selected date to datetime
date = datetime.strptime(date_str, '%d-%b-%y')

# User inputs for prediction
store_id = st.slider('Select Store ID', min_value=1, max_value=45, value=1)
cpi = st.number_input('Enter CPI', min_value=0.0, value=120.0)
unemployment = st.number_input('Enter Unemployment Rate (%)', min_value=0.0, value=8.0)

# Generate input data for prediction
input_data = pd.DataFrame({
    'Store': [store_id],
    'CPI': [cpi],
    'Unemployment': [unemployment],
    'Month': [date.month],
    'WeekOfYear': [date.isocalendar()[1]]
})

# Make predictions
try:
    predictions = model.predict(input_data)
    st.header('Predictions for Selected Holiday By Store')
    st.write(f'Predicted Weekly Sales for {holiday} on {date_str}: ${predictions[0]:,.2f}')
except Exception as e:
    st.error(f"Prediction error: {e}")

# Plot historical sales data
st.header('Original Actual vs. Predicted Plot')
historical_data = pd.read_csv('C:/Users/klbai/OneDrive/Desktop/Capstone/WalmartFinal2/predictions_with_actuals.csv')

# Aggregate historical sales data by store
store_agg = historical_data.groupby('Store').agg({
    'Weekly_Sales': 'mean',
    'Predicted_Weekly_Sales': 'mean'
}).reset_index()

# Rename columns for clarity
store_agg.rename(columns={
    'Weekly_Sales': 'Actual_Sales',
    'Predicted_Weekly_Sales': 'Predicted_Sales'
}, inplace=True)

# Create a bar plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=store_agg.melt(id_vars='Store', var_name='Sales_Type', value_name='Sales'),
            x='Store', y='Sales', hue='Sales_Type',
            palette={'Actual_Sales': 'lightblue', 'Predicted_Sales': 'teal'}, ax=ax)
ax.set_xlabel('Store')
ax.set_ylabel('Average Sales in Millions (USD)')
ax.set_title('Average Actual vs. Predicted Weekly Sales by Store')
ax.tick_params(axis='x', rotation=-45)
ax.legend(title='Sales Type')
ax.grid(True)
plt.tight_layout()

st.pyplot(fig)

