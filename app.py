import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Google Drive file IDs
MODEL_FILE_ID = "1EOwg2YhKFkmqjCLwh649K1zdaAJ0o1jm" 
SCALER_FILE_ID = "1Tsh8rx9BhXaL3-dDYqao5JC9hqACx4Gp"

@st.cache_resource
def download_from_gdrive(file_id, destination):
    """Download file from Google Drive"""
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    urllib.request.urlretrieve(URL, destination)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        # Check if models exist locally, if not download from Google Drive
        if not os.path.exists('best_rf_model.pkl'):
            with st.spinner('Downloading model from Google Drive...'):
                download_from_gdrive(MODEL_FILE_ID, 'best_rf_model.pkl')
        
        if not os.path.exists('scaler.pkl'):
            with st.spinner('Downloading scaler from Google Drive...'):
                download_from_gdrive(SCALER_FILE_ID, 'scaler.pkl')
        
        # Load the models
        model = joblib.load('best_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app predicts housing prices in California using a Random Forest model trained on the California Housing dataset.
The model considers various features like median income, house age, location, and more to estimate the median house value.
""")

# Sidebar for input features
st.sidebar.header("Input House Features")

# Feature input fields
col1, col2 = st.sidebar.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0.0, max_value=15.0, value=3.0, step=0.1,
                            help="Median income in tens of thousands of dollars")
    HouseAge = st.number_input("House Age", min_value=0.0, max_value=52.0, value=20.0, step=1.0,
                              help="Median house age in years")
    AveRooms = st.number_input("Average Rooms", min_value=0.0, max_value=50.0, value=5.0, step=0.1,
                              help="Average number of rooms per household")
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                               help="Average number of bedrooms per household")

with col2:
    Population = st.number_input("Population", min_value=0.0, max_value=40000.0, value=3000.0, step=100.0,
                                help="Block group population")
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, max_value=20.0, value=3.0, step=0.1,
                              help="Average number of household members")
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01,
                              help="Block group latitude")
    Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01,
                               help="Block group longitude")

# Predict button
if st.sidebar.button("Predict Price", type="primary"):
    try:
        # Load model and scaler
        model, scaler = load_model()
        
        if model is None:
            st.stop()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'MedInc': [MedInc],
            'HouseAge': [HouseAge],
            'AveRooms': [AveRooms],
            'AveBedrms': [AveBedrms],
            'Population': [Population],
            'AveOccup': [AveOccup],
            'Latitude': [Latitude],
            'Longitude': [Longitude]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted House Value", f"${prediction*100000:,.2f}")
        
        with col2:
            st.metric("Price Range", f"${(prediction-0.5)*100000:,.2f} - ${(prediction+0.5)*100000:,.2f}")
        
        with col3:
            st.metric("Model Confidence", "80.62%")
        
        # Display input summary
        st.subheader("Input Summary")
        st.dataframe(input_data)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Model information section
with st.expander("Model Information"):
    st.markdown("""
    ### Model Details
    - **Algorithm**: Random Forest Regressor (Hyperparameter Tuned)
    - **Training Data**: California Housing Dataset (20,640 samples)
    - **Features**: 8 geographic and demographic features
    - **Target**: Median house value in hundreds of thousands of dollars
    
    ### Performance Metrics
    - **R² Score**: 0.8062 (on test set)
    - **Mean Absolute Error**: $32,681
    - **Mean Squared Error**: 0.2540
    
    ### Feature Importance
    1. **Median Income**: Most important predictor
    2. **Average Occupancy**: Second most important
    3. **Longitude**: Location matters
    4. **Latitude**: Location matters
    """)

# Feature descriptions
with st.expander("Feature Descriptions"):
    st.markdown("""
    - **MedInc**: Median income for households within a block (in tens of thousands of US Dollars)
    - **HouseAge**: Median house age within a block (in years)
    - **AveRooms**: Average number of rooms among households within a block
    - **AveBedrms**: Average number of bedrooms among households within a block
    - **Population**: Total population within a block
    - **AveOccup**: Average number of household members
    - **Latitude**: Block latitude coordinate
    - **Longitude**: Block longitude coordinate
    """)

# Footer
st.markdown("---")
st.markdown("Built using Streamlit and Scikit-learn")
