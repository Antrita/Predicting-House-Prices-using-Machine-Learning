import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
import requests

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Extract file IDs from Google Drive URLs
# From: https://drive.google.com/file/d/1EOwg2YhKFkmqjCLwh649K1zdaAJ0o1jm/view?usp=sharing
# Extract: 1EOwg2YhKFkmqjCLwh649K1zdaAJ0o1jm
MODEL_FILE_ID = "1EOwg2YhKFkmqjCLwh649K1zdaAJ0o1jm"  # Your model file ID
SCALER_FILE_ID = "1Tsh8rx9BhXaL3-dDYqao5JC9hqACx4Gp"  # Your scaler file ID

@st.cache_resource
def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using requests"""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token for large files"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save response content to file"""
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        # Check if models exist locally, if not download from Google Drive
        if not os.path.exists('best_rf_model.pkl'):
            with st.spinner('Downloading model from Google Drive...'):
                download_file_from_google_drive(MODEL_FILE_ID, 'best_rf_model.pkl')
                st.success("Model downloaded successfully!")
        
        if not os.path.exists('scaler.pkl'):
            with st.spinner('Downloading scaler from Google Drive...'):
                download_file_from_google_drive(SCALER_FILE_ID, 'scaler.pkl')
                st.success("Scaler downloaded successfully!")
        
        # Load the models
        with st.spinner('Loading models...'):
            model = joblib.load('best_rf_model.pkl')
            scaler = joblib.load('scaler.pkl')
            st.success("Models loaded successfully!")
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the Google Drive file IDs are correct and the files are publicly accessible.")
        return None, None

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app predicts housing prices in California using a Random Forest model trained on the California Housing dataset.
The model considers various features like median income, house age, location, and more to estimate the median house value.
""")

# Initialize model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    model, scaler = load_model()
    if model is not None and scaler is not None:
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_loaded = True

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
    if not st.session_state.model_loaded:
        st.error("Model not loaded yet. Please wait or refresh the page.")
    else:
        try:
            # Get model and scaler from session state
            model = st.session_state.model
            scaler = st.session_state.scaler
            
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
            st.info("Try refreshing the page if the error persists.")

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
st.markdown("Built with Streamlit and Scikit-learn")

# Debug info (hidden by default)
with st.expander("Debug Information", expanded=False):
    st.write("Model loaded:", st.session_state.model_loaded if 'model_loaded' in st.session_state else False)
    st.write("Files exist locally:")
    st.write("- best_rf_model.pkl:", os.path.exists('best_rf_model.pkl'))
    st.write("- scaler.pkl:", os.path.exists('scaler.pkl'))
